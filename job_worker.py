from __future__ import annotations

import json
import logging
import queue
import threading
import time
from pathlib import Path

from auth import AuthStore, JobRecord
from service import ConversionError, ConversionService

log = logging.getLogger("job_worker")


def _job_log_fields(job_id: str, rec: JobRecord | None) -> tuple[str, str, str]:
    """用于日志：job_id、用户名、展示用文件名。"""
    if not rec:
        return job_id, "?", "?"
    name = rec.original_filename or Path(rec.input_file).name
    user = rec.owner_username or "?"
    return job_id, user, name


class JobQueueWorker:
    """单进程后台线程：从内存队列取 job_id，认领 DB 中 queued 任务并执行转换。"""

    def __init__(
        self,
        auth_store: AuthStore,
        conversion_service: ConversionService,
    ) -> None:
        self.auth = auth_store
        self.service = conversion_service
        self._q: queue.Queue[str] = queue.Queue()
        self._thread = threading.Thread(
            target=self._loop, name="docling-job-worker", daemon=True
        )
        self._thread.start()

    def enqueue(self, job_id: str) -> None:
        self._q.put(job_id)

    def requeue_queued_from_db(self) -> int:
        """服务启动后将库中排队任务重新放入内存队列（与持久化一致）。"""
        ids = self.auth.list_queued_job_ids()
        for jid in ids:
            self._q.put(jid)
        return len(ids)

    def _loop(self) -> None:
        while True:
            job_id = self._q.get()
            try:
                self._process_one(job_id)
            except Exception:  # noqa: BLE001
                log.exception("任务 worker 异常 job_id=%s", job_id)
            finally:
                self._q.task_done()

    def _process_one(self, job_id: str) -> None:
        if not self.auth.try_claim_job_running(job_id):
            return

        rec = self.auth.get_job(job_id)
        if not rec or not rec.output_file:
            jid, u, fn = _job_log_fields(job_id, rec)
            log.warning(
                "任务记录不完整 job_id=%s user=%s file=%s",
                jid,
                u,
                fn,
            )
            self.auth.mark_job_failed(job_id, "内部错误：任务记录不完整")
            return

        inp = Path(rec.input_file)
        out = Path(rec.output_file)
        jid, user, fname = _job_log_fields(job_id, rec)
        log.info(
            "阶段=开始执行 job_id=%s user=%s file=%s input=%s output=%s",
            jid,
            user,
            fname,
            inp.name,
            out.name,
        )

        if not inp.is_file():
            log.error(
                "任务输入文件不存在 job_id=%s user=%s file=%s path=%s",
                jid,
                user,
                fname,
                inp,
            )
            self.auth.mark_job_failed(job_id, "输入文件不存在")
            return

        stop_pulse = threading.Event()

        def on_pdf_pages(done: int, total: int) -> None:
            t = max(int(total), 1)
            d = max(0, min(int(done), t))
            pct = int(round(100.0 * d / t))
            pct = max(0, min(100, pct))
            # 对于 pdf-vl-primary：progress_callback 在「逐页转写」结束时才会回调到 done==total。
            # 但逐页转写完成后仍可能继续进行 LLM refine / 表格说明 / 图片说明等后处理，
            # 若直接显示 100% 会导致用户误以为“已经全部完成”。
            if d >= t and pct >= 100:
                pct = 98
            note = f"PDF 已完成 {d}/{t} 页"
            if d >= t:
                note = "PDF 页码已完成，正在后处理…"
            self.auth.update_job_progress(
                job_id,
                percent=pct,
                note=note,
                pages_done=d,
                pages_total=t,
            )
            log.info(
                "阶段=页级进度 job_id=%s user=%s file=%s pdf_pages=%s/%s progress_percent=%s",
                jid,
                user,
                fname,
                d,
                t,
                pct,
            )

        # 始终保留 pulse 线程：当存在页级进度回调时（progress_pages_total != None）
        # 会先“原样等待”；当页码回调已完成（progress_pages_done >= progress_pages_total）时，
        # 用少量增长表示后处理仍在进行，避免前端长期停在 98/100。
        skip_pulse = False

        def pulse_loop() -> None:
            t0 = time.monotonic()
            while not stop_pulse.wait(2.0):
                j = self.auth.get_job(job_id)
                if not j or j.status != "running":
                    return
                if j.progress_pages_total is not None:
                    # pdf-vl-primary：逐页回调存在
                    if (
                        j.progress_pages_done is not None
                        and j.progress_pages_total is not None
                        and j.progress_pages_done >= j.progress_pages_total
                    ):
                        # 页码已完成：仅在没到 99 的情况下做一次后处理进度补偿
                        cur = j.progress_percent
                        if cur is not None and cur >= 99:
                            continue
                        self.auth.update_job_progress(
                            job_id,
                            percent=99,
                            note="PDF 页码已完成，正在后处理…",
                        )
                        log.debug(
                            "阶段=后处理补偿 job_id=%s user=%s file=%s pulse_percent=%s",
                            jid,
                            user,
                            fname,
                            99,
                        )
                    continue

                # 已有页级字段时不应走估算支路（避免与 on_pdf_pages 交错写入时把 note 盖成「正在转换文档…」）
                if j.progress_pages_done is not None or j.progress_pages_total is not None:
                    continue

                # 非页级回调场景：使用估算进度
                elapsed = time.monotonic() - t0
                fake = min(88, 2 + int(elapsed * 1.15))
                cur = j.progress_percent
                if cur is not None and cur >= fake:
                    continue
                self.auth.update_job_progress(job_id, percent=fake, note="正在转换文档…")
                log.debug(
                    "阶段=估算进度 job_id=%s user=%s file=%s pulse_percent=%s",
                    jid,
                    user,
                    fname,
                    fake,
                )

        pulse_thread: threading.Thread | None = None
        pulse_thread = threading.Thread(
            target=pulse_loop,
            name=f"job-progress-{job_id[:8]}",
            daemon=True,
        )
        pulse_thread.start()
        try:
            conv_result = self.service.convert_to_markdown(
                str(inp), str(out), progress_callback=on_pdf_pages
            )
        except ConversionError as exc:
            log.error(
                "阶段=失败(业务) job_id=%s user=%s file=%s err=%s",
                jid,
                user,
                fname,
                exc,
            )
            self.auth.mark_job_failed(job_id, str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            log.exception(
                "阶段=失败(异常) job_id=%s user=%s file=%s",
                jid,
                user,
                fname,
            )
            self.auth.mark_job_failed(job_id, f"转换失败: {exc!s}"[:4000])
            return
        finally:
            stop_pulse.set()

        result_extra: str | None = None
        if conv_result.pdf_vl_failed_pages:
            result_extra = json.dumps(
                {"pdf_vl_failed_pages": list(conv_result.pdf_vl_failed_pages)},
                ensure_ascii=False,
            )

        latest = self.auth.get_job(job_id)
        if not latest or latest.status != "running":
            if out.exists():
                try:
                    out.unlink()
                except OSError:
                    pass
            return
        if latest.cancel_requested:
            self.auth.mark_job_cancelled_finished(job_id, "已取消")
            if out.exists():
                try:
                    out.unlink()
                except OSError:
                    pass
            return

        ok = self.auth.mark_job_succeeded(
            job_id, str(out.resolve()), result_extra=result_extra
        )
        if ok:
            log.info(
                "阶段=完成 job_id=%s user=%s file=%s output=%s",
                jid,
                user,
                fname,
                out.name,
            )
            if conv_result.pdf_vl_failed_pages:
                log.warning(
                    "阶段=完成(部分页失败) job_id=%s pages=%s",
                    jid,
                    list(conv_result.pdf_vl_failed_pages),
                )
        else:
            log.info(
                "阶段=完成未落库(可能已取消) job_id=%s user=%s file=%s",
                jid,
                user,
                fname,
            )
            if out.exists():
                try:
                    out.unlink()
                except OSError:
                    pass
