from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path

from auth import AuthStore
from service import ConversionError, ConversionService

log = logging.getLogger("job_worker")


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
            self.auth.mark_job_failed(job_id, "内部错误：任务记录不完整")
            return

        inp = Path(rec.input_file)
        out = Path(rec.output_file)
        if not inp.is_file():
            self.auth.mark_job_failed(job_id, "输入文件不存在")
            return

        stop_pulse = threading.Event()

        def on_pdf_pages(done: int, total: int) -> None:
            t = max(int(total), 1)
            d = max(0, min(int(done), t))
            pct = int(round(100.0 * d / t))
            pct = max(0, min(100, pct))
            self.auth.update_job_progress(
                job_id,
                percent=pct,
                note=f"PDF 已完成 {d}/{t} 页",
                pages_done=d,
                pages_total=t,
            )

        skip_pulse = inp.suffix.lower() == ".pdf" and bool(
            self.service.app_config.pdf_vl_primary
        )

        def pulse_loop() -> None:
            t0 = time.monotonic()
            while not stop_pulse.wait(2.0):
                j = self.auth.get_job(job_id)
                if not j or j.status != "running":
                    return
                if j.progress_pages_total is not None:
                    continue
                elapsed = time.monotonic() - t0
                fake = min(88, 2 + int(elapsed * 1.15))
                cur = j.progress_percent
                if cur is not None and cur >= fake:
                    continue
                self.auth.update_job_progress(
                    job_id, percent=fake, note="正在转换文档…"
                )

        pulse_thread: threading.Thread | None = None
        if not skip_pulse:
            pulse_thread = threading.Thread(
                target=pulse_loop,
                name=f"job-progress-{job_id[:8]}",
                daemon=True,
            )
            pulse_thread.start()
        try:
            self.service.convert_to_markdown(
                str(inp), str(out), progress_callback=on_pdf_pages
            )
        except ConversionError as exc:
            self.auth.mark_job_failed(job_id, str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            log.exception("转换失败 job_id=%s", job_id)
            self.auth.mark_job_failed(job_id, f"转换失败: {exc!s}"[:4000])
            return
        finally:
            stop_pulse.set()

        latest = self.auth.get_job(job_id)
        if latest and latest.cancel_requested:
            self.auth.mark_job_cancelled_finished(job_id, "已取消")
            if out.exists():
                try:
                    out.unlink()
                except OSError:
                    pass
            return

        self.auth.mark_job_succeeded(job_id, str(out.resolve()))
