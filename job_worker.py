from __future__ import annotations

import json
import logging
import multiprocessing as mp
import queue
import threading
import time
from pathlib import Path

from auth import AuthStore, JobRecord
from config import AppConfig
from service import ConversionError, ConversionService
from src.logging_utils import log_event, short_job_id

log = logging.getLogger("job_worker")


def _job_log_fields(job_id: str, rec: JobRecord | None) -> tuple[str, str, str]:
    """用于日志：job_id、用户名、展示用文件名。"""
    if not rec:
        return job_id, "?", "?"
    name = rec.original_filename or Path(rec.input_file).name
    user = rec.owner_username or "?"
    return job_id, user, name


def _run_conversion_in_subprocess(
    *,
    job_id: str,
    auth_db_path_str: str,
    input_file: str,
    output_file: str,
) -> None:
    """
    子进程任务执行入口：
    - 重新构造 ConversionService（避免跨进程共享重对象）
    - 仅在 DB 状态仍为 running 时写入 succeeded/failed（避免覆盖用户取消）
    """
    # 子进程内尽量延迟初始化，降低 spawn 成本。
    config = AppConfig.from_env()
    service = ConversionService(config)
    auth = AuthStore(Path(auth_db_path_str))

    inp = Path(input_file)
    out = Path(output_file)

    # 进度写入：pdf-vl-primary 有页级回调；其它路径只写 note（严格真实进度模式下也不会乱写百分比）。
    def on_pdf_pages(done: int, total: int) -> None:
        # 子进程内协作取消：若任务已不再 running，则立刻抛错中断后续流程
        j = auth.get_job(job_id)
        if not j or j.status != "running":
            raise RuntimeError("任务已取消")
        t = max(int(total), 1)
        d = max(0, min(int(done), t))
        pct = int(round(100.0 * d / t))
        pct = max(0, min(100, pct))
        if d >= t and pct >= 100:
            pct = 98
        note = f"PDF 已完成 {d}/{t} 页"
        if d >= t:
            note = "PDF 页码已完成，正在后处理…"
        auth.update_job_progress(
            job_id,
            percent=pct,
            note=note,
            pages_done=d,
            pages_total=t,
        )

    if not inp.is_file():
        auth.mark_job_failed(job_id, "输入文件不存在")
        return

    def cancel_check() -> bool:
        j = auth.get_job(job_id)
        return (not j) or j.status != "running"

    try:
        conv_result = service.convert_to_markdown(
            str(inp),
            str(out),
            progress_callback=on_pdf_pages,
            cancel_check=cancel_check,
        )
    except ConversionError as exc:
        auth.mark_job_failed(job_id, str(exc))
        return
    except Exception as exc:  # noqa: BLE001
        auth.mark_job_failed(job_id, f"转换失败: {exc!s}"[:4000])
        return

    result_extra: str | None = None
    if conv_result.pdf_vl_failed_pages:
        result_extra = json.dumps(
            {"pdf_vl_failed_pages": list(conv_result.pdf_vl_failed_pages)},
            ensure_ascii=False,
        )

    # 若已被取消/终态，不写入成功并清理输出，避免“取消了但生成了结果”。
    latest = auth.get_job(job_id)
    if not latest or latest.status != "running":
        if out.exists():
            try:
                out.unlink()
            except OSError:
                pass
        return

    auth.mark_job_succeeded(job_id, str(out.resolve()), result_extra=result_extra)


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
        self._active_lock = threading.Lock()
        self._active_job_id: str | None = None
        self._active_proc: mp.Process | None = None
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

    def cancel(self, job_id: str) -> bool:
        """
        尝试终止当前正在执行的子进程（若该 job 正在运行）。

        返回是否对“当前活跃 job”发起了终止动作（best-effort，不保证立刻终止成功）。
        """
        with self._active_lock:
            if self._active_job_id != job_id:
                return False
            p = self._active_proc
        if p and p.is_alive():
            try:
                p.terminate()
            except Exception:  # noqa: BLE001
                return False
            return True
        return False

    def _loop(self) -> None:
        while True:
            job_id = self._q.get()
            try:
                self._process_one(job_id)
            except Exception:  # noqa: BLE001
                log.exception("event=worker.loop.error job=%s", short_job_id(job_id))
            finally:
                self._q.task_done()

    def _process_one(self, job_id: str) -> None:
        if not self.auth.try_claim_job_running(job_id):
            return

        rec = self.auth.get_job(job_id)
        if not rec or not rec.output_file:
            jid, u, fn = _job_log_fields(job_id, rec)
            log_event(log, logging.WARNING, "job.record.incomplete", job=short_job_id(jid), user=u, file=fn)
            self.auth.mark_job_failed(job_id, "内部错误：任务记录不完整")
            return

        inp = Path(rec.input_file)
        out = Path(rec.output_file)
        jid, user, fname = _job_log_fields(job_id, rec)
        log_event(
            log,
            logging.INFO,
            "job.start",
            job=short_job_id(jid),
            user=user,
            file=fname,
            input=inp.name,
            output=out.name,
        )
        # 严格真实进度模式：启动后先展示“准备中”，不写入估算百分比。
        self.auth.update_job_progress(job_id, note="图转文准备中（正在初始化）")

        # 将实际转换放到子进程，便于「取消」时直接终止，让队列可以立刻拉起下一个任务。
        proc = mp.Process(
            target=_run_conversion_in_subprocess,
            kwargs={
                "job_id": job_id,
                "auth_db_path_str": str(self.auth.db_path),
                "input_file": str(inp),
                "output_file": str(out),
            },
            name=f"docling-job-{job_id[:8]}",
            daemon=True,
        )
        with self._active_lock:
            self._active_job_id = job_id
            self._active_proc = proc

        proc.start()
        try:
            # 等待子进程结束；若任务被取消则立即终止子进程，释放 worker 去处理下一个排队任务。
            while proc.is_alive():
                latest = self.auth.get_job(job_id)
                if not latest or latest.status != "running":
                    try:
                        proc.terminate()
                    except Exception:  # noqa: BLE001
                        pass
                    break
                proc.join(timeout=0.5)
            proc.join(timeout=2.0)
        finally:
            with self._active_lock:
                if self._active_job_id == job_id:
                    self._active_job_id = None
                    self._active_proc = None

        # 若子进程异常退出且任务仍处于 running，则标记失败（避免永远卡在 running）。
        latest = self.auth.get_job(job_id)
        if latest and latest.status == "running" and proc.exitcode not in (0, None):
            self.auth.mark_job_failed(job_id, "转换进程异常退出")
