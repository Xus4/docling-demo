from __future__ import annotations

import logging
import queue
import threading
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

        try:
            self.service.convert_to_markdown(str(inp), str(out))
        except ConversionError as exc:
            self.auth.mark_job_failed(job_id, str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            log.exception("转换失败 job_id=%s", job_id)
            self.auth.mark_job_failed(job_id, f"转换失败: {exc!s}"[:4000])
            return

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
