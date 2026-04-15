from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import queue
import threading
from pathlib import Path

from src.core.auth import AuthStore, JobRecord
from config import AppConfig
from src.core.service import ConversionError, ConversionService
from src.logging_utils import configure_logging, log_event, short_job_id

log = logging.getLogger("job_worker")


def _job_log_fields(job_id: str, rec: JobRecord | None) -> tuple[str, str, str]:
    if not rec:
        return job_id, "?", "?"
    name = rec.original_filename or Path(rec.input_file).name
    user = rec.owner_username or "?"
    return job_id, user, name


def _run_single_file_conversion(
    *,
    job_id: str,
    auth: AuthStore,
    service: ConversionService,
    input_file: Path,
    output_file: Path,
    mineru_backend: str | None,
    mineru_task_id: str | None,
) -> None:
    def on_parse_progress(done: int, total: int) -> None:
        # 子进程内协作取消：若任务已不再 running，则立刻抛错中断后续流程
        j = auth.get_job(job_id)
        if not j or j.status != "running":
            raise RuntimeError("任务已取消")
        t = max(int(total), 1)
        d = max(0, min(int(done), t))
        pct = int(round(100.0 * d / t))
        pct = max(0, min(99, pct))
        note = "任务已提交，等待调度…"
        if pct >= 10:
            note = "正在解析中…"
        if d >= t:
            pct = 99
            note = "解析完成，正在整理结果…"
        auth.update_job_progress(
            job_id,
            percent=pct,
            note=note,
            pages_done=None,
            pages_total=None,
        )

    if not input_file.is_file():
        auth.mark_job_failed(job_id, "输入文件不存在")
        return

    def cancel_check() -> bool:
        j = auth.get_job(job_id)
        return (not j) or j.status != "running"

    try:
        service.convert_to_markdown(
            str(input_file),
            str(output_file),
            backend_override=mineru_backend,
            remote_task_id=mineru_task_id,
            on_remote_task_id=lambda tid: auth.set_job_mineru_task_id(job_id, tid),
            progress_callback=on_parse_progress,
            cancel_check=cancel_check,
        )

    except ConversionError as exc:
        auth.update_job_file_counts(
            job_id,
            total_files=1,
            processed_files=1,
            succeeded_files=0,
            failed_files=1,
        )
        auth.mark_job_failed(job_id, str(exc))
        return
    except Exception as exc:  # noqa: BLE001
        auth.update_job_file_counts(
            job_id,
            total_files=1,
            processed_files=1,
            succeeded_files=0,
            failed_files=1,
        )
        auth.mark_job_failed(job_id, f"转换失败: {exc!s}"[:4000])
        return

    latest = auth.get_job(job_id)
    if not latest or latest.status != "running":
        if output_file.exists():
            try:
                output_file.unlink()
            except OSError:
                pass
        return

    auth.update_job_file_counts(
        job_id,
        total_files=1,
        processed_files=1,
        succeeded_files=1,
        failed_files=0,
    )
    auth.mark_job_succeeded(job_id, str(output_file.resolve()), result_extra=None)


def _run_directory_conversion(
    *,
    job_id: str,
    auth: AuthStore,
    service: ConversionService,
    input_root: Path,
    output_root: Path,
    mineru_backend: str | None,
) -> None:
    if not input_root.is_dir():
        auth.mark_job_failed(job_id, "输入目录不存在")
        return

    files = list(service.iter_supported_files(input_root))
    total = len(files)
    auth.update_job_file_counts(
        job_id,
        total_files=total,
        processed_files=0,
        succeeded_files=0,
        failed_files=0,
    )

    if total <= 0:
        auth.mark_job_failed(job_id, "文件夹中没有可处理的文件")
        return

    processed = 0
    succeeded = 0
    failed = 0
    failed_items: list[dict[str, str]] = []

    for idx, src in enumerate(files, start=1):
        latest = auth.get_job(job_id)
        if not latest or latest.status != "running":
            return

        rel = src.relative_to(input_root)
        dst = (output_root / rel).with_suffix(".md")
        dst.parent.mkdir(parents=True, exist_ok=True)

        def make_pdf_callback(file_idx: int, file_total: int, file_name: str):
            def on_parse_progress(done: int, total: int) -> None:
                j = auth.get_job(job_id)
                if not j or j.status != "running":
                    raise RuntimeError("任务已取消")
                t = max(int(total), 1)
                d = max(0, min(int(done), t))
                combined_pct = int(round(100.0 * ((file_idx - 1) + (d / t)) / file_total))
                combined_pct = max(0, min(99, combined_pct))
                note = f"正在解析（{file_idx}/{file_total}）…"
                if d >= t:
                    combined_pct = min(99, max(combined_pct, int(round(100.0 * file_idx / file_total)) - 1))
                    note = f"文件解析完成（{file_idx}/{file_total}），正在整理结果…"
                auth.update_job_progress(
                    job_id,
                    percent=combined_pct,
                    note=note,
                    pages_done=None,
                    pages_total=None,
                    current_file_name=file_name,
                )
            return on_parse_progress

        def cancel_check() -> bool:
            j = auth.get_job(job_id)
            return (not j) or j.status != "running"

        pct = int(round(((idx - 1) / total) * 100))
        auth.update_job_progress(
            job_id,
            percent=max(0, min(99, pct)),
            note=f"正在处理 {idx}/{total}：{rel.as_posix()}",
            pages_done=None,
            pages_total=None,
            current_file_name=src.name,
        )

        try:
            service.convert_to_markdown(
                str(src),
                str(dst),
                backend_override=mineru_backend,
                progress_callback=make_pdf_callback(idx, total, src.name),
                cancel_check=cancel_check,
            )
            succeeded += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            failed_items.append(
                {
                    "file": rel.as_posix(),
                    "error": str(exc)[:500],
                }
            )

        processed += 1
        auth.update_job_file_counts(
            job_id,
            total_files=total,
            processed_files=processed,
            succeeded_files=succeeded,
            failed_files=failed,
        )
        pct_done = int(round((processed / total) * 100))
        if processed < total:
            pct_done = min(pct_done, 99)
        auth.update_job_progress(
            job_id,
            percent=pct_done,
            note=f"已完成 {processed}/{total} 个文件（成功 {succeeded}，失败 {failed}）",
        )

    latest = auth.get_job(job_id)
    if not latest or latest.status != "running":
        return

    if succeeded <= 0:
        msg = "文件夹处理完成，但没有成功转换的文件"
        if failed_items:
            msg += f"；失败 {failed} 个"
        auth.mark_job_failed(job_id, msg[:4000])
        return

    result_extra = None
    if failed_items:
        result_extra = json.dumps(
            {
                "failed_files": failed_items[:200],
                "summary": {
                    "total_files": total,
                    "processed_files": processed,
                    "succeeded_files": succeeded,
                    "failed_files": failed,
                },
            },
            ensure_ascii=False,
        )

    auth.mark_job_succeeded(job_id, str(output_root.resolve()), result_extra=result_extra)


def _run_conversion_in_subprocess(
    *,
    job_id: str,
    database_url: str,
    input_file: str,
    output_file: str,
    input_root: str | None,
    output_root: str | None,
    is_directory: int,
    mineru_backend: str | None,
    mineru_task_id: str | None,
) -> None:
    config = AppConfig.from_env()
    run_log_file = (os.getenv("RUN_LOG_FILE") or "").strip()
    if run_log_file:
        configure_logging(
            verbose=bool(config.debug),
            log_file=Path(run_log_file),
            rotate_max_bytes=max(1024 * 1024, int(os.getenv("LOG_MAX_BYTES", "52428800"))),
            rotate_backup_count=max(1, int(os.getenv("LOG_BACKUP_COUNT", "10"))),
            rotate_mode=os.getenv("LOG_ROTATE_MODE", "time").strip().lower(),
            retention_days=max(1, int(os.getenv("LOG_RETENTION_DAYS", "7"))),
            app="job_worker",
        )
    service = ConversionService(config)
    auth = AuthStore(database_url)

    try:
        if int(is_directory):
            _run_directory_conversion(
                job_id=job_id,
                auth=auth,
                service=service,
                input_root=Path(input_root or ""),
                output_root=Path(output_root or ""),
                mineru_backend=mineru_backend,
            )
            return

        _run_single_file_conversion(
            job_id=job_id,
            auth=auth,
            service=service,
            input_file=Path(input_file),
            output_file=Path(output_file),
            mineru_backend=mineru_backend,
            mineru_task_id=mineru_task_id,
        )
    except (KeyboardInterrupt, SystemExit):
        # 服务重启/中断时，将任务回退为 queued，保留 mineru_task_id 以便下次恢复轮询。
        auth.requeue_running_job_for_resume(job_id)
        raise


class JobQueueWorker:
    """后台调度线程：支持多个任务并发执行（每任务一个子进程）。"""

    def __init__(
        self,
        auth_store: AuthStore,
        conversion_service: ConversionService,
        *,
        max_parallel_jobs: int = 1,
    ) -> None:
        self.auth = auth_store
        self.service = conversion_service
        self.max_parallel_jobs = max(1, int(max_parallel_jobs))
        self._q: queue.Queue[str] = queue.Queue()
        self._active_lock = threading.Lock()
        self._active_procs: dict[str, mp.Process] = {}
        self._thread = threading.Thread(
            target=self._loop, name="parse-job-worker", daemon=True
        )
        self._thread.start()

    def enqueue(self, job_id: str) -> None:
        self._q.put(job_id)

    def requeue_queued_from_db(self) -> int:
        ids = self.auth.list_queued_job_ids()
        for jid in ids:
            self._q.put(jid)
        return len(ids)

    def cancel(self, job_id: str) -> bool:
        with self._active_lock:
            p = self._active_procs.get(job_id)
        if p and p.is_alive():
            try:
                p.terminate()
                # Best-effort join to speed up slot release after manual cancellation.
                p.join(timeout=1.0)
            except Exception:  # noqa: BLE001
                return False
            with self._active_lock:
                cur = self._active_procs.get(job_id)
                if cur is p:
                    self._active_procs.pop(job_id, None)
            return True
        return False

    def _active_count(self) -> int:
        with self._active_lock:
            return len(self._active_procs)

    def _loop(self) -> None:
        while True:
            self._reap_finished_processes()

            if self._active_count() >= self.max_parallel_jobs:
                threading.Event().wait(0.1)
                continue

            try:
                job_id = self._q.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                self._start_one(job_id)
            except Exception:  # noqa: BLE001
                log.exception("event=worker.loop.error job=%s", short_job_id(job_id))
            finally:
                self._q.task_done()

    def _start_one(self, job_id: str) -> None:
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
        self.auth.update_job_progress(job_id, note="准备处理任务…")

        proc = mp.Process(
            target=_run_conversion_in_subprocess,
            kwargs={
                "job_id": job_id,
                "database_url": str(self.auth.database_url),
                "input_file": str(inp),
                "output_file": str(out),
                "input_root": rec.input_root,
                "output_root": rec.output_root,
                "is_directory": rec.is_directory,
                "mineru_backend": rec.mineru_backend,
                "mineru_task_id": rec.mineru_task_id,
            },
            name=f"parse-job-{job_id[:8]}",
            daemon=True,
        )
        try:
            proc.start()
        except Exception:  # noqa: BLE001
            self.auth.mark_job_failed(job_id, "启动转换进程失败")
            raise

        with self._active_lock:
            self._active_procs[job_id] = proc

    def _reap_finished_processes(self) -> None:
        with self._active_lock:
            items = list(self._active_procs.items())

        finished: list[tuple[str, mp.Process]] = []
        for job_id, proc in items:
            if proc.is_alive():
                latest = self.auth.get_job(job_id)
                if not latest or latest.status != "running":
                    try:
                        proc.terminate()
                    except Exception:  # noqa: BLE001
                        pass
                continue
            try:
                proc.join(timeout=0.1)
            except Exception:  # noqa: BLE001
                pass
            finished.append((job_id, proc))

        if not finished:
            return

        with self._active_lock:
            for job_id, proc in finished:
                cur = self._active_procs.get(job_id)
                if cur is proc:
                    self._active_procs.pop(job_id, None)

        for job_id, proc in finished:
            latest = self.auth.get_job(job_id)
            if latest and latest.status == "running" and proc.exitcode not in (0, None):
                self.auth.mark_job_failed(job_id, "转换进程异常退出")
