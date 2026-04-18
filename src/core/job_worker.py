from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import queue
import threading
import time
from pathlib import Path

import httpx

from src.core.auth import AuthStore, JobRecord
from config import AppConfig
from src.core.service import ConversionError, ConversionService
from src.logging_utils import configure_logging, log_event, short_job_id

log = logging.getLogger("job_worker")

# 与 mineru_client / ConversionService 的 processing_stage 键一致，用于同步 progress_note
_STAGE_NOTES: dict[str, str] = {
    "mineru_prepare": "准备解析…",
    "mineru_upload": "正在提交至解析服务…",
    "mineru_remote": "正在解析…",
    "mineru_materialize": "正在整理输出文件…",
    "xlsx_extract": "正在将 Excel 转为 Markdown…",
    "semantic_enhance": "表格语义补充中…",
}


def _emit_processing_stage_note(auth: AuthStore, job_id: str, stage: str) -> None:
    auth.set_job_processing_stage(job_id, stage)
    note = _STAGE_NOTES.get(stage)
    if note:
        auth.update_job_progress(job_id, note=note)


def _log_stage_enter(job_id: str, user: str, file: str, stage: str) -> None:
    log_event(
        log,
        logging.INFO,
        "job.stage.enter",
        job=short_job_id(job_id),
        user=user,
        file=file,
        stage=stage,
        stage_note=_STAGE_NOTES.get(stage, ""),
    )


def _job_log_fields(job_id: str, rec: JobRecord | None) -> tuple[str, str, str]:
    if not rec:
        return job_id, "?", "?"
    name = rec.original_filename or Path(rec.input_file).name
    user = rec.owner_username or "?"
    return job_id, user, name


def _should_requeue_network_instead_of_fail(exc: BaseException) -> bool:
    """MinerU 连不上或连接抖动：回队重试，不把任务标为失败。"""
    text = str(exc)
    if "任务已取消" in text:
        return False
    if isinstance(exc, httpx.RequestError):
        return True
    markers = (
        "Server disconnected",
        "disconnected without sending",
        "Connection reset",
        "ReadTimeout",
        "ConnectTimeout",
        "ConnectError",
        "网络不稳定，已重试",
        "网络不稳定，已持续重试",
        "MinerU 网络异常",
        "All connection attempts failed",
        "getaddrinfo failed",
        "Name or service not known",
        "Broken pipe",
        "WriteTimeout",
        "RemoteProtocolError",
        "LocalProtocolError",
    )
    return any(m in text for m in markers)


def _semantic_pct_from_table_ratio(
    *,
    done: int,
    tables_total: int,
    floor_pct: int,
    span_pct: int,
) -> int:
    """按 ``done/tables_total`` 在 ``[floor_pct, floor_pct+span_pct]`` 内线性映射（封顶 99）。"""
    t = max(int(tables_total), 0)
    d = max(0, min(int(done), t if t > 0 else int(done)))
    if t <= 0:
        return min(99, max(0, int(floor_pct) + max(1, int(span_pct))))
    lo = max(0, min(99, int(floor_pct)))
    span = max(1, min(int(span_pct), 99 - lo))
    out = lo + int(round(span * (d / t)))
    return min(99, max(lo, out))


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
    rec0 = auth.get_job(job_id)
    jid, user, fname = _job_log_fields(job_id, rec0)
    parse_t0 = {"t": None}

    def on_parse_progress(done: int, total: int) -> None:
        # 子进程内协作取消：若任务已不再 running，则立刻抛错中断后续流程
        j = auth.get_job(job_id)
        if not j or j.status != "running":
            raise RuntimeError("任务已取消")
        t = max(int(total), 1)
        d = max(0, min(int(done), t))
        raw_pct = int(round(100.0 * d / t))
        raw_pct = max(0, min(99, raw_pct))
        note = "任务已提交，等待调度…"
        if d >= t:
            # 与解析阶段封顶 10% 对齐，语义阶段再按表格比例铺到 99%
            pct = 10
            note = "解析完成，正在整理结果…"
        else:
            # 解析未完成：进度条不超过 10%，避免远程轮询稍久就显示八九成
            pct = min(10, raw_pct)
            if d * 100 >= 90 * t:
                # 接近完成：仍归为解析阶段（不再单独显示「拉取结果」）
                note = "正在解析中…"
            elif raw_pct >= 10:
                note = "正在解析中…"
        if parse_t0["t"] is None:
            parse_t0["t"] = time.time()
        eta_sec = None
        if d > 0 and d < t and parse_t0["t"] is not None:
            elapsed = max(0.001, time.time() - parse_t0["t"])
            estimated_total = elapsed / (d / t)
            remain = estimated_total - elapsed
            if remain > 0:
                eta_sec = int(round(remain))
        auth.update_job_progress(
            job_id,
            percent=pct,
            note=note,
            clear_progress_pages=True,
        )
        # 轮询极频繁，默认只打 DEBUG；阶段完成（d>=t）再打 INFO，避免刷屏
        _lvl = logging.INFO if d >= t else logging.DEBUG
        log_event(
            log,
            _lvl,
            "job.stage.progress",
            job=short_job_id(jid),
            user=user,
            file=fname,
            stage="mineru_remote",
            stage_progress=f"{d}/{t}",
            percent=pct,
            eta_sec=eta_sec if eta_sec is not None else "-",
            note=note,
        )

    if not input_file.is_file():
        auth.mark_job_failed(job_id, "输入文件不存在")
        return

    def cancel_check() -> bool:
        j = auth.get_job(job_id)
        return (not j) or j.status != "running"

    def _on_stage(name: str) -> None:
        _emit_processing_stage_note(auth, job_id, name)
        _log_stage_enter(jid, user, fname, name)

    def on_semantic_progress(done: int, tables_total: int, eta_sec: float | None) -> None:
        j = auth.get_job(job_id)
        if not j or j.status != "running":
            raise RuntimeError("任务已取消")
        t = max(int(tables_total), 0)
        d = max(0, min(int(done), t if t > 0 else int(done)))
        if t > 0:
            # 与解析结束刻度 10% 衔接，按表格完成比例线性映射到 99%（PDF/Word/图/xlsx 等同路径）
            pct = _semantic_pct_from_table_ratio(
                done=d,
                tables_total=t,
                floor_pct=10,
                span_pct=89,
            )
            note = f"表格语义补充中（{d}/{t}）…"
        else:
            pct = 99
            note = "表格语义补充（无待处理表格）…"
        if eta_sec is not None and eta_sec > 1:
            note += f" 预计剩余 {int(round(eta_sec))} 秒"
        auth.update_job_progress(
            job_id,
            percent=pct,
            note=note,
            pages_done=d if t > 0 else None,
            pages_total=t if t > 0 else None,
            clear_progress_pages=t <= 0,
        )
        _sem_lvl = logging.INFO if t > 0 and d >= t else logging.DEBUG
        log_event(
            log,
            _sem_lvl,
            "job.stage.progress",
            job=short_job_id(jid),
            user=user,
            file=fname,
            stage="semantic_enhance",
            stage_progress=f"{d}/{t}" if t > 0 else "-",
            percent=pct,
            eta_sec=int(round(eta_sec)) if eta_sec is not None and eta_sec > 0 else "-",
            note=note,
        )

    try:
        service.convert_to_markdown(
            str(input_file),
            str(output_file),
            backend_override=mineru_backend,
            remote_task_id=mineru_task_id,
            on_remote_task_id=lambda tid: auth.set_job_mineru_task_id(job_id, tid),
            progress_callback=on_parse_progress,
            semantic_progress_callback=on_semantic_progress,
            cancel_check=cancel_check,
            processing_stage_callback=_on_stage,
        )

    except ConversionError as exc:
        if _should_requeue_network_instead_of_fail(exc) and auth.requeue_running_job_for_resume(job_id):
            log_event(
                log,
                logging.WARNING,
                "job.transient_network.requeue",
                job=short_job_id(jid),
                user=user,
                file=fname,
                error=str(exc)[:500],
            )
            return
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
        if _should_requeue_network_instead_of_fail(exc) and auth.requeue_running_job_for_resume(job_id):
            log_event(
                log,
                logging.WARNING,
                "job.transient_network.requeue",
                job=short_job_id(jid),
                user=user,
                file=fname,
                error=str(exc)[:500],
            )
            return
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
    latest_done = auth.get_job(job_id)
    jid, user, fname = _job_log_fields(job_id, latest_done)
    log_event(
        log,
        logging.INFO,
        "job.done",
        job=short_job_id(jid),
        user=user,
        file=fname,
        status="succeeded",
        total_files=1,
        processed_files=1,
        succeeded_files=1,
        failed_files=0,
        output=str(output_file.resolve()),
    )


def _run_directory_conversion(
    *,
    job_id: str,
    auth: AuthStore,
    service: ConversionService,
    input_root: Path,
    output_root: Path,
    mineru_backend: str | None,
) -> None:
    rec0 = auth.get_job(job_id)
    jid, user, fname = _job_log_fields(job_id, rec0)
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

        sem_anchor_pct: dict[str, int | None] = {"v": None}

        rel = src.relative_to(input_root)
        dst = (output_root / rel).with_suffix(".md")
        dst.parent.mkdir(parents=True, exist_ok=True)

        def make_pdf_callback(file_idx: int, file_total: int, file_name: str):
            parse_t0 = {"t": None}
            def on_parse_progress(done: int, total: int) -> None:
                j = auth.get_job(job_id)
                if not j or j.status != "running":
                    raise RuntimeError("任务已取消")
                t = max(int(total), 1)
                d = max(0, min(int(done), t))
                if d >= t:
                    file_inner = 1.0
                else:
                    # 单文件内解析未完成：最多折算为本文件的 10%，避免条一下子顶到八九十
                    file_inner = min(0.10, d / t)
                combined_pct = int(
                    round(100.0 * ((file_idx - 1) + file_inner) / file_total)
                )
                combined_pct = max(0, min(99, combined_pct))
                note = f"正在解析（{file_idx}/{file_total}）…"
                if d >= t:
                    combined_pct = min(
                        99,
                        max(
                            combined_pct,
                            int(round(100.0 * file_idx / file_total)) - 1,
                        ),
                    )
                    note = f"文件解析完成（{file_idx}/{file_total}），正在整理结果…"
                elif d * 100 >= 90 * t:
                    note = f"正在解析（{file_idx}/{file_total}）…"
                if parse_t0["t"] is None:
                    parse_t0["t"] = time.time()
                eta_sec = None
                if d > 0 and d < t and parse_t0["t"] is not None:
                    elapsed = max(0.001, time.time() - parse_t0["t"])
                    estimated_total = elapsed / (d / t)
                    remain = estimated_total - elapsed
                    if remain > 0:
                        eta_sec = int(round(remain))
                auth.update_job_progress(
                    job_id,
                    percent=combined_pct,
                    note=note,
                    clear_progress_pages=True,
                    current_file_name=file_name,
                )
                _lvl = logging.INFO if d >= t else logging.DEBUG
                log_event(
                    log,
                    _lvl,
                    "job.stage.progress",
                    job=short_job_id(jid),
                    user=user,
                    file=rel.as_posix(),
                    stage="mineru_remote",
                    stage_progress=f"{d}/{t}",
                    percent=combined_pct,
                    eta_sec=eta_sec if eta_sec is not None else "-",
                    note=note,
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
            clear_progress_pages=True,
            current_file_name=src.name,
        )

        def _on_stage(name: str) -> None:
            _emit_processing_stage_note(auth, job_id, name)
            _log_stage_enter(jid, user, rel.as_posix(), name)

        def on_semantic_progress(done: int, tables_total: int, eta_sec: float | None) -> None:
            j = auth.get_job(job_id)
            if not j or j.status != "running":
                raise RuntimeError("任务已取消")
            t = max(int(tables_total), 0)
            d = max(0, min(int(done), t if t > 0 else int(done)))
            if sem_anchor_pct["v"] is None:
                cur = auth.get_job(job_id)
                raw_lo = (
                    int(cur.progress_percent)
                    if cur and cur.progress_percent is not None
                    else int(round(100.0 * (idx - 1) / max(total, 1)))
                )
                sem_anchor_pct["v"] = max(0, min(99, raw_lo))
            lo = int(sem_anchor_pct["v"])
            if t > 0:
                slot = max(1, int(round(100.0 / max(total, 1))))
                span = max(1, min(slot, 99 - lo))
                combined_pct = _semantic_pct_from_table_ratio(
                    done=d,
                    tables_total=t,
                    floor_pct=lo,
                    span_pct=span,
                )
                note = f"表格语义补充中（{idx}/{total}，表格 {d}/{t}）…"
            else:
                combined_pct = min(
                    99,
                    max(lo, int(round(100.0 * ((idx - 1) + 0.5) / max(total, 1)))),
                )
                note = f"表格语义补充中（{idx}/{total}）…"
            if eta_sec is not None and eta_sec > 1:
                note += f" 预计剩余 {int(round(eta_sec))} 秒"
            auth.update_job_progress(
                job_id,
                percent=combined_pct,
                note=note,
                pages_done=d if t > 0 else None,
                pages_total=t if t > 0 else None,
                clear_progress_pages=t <= 0,
                current_file_name=src.name,
            )
            _sem_lvl = logging.INFO if t > 0 and d >= t else logging.DEBUG
            log_event(
                log,
                _sem_lvl,
                "job.stage.progress",
                job=short_job_id(jid),
                user=user,
                file=rel.as_posix(),
                stage="semantic_enhance",
                stage_progress=f"{d}/{t}" if t > 0 else "-",
                percent=combined_pct,
                eta_sec=int(round(eta_sec)) if eta_sec is not None and eta_sec > 0 else "-",
                note=note,
            )

        try:
            service.convert_to_markdown(
                str(src),
                str(dst),
                backend_override=mineru_backend,
                progress_callback=make_pdf_callback(idx, total, src.name),
                semantic_progress_callback=on_semantic_progress,
                cancel_check=cancel_check,
                processing_stage_callback=_on_stage,
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
            clear_progress_pages=True,
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
    latest_done = auth.get_job(job_id)
    jid, user, fname = _job_log_fields(job_id, latest_done)
    log_event(
        log,
        logging.INFO,
        "job.done",
        job=short_job_id(jid),
        user=user,
        file=fname,
        status="succeeded",
        total_files=total,
        processed_files=processed,
        succeeded_files=succeeded,
        failed_files=failed,
        output=str(output_root.resolve()),
    )


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
