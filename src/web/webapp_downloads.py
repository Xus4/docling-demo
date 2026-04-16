from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable
import logging
from pathlib import Path
from typing import Protocol

from fastapi import BackgroundTasks, HTTPException
from fastapi.responses import FileResponse


class _JobLike(Protocol):
    job_id: str
    output_root: str | None
    output_file: str | None
    original_filename: str
    status: str


class _AuthStoreLike(Protocol):
    def get_job(self, jid: str) -> _JobLike | None: ...


def _job_output_dir(job: _JobLike, *, workspace_output_root: Path | None = None) -> Path:
    """解析任务输出目录：依次尝试库里的 output_root、output_file 所在目录、以及当前配置的 output_dir/job_id。"""
    candidates: list[Path] = []
    if job.output_root and str(job.output_root).strip():
        candidates.append(Path(job.output_root))
    if job.output_file and str(job.output_file).strip():
        out_path = Path(job.output_file)
        candidates.append(out_path if out_path.is_dir() else out_path.parent)
    if workspace_output_root is not None:
        jid = str(getattr(job, "job_id", "") or "").strip()
        if jid:
            candidates.append(Path(workspace_output_root) / jid)
    for job_dir in candidates:
        if job_dir.is_dir():
            return job_dir
    if candidates:
        raise HTTPException(
            status_code=400,
            detail="未找到转换结果目录，输出可能已被移动或删除",
        )
    raise HTTPException(status_code=400, detail="未找到转换结果")


def build_single_download_response(
    *,
    jid: str,
    user: object,
    auth_store: _AuthStoreLike,
    can_access_job: Callable[[object, _JobLike], bool],
    zip_job_output_folder: Callable[[Path], Path],
    short_job_id: Callable[[str], str],
    log: logging.Logger,
    background_tasks: BackgroundTasks,
    workspace_output_root: Path | None = None,
) -> FileResponse:
    job = auth_store.get_job(jid)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    if not can_access_job(user, job):
        raise HTTPException(status_code=403, detail="无权下载该任务")
    if job.status != "succeeded":
        raise HTTPException(status_code=400, detail="任务未完成或不可下载")

    job_dir = _job_output_dir(job, workspace_output_root=workspace_output_root)
    try:
        zip_path = zip_job_output_folder(job_dir)
    except Exception as exc:
        log.exception("event=job.download.zip_failed job=%s dir=%s", short_job_id(jid), job_dir)
        raise HTTPException(status_code=500, detail="打包下载失败") from exc
    background_tasks.add_task(lambda p=zip_path: p.unlink(missing_ok=True))
    return FileResponse(
        path=zip_path,
        media_type="application/zip",
        filename=f"{Path(job.original_filename).stem or 'result'}.zip",
    )


def build_batch_download_response(
    *,
    raw_ids: list[object],
    user: object,
    normalize_job_id: Callable[[str], str],
    auth_store: _AuthStoreLike,
    can_access_job: Callable[[object, _JobLike], bool],
    zip_job_output_folder: Callable[[Path], Path],
    log: logging.Logger,
    background_tasks: BackgroundTasks,
    workspace_output_root: Path | None = None,
    zip_filename: str | None = None,
) -> FileResponse:
    job_ids: list[str] = []
    for x in raw_ids:
        jid = normalize_job_id(str(x))
        if jid not in job_ids:
            job_ids.append(jid)
    if not job_ids:
        raise HTTPException(status_code=400, detail="未选择可下载的任务")

    staging = Path(tempfile.mkdtemp(prefix="docparse_jobs_"))
    try:
        for jid in job_ids:
            job = auth_store.get_job(jid)
            if not job:
                raise HTTPException(status_code=404, detail=f"任务不存在: {jid}")
            if not can_access_job(user, job):
                raise HTTPException(status_code=403, detail=f"无权下载任务: {jid}")
            if job.status != "succeeded":
                raise HTTPException(status_code=400, detail="仅支持下载已完成任务")

            try:
                job_dir = _job_output_dir(job, workspace_output_root=workspace_output_root)
            except HTTPException as exc:
                if exc.status_code == 400 and exc.detail:
                    raise HTTPException(
                        status_code=400,
                        detail=f"{exc.detail}（任务 {jid[:8]}）",
                    ) from exc
                raise
            stem = Path(job.original_filename).stem or "result"
            target = staging / f"{stem}_{jid[:8]}"
            shutil.copytree(job_dir, target, dirs_exist_ok=True)

        zip_path = zip_job_output_folder(staging)
    except HTTPException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(staging, ignore_errors=True)
        log.exception("event=jobs.batch_download.failed count=%s", len(job_ids))
        raise HTTPException(status_code=500, detail="批量打包下载失败") from exc

    background_tasks.add_task(lambda p=zip_path: p.unlink(missing_ok=True))
    background_tasks.add_task(lambda d=staging: shutil.rmtree(d, ignore_errors=True))
    return FileResponse(
        path=zip_path,
        media_type="application/zip",
        filename=zip_filename or "jobs.zip",
    )

