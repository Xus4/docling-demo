from __future__ import annotations

import logging
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from fastapi import HTTPException

from src.logging_utils import log_event, short_job_id


class _UserLike(Protocol):
    username: str


class _JobLike(Protocol):
    owner_username: str
    status: str
    original_filename: str
    is_directory: int
    output_root: str | None
    output_file: str | None


class _AuthStoreLike(Protocol):
    def get_job(self, jid: str) -> _JobLike | None: ...
    def try_mark_job_cancelled(self, jid: str) -> bool: ...
    def try_reset_job_queued(self, jid: str, output_file: str | None = None) -> bool: ...
    def delete_job(self, jid: str) -> bool: ...


class _JobWorkerLike(Protocol):
    def cancel(self, jid: str) -> None: ...
    def enqueue(self, jid: str) -> None: ...


def get_job_detail_payload(
    *,
    jid: str,
    user: _UserLike,
    auth_store: _AuthStoreLike,
    can_access_job: Callable[[_UserLike, _JobLike], bool],
    job_to_api_dict: Callable[[_JobLike], dict[str, object]],
) -> dict[str, object]:
    job = auth_store.get_job(jid)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    if not can_access_job(user, job):
        raise HTTPException(status_code=403, detail="无权访问该任务")
    return job_to_api_dict(job)


def cancel_job_payload(
    *,
    jid: str,
    user: _UserLike,
    auth_store: _AuthStoreLike,
    can_access_job: Callable[[_UserLike, _JobLike], bool],
    job_worker: _JobWorkerLike,
    log: logging.Logger,
) -> dict[str, str]:
    job = auth_store.get_job(jid)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    if not can_access_job(user, job):
        raise HTTPException(status_code=403, detail="无权操作该任务")

    if job.status in ("queued", "running"):
        if auth_store.try_mark_job_cancelled(jid):
            if job.status == "running":
                try:
                    job_worker.cancel(jid)
                except Exception:  # noqa: BLE001
                    pass
            log_event(
                log,
                logging.INFO,
                "job.cancelled",
                job=short_job_id(jid),
                owner=job.owner_username,
                operator=user.username,
                file=job.original_filename,
                prior_status=job.status,
            )
            return {"message": "已取消", "status": "cancelled"}
        raise HTTPException(status_code=409, detail="任务状态已变更，请刷新")

    raise HTTPException(status_code=400, detail="当前状态不可取消")


def retry_job_payload(
    *,
    jid: str,
    user: _UserLike,
    auth_store: _AuthStoreLike,
    can_access_job: Callable[[_UserLike, _JobLike], bool],
    job_worker: _JobWorkerLike,
    output_dir_root: Path,
    log: logging.Logger,
) -> dict[str, str]:
    job = auth_store.get_job(jid)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    if not can_access_job(user, job):
        raise HTTPException(status_code=403, detail="无权操作该任务")
    if job.status in ("queued", "running"):
        raise HTTPException(status_code=400, detail="当前状态不可重试")

    out_dir = output_dir_root / jid
    if out_dir.is_dir():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    if job.is_directory:
        retry_output = str(Path(job.output_root or out_dir).resolve())
    else:
        existing = (job.output_file or "").strip()
        if existing:
            retry_output = str(Path(existing).resolve())
        else:
            stem = Path(job.original_filename or "result").stem or "result"
            retry_output = str((out_dir / f"{stem}.md").resolve())

    if not auth_store.try_reset_job_queued(jid, output_file=retry_output):
        raise HTTPException(status_code=409, detail="任务状态已变更，请刷新")

    job_worker.enqueue(jid)
    log_event(
        log,
        logging.INFO,
        "job.retried",
        job=short_job_id(jid),
        owner=job.owner_username,
        operator=user.username,
        file=job.original_filename,
        prior_status=job.status,
    )
    return {"message": "已重试", "status": "queued"}


def delete_job_payload(
    *,
    jid: str,
    user: _UserLike,
    auth_store: _AuthStoreLike,
    can_access_job: Callable[[_UserLike, _JobLike], bool],
    remove_job_workspace: Callable[[str], None],
) -> dict[str, str]:
    job = auth_store.get_job(jid)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    if not can_access_job(user, job):
        raise HTTPException(status_code=403, detail="无权删除该任务")
    if job.status == "running":
        raise HTTPException(status_code=409, detail="任务正在执行，请先取消后再删除")
    if not auth_store.delete_job(jid):
        raise HTTPException(status_code=404, detail="任务不存在")
    remove_job_workspace(jid)
    return {"message": "已删除"}
