from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol

from fastapi import HTTPException


class _TerminalResult(Protocol):
    status: str
    error_message: str | None


class _JobRecord(Protocol):
    original_filename: str


async def handle_legacy_convert(
    *,
    request: object,
    file: object,
    ingest_uploads_create_job: Callable[..., Awaitable[str]],
    wait_job_terminal: Callable[
        [str, Callable[[str], _JobRecord | None]],
        _TerminalResult,
    ],
    get_job: Callable[[str], _JobRecord | None],
) -> dict[str, str]:
    """兼容旧版 /convert：创建任务并阻塞等待终态。"""
    job_id = await ingest_uploads_create_job(
        request,
        file=file,
        files=None,
        relative_paths=None,
        upload_kind="file",
        root_name="",
    )
    final = wait_job_terminal(job_id, get_job)
    rec = get_job(job_id)
    filename = rec.original_filename if rec else ""

    if final.status == "succeeded":
        return {
            "job_id": job_id,
            "filename": filename,
            "download_url": f"/download/{job_id}",
        }
    if final.status == "cancelled":
        raise HTTPException(status_code=400, detail=final.error_message or "任务已取消")
    raise HTTPException(status_code=400, detail=final.error_message or "转换失败")

