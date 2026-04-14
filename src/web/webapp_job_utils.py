from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from fastapi import HTTPException

from src.core.auth import JobRecord

_JOB_ID_RE = re.compile(r"^[a-f0-9]{32}$")


def normalize_job_id(raw: str) -> str:
    jid = raw.strip()
    if not _JOB_ID_RE.fullmatch(jid):
        raise HTTPException(status_code=400, detail="无效的任务 ID")
    return jid


def remove_job_workspace(job_id: str, input_dir: Path, output_dir: Path) -> None:
    for base in (input_dir, output_dir):
        d = base / job_id
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)


def job_to_api_dict(
    job: JobRecord,
    *,
    queue_position: int | None = None,
    queue_total: int = 0,
) -> dict[str, object | None]:
    out: dict[str, object | None] = {
        "job_id": job.job_id,
        "owner_username": job.owner_username,
        "original_filename": job.original_filename,
        "status": job.status,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "error_message": job.error_message,
        "is_directory": bool(job.is_directory),
        "total_files": job.total_files,
        "processed_files": job.processed_files,
        "succeeded_files": job.succeeded_files,
        "failed_files": job.failed_files,
    }
    if job.status == "succeeded":
        out["download_url"] = f"/jobs/{job.job_id}/download"
    else:
        out["download_url"] = None

    out["queue_position"] = queue_position
    out["queue_total"] = queue_total

    if job.status == "running":
        out["progress_percent"] = job.progress_percent
        out["progress_note"] = job.progress_note
        out["progress_pages_done"] = job.progress_pages_done
        out["progress_pages_total"] = job.progress_pages_total
        out["current_file_name"] = job.current_file_name
    elif job.status == "succeeded":
        out["progress_percent"] = 100
        out["progress_note"] = "已完成"
        out["progress_pages_done"] = None
        out["progress_pages_total"] = None
        out["current_file_name"] = None
    else:
        out["progress_percent"] = None
        out["progress_note"] = None
        out["progress_pages_done"] = None
        out["progress_pages_total"] = None
        out["current_file_name"] = None

    out["pdf_vl_failed_pages"] = None
    out["failed_files_preview"] = None
    if job.status == "succeeded" and job.result_extra:
        try:
            data = json.loads(job.result_extra)
            if isinstance(data, dict):
                pages = data.get("pdf_vl_failed_pages")
                if isinstance(pages, list) and pages:
                    nums: list[int] = []
                    for x in pages:
                        if isinstance(x, int):
                            nums.append(x)
                        elif isinstance(x, str) and x.strip().isdigit():
                            nums.append(int(x.strip()))
                    if nums:
                        out["pdf_vl_failed_pages"] = nums

                failed_files = data.get("failed_files")
                if isinstance(failed_files, list) and failed_files:
                    preview: list[dict[str, str]] = []
                    for item in failed_files[:20]:
                        if isinstance(item, dict):
                            preview.append(
                                {
                                    "file": str(item.get("file", "")),
                                    "error": str(item.get("error", "")),
                                }
                            )
                    if preview:
                        out["failed_files_preview"] = preview
        except (json.JSONDecodeError, TypeError, ValueError):
            out["pdf_vl_failed_pages"] = None

    return out


def wait_job_terminal(
    job_id: str,
    get_job: Callable[[str], JobRecord | None],
    poll_seconds: float = 0.25,
) -> JobRecord:
    terminal = frozenset({"succeeded", "failed", "cancelled"})
    while True:
        row = get_job(job_id)
        if row and row.status in terminal:
            return row
        time.sleep(poll_seconds)


def safe_rel_path(raw: str) -> Path:
    p = Path((raw or "").replace("\\", "/").strip("/").strip())
    if not str(p) or str(p) == ".":
        raise HTTPException(status_code=400, detail="无效的相对路径")
    if p.is_absolute():
        raise HTTPException(status_code=400, detail="相对路径非法")
    if ".." in p.parts:
        raise HTTPException(status_code=400, detail="相对路径非法")
    return p


def zip_job_output_folder(job_dir: Path) -> Path:
    fd, tmp_zip = tempfile.mkstemp(suffix=".zip")
    os.close(fd)
    Path(tmp_zip).unlink(missing_ok=True)
    archive_base = Path(tmp_zip).with_suffix("")
    return Path(shutil.make_archive(str(archive_base), "zip", root_dir=str(job_dir)))

