from __future__ import annotations

import mimetypes
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from fastapi import HTTPException
from fastapi.responses import FileResponse

from src.web.webapp_job_utils import safe_rel_path


class _JobLike(Protocol):
    job_id: str
    input_file: str
    input_root: str | None
    output_file: str | None
    output_root: str | None
    original_filename: str
    status: str


class _AuthStoreLike(Protocol):
    def get_job(self, jid: str) -> _JobLike | None: ...


_MEDIA_TYPES = {
    ".md": "text/markdown; charset=utf-8",
    ".markdown": "text/markdown; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xls": "application/vnd.ms-excel",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".ppt": "application/vnd.ms-powerpoint",
}


def _media_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in _MEDIA_TYPES:
        return _MEDIA_TYPES[ext]
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "application/octet-stream"


def _inline_file_response(path: Path, filename: str | None = None) -> FileResponse:
    return FileResponse(
        path=path,
        media_type=_media_type(path),
        filename=filename or path.name,
        content_disposition_type="inline",
    )


def _require_job(
    *,
    jid: str,
    user: object,
    auth_store: _AuthStoreLike,
    can_access_job: Callable[[object, _JobLike], bool],
) -> _JobLike:
    job = auth_store.get_job(jid)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    if not can_access_job(user, job):
        raise HTTPException(status_code=403, detail="无权预览该任务")
    return job


def _existing_file(path: Path, *, missing_detail: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise HTTPException(status_code=400, detail=missing_detail)
    return resolved


def _safe_child_file(root: Path, relative_path: str, *, missing_detail: str) -> Path:
    rel = safe_rel_path(relative_path)
    base = root.resolve()
    candidate = (base / rel).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="预览路径非法") from exc
    return _existing_file(candidate, missing_detail=missing_detail)


def _job_output_dir(job: _JobLike, *, workspace_output_root: Path | None = None) -> Path:
    candidates: list[Path] = []
    if job.output_root and str(job.output_root).strip():
        candidates.append(Path(job.output_root))
    if job.output_file and str(job.output_file).strip():
        out_path = Path(job.output_file)
        candidates.append(out_path if out_path.is_dir() else out_path.parent)
    if workspace_output_root is not None:
        candidates.append(Path(workspace_output_root) / str(job.job_id))

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_dir():
            return resolved

    raise HTTPException(status_code=400, detail="未找到转换结果目录，输出可能已被移动或删除")


def build_source_preview_response(
    *,
    jid: str,
    user: object,
    auth_store: _AuthStoreLike,
    can_access_job: Callable[[object, _JobLike], bool],
    relative_path: str | None = None,
) -> FileResponse:
    job = _require_job(
        jid=jid,
        user=user,
        auth_store=auth_store,
        can_access_job=can_access_job,
    )
    if relative_path:
        if not job.input_root or not str(job.input_root).strip():
            raise HTTPException(status_code=400, detail="该任务没有可预览的输入目录")
        path = _safe_child_file(
            Path(job.input_root),
            relative_path,
            missing_detail="未找到预览源文件",
        )
        return _inline_file_response(path)

    path = _existing_file(Path(job.input_file), missing_detail="未找到预览源文件")
    return _inline_file_response(path, filename=job.original_filename or path.name)


def build_output_preview_response(
    *,
    jid: str,
    user: object,
    auth_store: _AuthStoreLike,
    can_access_job: Callable[[object, _JobLike], bool],
    workspace_output_root: Path | None,
    relative_path: str | None = None,
) -> FileResponse:
    job = _require_job(
        jid=jid,
        user=user,
        auth_store=auth_store,
        can_access_job=can_access_job,
    )
    if job.status != "succeeded":
        raise HTTPException(status_code=400, detail="任务未完成，暂不可预览结果")

    if relative_path:
        output_dir = _job_output_dir(job, workspace_output_root=workspace_output_root)
        path = _safe_child_file(
            output_dir,
            relative_path,
            missing_detail="未找到预览结果文件",
        )
        return _inline_file_response(path)

    if job.output_file and str(job.output_file).strip():
        path = Path(job.output_file).resolve()
        if path.is_file():
            return _inline_file_response(path, filename=path.name)
        if path.is_dir():
            md_files = sorted(path.rglob("*.md"))
            if md_files:
                return _inline_file_response(md_files[0])

    output_dir = _job_output_dir(job, workspace_output_root=workspace_output_root)
    md_files = sorted(output_dir.rglob("*.md"))
    if not md_files:
        raise HTTPException(status_code=400, detail="未找到可预览的 Markdown 结果")
    return _inline_file_response(md_files[0])
