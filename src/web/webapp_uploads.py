from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Request, UploadFile

from src.core.auth import AuthUser
from src.core.service import ConversionError
from src.logging_utils import log_event


async def ingest_uploads_create_job(
    request: Request,
    *,
    file: UploadFile | None,
    files: list[UploadFile] | None,
    relative_paths: list[str] | None,
    upload_kind: str,
    root_name: str,
    mineru_backend: str | None = None,
    require_auth_user: Callable[[Request], AuthUser],
    safe_rel_path: Callable[[str], Path],
    remove_job_workspace: Callable[[str], None],
    service: Any,
    auth_store: Any,
    job_worker: Any,
    max_file_size_bytes: int,
    log: logging.Logger,
    short_job_id: Callable[[str], str],
) -> str:
    user = require_auth_user(request)
    backend_raw = (mineru_backend or "").strip()
    if backend_raw:
        allowed_backends = {
            "pipeline",
            "vlm-auto-engine",
            "vlm-http-client",
            "hybrid-auto-engine",
            "hybrid-http-client",
        }
        if backend_raw not in allowed_backends:
            raise HTTPException(status_code=400, detail=f"不支持的 mineru_backend: {backend_raw}")
    backend_value = backend_raw or None

    upload_kind = (upload_kind or "file").strip().lower()
    if upload_kind not in {"file", "folder"}:
        raise HTTPException(status_code=400, detail="upload_kind 仅支持 file 或 folder")

    merged_files: list[UploadFile] = []
    if files:
        merged_files.extend(files)
    if file is not None:
        merged_files.append(file)

    merged_files = [f for f in merged_files if f is not None and f.filename]
    if not merged_files:
        raise HTTPException(status_code=400, detail="未收到上传文件")

    if not relative_paths:
        if upload_kind == "folder":
            raise HTTPException(status_code=400, detail="文件夹上传缺少 relative_paths")
        relative_paths = [merged_files[0].filename or "upload"]

    if len(relative_paths) != len(merged_files):
        raise HTTPException(status_code=400, detail="files 与 relative_paths 数量不一致")

    paths = None
    try:
        if upload_kind == "file":
            if len(merged_files) != 1:
                raise HTTPException(status_code=400, detail="单文件上传只能包含 1 个文件")
            filename_raw = merged_files[0].filename or "upload"
            filename = Path(filename_raw).name or "upload"
            service.validate_extension(filename)
            paths = service.create_job_paths(filename, is_directory=False)
            total_size = 0
            dst = paths.input_entry
            dst.parent.mkdir(parents=True, exist_ok=True)

            with dst.open("wb") as buffer:
                while True:
                    if await request.is_disconnected():
                        raise HTTPException(status_code=499, detail="客户端已取消上传")
                    chunk = await merged_files[0].read(1024 * 1024)
                    if not chunk:
                        break
                    total_size += len(chunk)
                    if total_size > max_file_size_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail=f"文件过大，最大允许 {max_file_size_bytes} 字节",
                        )
                    buffer.write(chunk)

            await merged_files[0].close()

            auth_store.insert_job(
                job_id=paths.job_id,
                owner_username=user.username,
                role_snapshot=user.role,
                original_filename=filename,
                input_file=str(paths.input_entry.resolve()),
                output_file=str(paths.output_entry.resolve()),
                is_directory=0,
                input_root=str(paths.input_root.resolve()),
                output_root=str(paths.output_root.resolve()),
                total_files=1,
                processed_files=0,
                succeeded_files=0,
                failed_files=0,
                mineru_backend=backend_value,
            )
            job_worker.enqueue(paths.job_id)
            log.info(
                "单文件任务已创建并入队 job_id=%s user=%s role=%s file=%s size_bytes=%s",
                paths.job_id,
                user.username,
                user.role,
                filename,
                total_size,
            )
            return paths.job_id

        normalized_root = Path(root_name).name.strip() or "uploaded_folder"
        paths = service.create_job_paths(normalized_root, is_directory=True)
        input_root = paths.input_entry
        input_root.mkdir(parents=True, exist_ok=True)

        total_size = 0
        for up, rel_raw in zip(merged_files, relative_paths, strict=False):
            rel = safe_rel_path(rel_raw)
            dst = input_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            with dst.open("wb") as buffer:
                while True:
                    if await request.is_disconnected():
                        raise HTTPException(status_code=499, detail="客户端已取消上传")
                    chunk = await up.read(1024 * 1024)
                    if not chunk:
                        break
                    total_size += len(chunk)
                    if total_size > max_file_size_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail=f"上传总大小过大，最大允许 {max_file_size_bytes} 字节",
                        )
                    buffer.write(chunk)
            await up.close()

        supported_files = list(service.iter_supported_files(input_root))
        if not supported_files:
            raise HTTPException(status_code=400, detail="文件夹中没有可处理的文件")

        auth_store.insert_job(
            job_id=paths.job_id,
            owner_username=user.username,
            role_snapshot=user.role,
            original_filename=normalized_root,
            input_file=str(paths.input_entry.resolve()),
            output_file=str(paths.output_root.resolve()),
            is_directory=1,
            input_root=str(paths.input_entry.resolve()),
            output_root=str(paths.output_root.resolve()),
            total_files=len(supported_files),
            processed_files=0,
            succeeded_files=0,
            failed_files=0,
            mineru_backend=backend_value,
        )
        job_worker.enqueue(paths.job_id)
        log_event(
            log,
            logging.INFO,
            "job.created",
            job=short_job_id(paths.job_id),
            user=user.username,
            role=user.role,
            file=normalized_root,
            size_bytes=total_size,
        )

        return paths.job_id

    except HTTPException:
        if paths:
            remove_job_workspace(paths.job_id)
        raise
    except ConversionError as exc:
        if paths:
            remove_job_workspace(paths.job_id)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("event=job.create.failed")
        raise HTTPException(status_code=500, detail="创建任务失败") from exc

