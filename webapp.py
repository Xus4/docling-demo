from __future__ import annotations
from fastapi.responses import HTMLResponse
import json
import logging
import os
import re
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

from auth import AuthStore, AuthUser, JobRecord
from config import AppConfig
from job_worker import JobQueueWorker
from service import ConversionError, ConversionService
from src.logging_utils import configure_logging, log_event, short_job_id

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"

config = AppConfig.from_env()
config.ensure_dirs()
service = ConversionService(config)
auth_store = AuthStore(config.auth_db_path)
auth_store.bootstrap_users(
    users=config.auth_users,
    initial_password=config.initial_password,
    admin_username=config.auth_admin_username,
)
job_worker = JobQueueWorker(auth_store, service)

run_log_file_env = (os.getenv("RUN_LOG_FILE") or "").strip()
if run_log_file_env:
    run_log_file = Path(run_log_file_env).expanduser().resolve()
    run_log_file.parent.mkdir(parents=True, exist_ok=True)
else:
    log_dir = Path(os.getenv("LOG_DIR", str(config.data_dir / "logs"))).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    run_log_name = (
        "webapp_"
        + time.strftime("%Y%m%d_%H%M%S")
        + "_"
        + str(os.getpid())
        + ".log"
    )
    run_log_file = (log_dir / run_log_name).resolve()
    os.environ["RUN_LOG_FILE"] = str(run_log_file)
configure_logging(
    verbose=bool(config.debug),
    log_file=run_log_file,
    rotate_max_bytes=max(1024 * 1024, int(os.getenv("LOG_MAX_BYTES", "52428800"))),
    rotate_backup_count=max(1, int(os.getenv("LOG_BACKUP_COUNT", "10"))),
    app="webapp",
)
log = logging.getLogger("webapp")


_JOB_STATUS_POLL_RE = re.compile(
    r"GET /jobs/[a-f0-9]{32} HTTP/",
    re.IGNORECASE,
)


class _SuppressJobsListAccessLogFilter(logging.Filter):
    """屏蔽前端轮询产生的 access 行：GET /jobs、GET /jobs?...、GET /jobs/{job_id}。"""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        if '"GET /jobs?' in msg or '"GET /jobs HTTP' in msg:
            return False
        if _JOB_STATUS_POLL_RE.search(msg):
            return False
        return True


logging.getLogger("uvicorn.access").addFilter(_SuppressJobsListAccessLogFilter())


@asynccontextmanager
async def _lifespan(app: FastAPI):
    n_reset = auth_store.reset_orphan_running_jobs_to_queued()
    if n_reset:
        log_event(log, logging.INFO, "worker.recover.running_to_queued", count=n_reset)
    n = job_worker.requeue_queued_from_db()
    if n:
        log_event(log, logging.INFO, "worker.recover.queued_requeued", count=n)
    yield


app = FastAPI(
    title="Docling Demo Web Service",
    debug=config.debug,
    lifespan=_lifespan,
)
app.add_middleware(
    SessionMiddleware,
    secret_key=config.session_secret,
    same_site="lax",
    https_only=False,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class LoginRequest(BaseModel):
    username: str
    password: str


def _require_auth_user(request: Request) -> AuthUser:
    username = request.session.get("username")
    if not username:
        raise HTTPException(status_code=401, detail="请先登录")
    user = auth_store.get_user(str(username))
    if not user:
        request.session.clear()
        raise HTTPException(status_code=401, detail="登录状态已失效，请重新登录")
    return user


def _is_admin(user: AuthUser) -> bool:
    return user.role == "admin"


def _can_access_job(user: AuthUser, job: JobRecord) -> bool:
    return _is_admin(user) or job.owner_username == user.username


_JOB_ID_RE = re.compile(r"^[a-f0-9]{32}$")


def _normalize_job_id(raw: str) -> str:
    jid = raw.strip()
    if not _JOB_ID_RE.fullmatch(jid):
        raise HTTPException(status_code=400, detail="无效的任务 ID")
    return jid


def _remove_job_workspace(job_id: str) -> None:
    for base in (config.input_dir, config.output_dir):
        d = base / job_id
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)


def _job_to_api_dict(job: JobRecord) -> dict[str, object | None]:
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

    q_pos, q_total = auth_store.get_queue_position(job.job_id)
    out["queue_position"] = q_pos
    out["queue_total"] = q_total

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


def _wait_job_terminal(job_id: str, poll_seconds: float = 0.25) -> JobRecord:
    terminal = frozenset({"succeeded", "failed", "cancelled"})
    while True:
        row = auth_store.get_job(job_id)
        if row and row.status in terminal:
            return row
        time.sleep(poll_seconds)


def _safe_rel_path(raw: str) -> Path:
    p = Path((raw or "").replace("\\", "/").strip("/").strip())
    if not str(p) or str(p) == ".":
        raise HTTPException(status_code=400, detail="无效的相对路径")
    if p.is_absolute():
        raise HTTPException(status_code=400, detail="相对路径非法")
    if ".." in p.parts:
        raise HTTPException(status_code=400, detail="相对路径非法")
    return p


async def _ingest_uploads_create_job(
    request: Request,
    *,
    file: UploadFile | None,
    files: list[UploadFile] | None,
    relative_paths: list[str] | None,
    upload_kind: str,
    root_name: str,
) -> str:
    user = _require_auth_user(request)

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
        service.cleanup_old_jobs()

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
                    if total_size > config.max_file_size_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail=f"文件过大，最大允许 {config.max_file_size_bytes} 字节",
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

        # folder mode
        normalized_root = Path(root_name).name.strip() or "uploaded_folder"
        paths = service.create_job_paths(normalized_root, is_directory=True)
        input_root = paths.input_entry
        input_root.mkdir(parents=True, exist_ok=True)

        total_size = 0
        saved_count = 0
        for up, rel_raw in zip(merged_files, relative_paths, strict=False):
            rel = _safe_rel_path(rel_raw)
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
                    if total_size > config.max_file_size_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail=f"上传总大小过大，最大允许 {config.max_file_size_bytes} 字节",
                        )
                    buffer.write(chunk)
            await up.close()
            saved_count += 1

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
            _remove_job_workspace(paths.job_id)
        raise
    except ConversionError as exc:
        if paths:
            _remove_job_workspace(paths.job_id)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("event=job.create.failed")
        raise HTTPException(status_code=500, detail="创建任务失败") from exc




INDEX_HTML = (STATIC_DIR / "index.html").read_text(encoding="utf-8")

@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML



@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/auth/login")
def login(payload: LoginRequest, request: Request) -> dict[str, str]:
    user = auth_store.authenticate(payload.username, payload.password)
    if not user:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    request.session["username"] = user.username
    request.session["role"] = user.role
    return {"username": user.username, "role": user.role}


@app.post("/auth/logout")
def logout(request: Request) -> dict[str, str]:
    request.session.clear()
    return {"message": "ok"}


def _list_jobs_payload(
    user: AuthUser,
    owner: str | None,
    status: str | None,
    page: int,
    page_size: int,
) -> dict[str, object]:
    offset = (page - 1) * page_size
    items, total = auth_store.list_jobs(
        viewer_username=user.username,
        viewer_role=user.role,
        owner_filter=owner if _is_admin(user) else None,
        status_filter=status,
        limit=page_size,
        offset=offset,
    )
    return {
        "items": [_job_to_api_dict(j) for j in items],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@app.get("/auth/me")
def me(request: Request) -> dict[str, str]:
    user = _require_auth_user(request)
    return {"username": user.username, "role": user.role}


@app.get("/auth/bootstrap")
def auth_bootstrap(request: Request) -> dict[str, object]:
    """单次请求返回当前用户与首页任务列表（与首次进入工作台时的默认筛选一致）。"""
    user = _require_auth_user(request)
    return {
        "username": user.username,
        "role": user.role,
        "jobs": _list_jobs_payload(user, None, None, 1, 100),
    }


@app.get("/auth/users")
def list_auth_users(request: Request) -> dict[str, list[str]]:
    user = _require_auth_user(request)
    if not _is_admin(user):
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return {"usernames": auth_store.list_usernames()}


@app.post("/jobs")
async def create_job(
    request: Request,
    file: UploadFile | None = File(None),
    files: list[UploadFile] | None = File(None),
    relative_paths: list[str] | None = Form(None),
    upload_kind: str = Form("file"),
    root_name: str = Form(""),
) -> dict[str, object]:
    job_id = await _ingest_uploads_create_job(
        request,
        file=file,
        files=files,
        relative_paths=relative_paths,
        upload_kind=upload_kind,
        root_name=root_name,
    )
    rec = auth_store.get_job(job_id)
    if not rec:
        raise HTTPException(status_code=500, detail="任务创建异常")
    return {
        "job_id": job_id,
        "status": rec.status,
        "created_at": rec.created_at,
    }


@app.get("/jobs")
def list_jobs(
    request: Request,
    owner: str | None = Query(None, description="管理员按提交人筛选"),
    status: str | None = Query(None, description="queued|running|succeeded|failed|cancelled"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=500),
) -> dict[str, object]:
    user = _require_auth_user(request)
    return _list_jobs_payload(user, owner, status, page, page_size)


@app.get("/jobs/{job_id}")
def get_job_detail(job_id: str, request: Request) -> dict[str, object]:
    jid = _normalize_job_id(job_id)
    user = _require_auth_user(request)
    job = auth_store.get_job(jid)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    if not _can_access_job(user, job):
        raise HTTPException(status_code=403, detail="无权访问该任务")
    return _job_to_api_dict(job)


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str, request: Request) -> dict[str, str]:
    jid = _normalize_job_id(job_id)
    user = _require_auth_user(request)
    job = auth_store.get_job(jid)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    if not _can_access_job(user, job):
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


@app.post("/jobs/{job_id}/retry")
def retry_job(job_id: str, request: Request) -> dict[str, str]:
    jid = _normalize_job_id(job_id)
    user = _require_auth_user(request)
    job = auth_store.get_job(jid)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    if not _can_access_job(user, job):
        raise HTTPException(status_code=403, detail="无权操作该任务")
    if job.status in ("queued", "running"):
        raise HTTPException(status_code=400, detail="当前状态不可重试")

    out_dir = config.output_dir / jid
    if out_dir.is_dir():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not auth_store.try_reset_job_queued(jid):
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


@app.delete("/jobs/{job_id}")
def delete_job_record(job_id: str, request: Request) -> dict[str, str]:
    jid = _normalize_job_id(job_id)
    user = _require_auth_user(request)
    job = auth_store.get_job(jid)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    if not _can_access_job(user, job):
        raise HTTPException(status_code=403, detail="无权删除该任务")
    if job.status == "running":
        raise HTTPException(
            status_code=409,
            detail="任务正在执行，请先点击「取消」结束后再删除记录",
        )
    if not auth_store.delete_job(jid):
        raise HTTPException(status_code=404, detail="任务不存在")
    _remove_job_workspace(jid)
    return {"message": "已删除"}


def _zip_job_output_folder(job_dir: Path) -> Path:
    fd, tmp_zip = tempfile.mkstemp(suffix=".zip")
    os.close(fd)
    Path(tmp_zip).unlink(missing_ok=True)
    archive_base = Path(tmp_zip).with_suffix("")
    return Path(shutil.make_archive(str(archive_base), "zip", root_dir=str(job_dir)))


@app.get("/jobs/{job_id}/download")
def download_job_result(
    job_id: str, request: Request, background_tasks: BackgroundTasks
) -> FileResponse:
    jid = _normalize_job_id(job_id)
    user = _require_auth_user(request)
    job = auth_store.get_job(jid)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    if not _can_access_job(user, job):
        raise HTTPException(status_code=403, detail="无权下载该任务")
    if job.status != "succeeded":
        raise HTTPException(status_code=400, detail="任务未完成或不可下载")

    if job.output_root:
        job_dir = Path(job.output_root)
    elif job.output_file:
        out_path = Path(job.output_file)
        job_dir = out_path if out_path.is_dir() else out_path.parent
    else:
        raise HTTPException(status_code=404, detail="未找到转换结果")

    if not job_dir.is_dir():
        raise HTTPException(status_code=404, detail="未找到转换结果目录")

    try:
        zip_path = _zip_job_output_folder(job_dir)
    except Exception as exc:
        log.exception("event=job.download.zip_failed job=%s dir=%s", short_job_id(jid), job_dir)
        raise HTTPException(status_code=500, detail="打包下载失败") from exc
    background_tasks.add_task(lambda p=zip_path: p.unlink(missing_ok=True))
    return FileResponse(
        path=zip_path,
        media_type="application/zip",
        filename=f"{Path(job.original_filename).stem or 'result'}.zip",
    )


@app.post("/convert")
async def convert(request: Request, file: UploadFile = File(...)) -> dict[str, str]:
    """兼容旧客户端：创建异步任务并阻塞直到结束，返回原 download_url 形态。"""
    job_id = await _ingest_uploads_create_job(
        request,
        file=file,
        files=None,
        relative_paths=None,
        upload_kind="file",
        root_name="",
    )
    final = _wait_job_terminal(job_id)
    rec = auth_store.get_job(job_id)
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


@app.get("/download/{job_id}")
def download_legacy(
    job_id: str, request: Request, background_tasks: BackgroundTasks
) -> FileResponse:
    return download_job_result(job_id, request, background_tasks)
