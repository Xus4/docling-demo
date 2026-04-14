from __future__ import annotations
from fastapi.responses import HTMLResponse
import logging
import os
import re
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

from src.core.access_token import (
    create_access_token,
)
from src.core.auth import AuthStore, AuthUser
from config import AppConfig
from src.core.oa_auth import authenticate_with_oa
from src.core.job_worker import JobQueueWorker
from src.core.service import ConversionService
from src.logging_utils import configure_logging, log_event, short_job_id
from src.web.webapp_job_utils import (
    job_to_api_dict as _job_to_api_dict,
    normalize_job_id as _normalize_job_id,
    remove_job_workspace as _remove_job_workspace,
    safe_rel_path as _safe_rel_path,
    wait_job_terminal as _wait_job_terminal,
    zip_job_output_folder as _zip_job_output_folder,
)
from src.web.webapp_uploads import ingest_uploads_create_job
from src.web.webapp_auth import (
    can_access_job as _can_access_job,
    is_admin as _is_admin,
    require_auth_user,
)
from src.web.webapp_jobs_view import list_jobs_payload
from src.web.webapp_downloads import (
    build_batch_download_response,
    build_single_download_response,
)
from src.web.webapp_login import auth_bootstrap_payload, handle_login
from src.web.webapp_job_actions import (
    cancel_job_payload,
    delete_job_payload,
    get_job_detail_payload,
    retry_job_payload,
)
from src.web.webapp_legacy_convert import handle_legacy_convert

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"

config = AppConfig.from_env()
config.ensure_dirs()
service = ConversionService(config)
auth_store = AuthStore(config.database_url)
if config.oa_auth_enabled:
    auth_store.ensure_env_admin_user(
        config.auth_admin_username,
        config.initial_password,
    )
else:
    auth_store.bootstrap_users(
        users=config.auth_users,
        initial_password=config.initial_password,
        admin_username=config.auth_admin_username,
    )
job_worker = JobQueueWorker(
    auth_store,
    service,
    max_parallel_jobs=config.worker_max_parallel_jobs,
)

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
    rotate_mode=os.getenv("LOG_ROTATE_MODE", "time").strip().lower(),
    retention_days=max(1, int(os.getenv("LOG_RETENTION_DAYS", "7"))),
    app="webapp",
)
log = logging.getLogger("webapp")


_JOB_STATUS_POLL_RE = re.compile(
    r"GET /jobs/[a-f0-9]{32} HTTP/",
    re.IGNORECASE,
)


class _SuppressJobsListAccessLogFilter(logging.Filter):
    """屏蔽前端轮询产生的 /jobs 访问日志噪音。"""

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
    title="鏅烘灑鏂囨。",
    debug=config.debug,
    lifespan=_lifespan,
)
app.add_middleware(
    SessionMiddleware,
    secret_key=config.session_secret,
    same_site="lax",
    https_only=False,
    max_age=config.access_token_ttl_sec,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class LoginRequest(BaseModel):
    username: str
    password: str


def _require_auth_user(request: Request) -> AuthUser:
    return require_auth_user(
        request,
        access_token_secret=config.access_token_secret,
        oa_auth_enabled=config.oa_auth_enabled,
        get_user=auth_store.get_user,
    )


def _remove_job_workspace_for_app(job_id: str) -> None:
    _remove_job_workspace(job_id, config.input_dir, config.output_dir)


async def _ingest_uploads_create_job(
    request: Request,
    *,
    file: UploadFile | None,
    files: list[UploadFile] | None,
    relative_paths: list[str] | None,
    upload_kind: str,
    root_name: str,
) -> str:
    return await ingest_uploads_create_job(
        request,
        file=file,
        files=files,
        relative_paths=relative_paths,
        upload_kind=upload_kind,
        root_name=root_name,
        require_auth_user=_require_auth_user,
        safe_rel_path=_safe_rel_path,
        remove_job_workspace=_remove_job_workspace_for_app,
        service=service,
        auth_store=auth_store,
        job_worker=job_worker,
        max_file_size_bytes=config.max_file_size_bytes,
        log=log,
        short_job_id=short_job_id,
    )




INDEX_HTML = (STATIC_DIR / "index.html").read_text(encoding="utf-8")

@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML



@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/app/config")
def app_config_public() -> dict[str, object]:
    return {
        "allowed_types": sorted(config.allowed_types),
        "max_file_size_bytes": config.max_file_size_bytes,
    }


@app.post("/auth/login")
def login(payload: LoginRequest, request: Request) -> dict[str, object]:
    return handle_login(
        username=payload.username,
        password=payload.password,
        request=request,
        config=config,
        auth_store=auth_store,
        authenticate_with_oa=authenticate_with_oa,
        create_access_token=create_access_token,
    )


@app.post("/auth/logout")
def logout(request: Request) -> dict[str, str]:
    request.session.clear()
    return {"message": "ok"}


def _list_jobs_payload(
    user: AuthUser,
    owner: str | None,
    status: str | None,
    q: str | None,
    page: int,
    page_size: int,
) -> dict[str, object]:
    return list_jobs_payload(
        user=user,
        owner=owner,
        status=status,
        q=q,
        page=page,
        page_size=page_size,
        auth_store=auth_store,
        is_admin=_is_admin,
        job_to_api_dict=_job_to_api_dict,
        log=log,
    )


@app.get("/auth/me")
def me(request: Request) -> dict[str, str]:
    user = _require_auth_user(request)
    return {"username": user.username, "role": user.role}


@app.get("/auth/bootstrap")
def auth_bootstrap(request: Request) -> dict[str, object]:
    """单次请求返回当前用户与默认任务列表，用于前端初始化。"""
    user = _require_auth_user(request)
    return auth_bootstrap_payload(user=user, list_jobs_payload=_list_jobs_payload)


@app.get("/auth/users")
def list_auth_users(request: Request) -> dict[str, list[str]]:
    user = _require_auth_user(request)
    if not _is_admin(user):
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return {"usernames": auth_store.list_job_owner_usernames()}


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
    q: str | None = Query(None, description="按原始文件名关键词搜索"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=500),
) -> dict[str, object]:
    user = _require_auth_user(request)
    return _list_jobs_payload(user, owner, status, q, page, page_size)


@app.get("/jobs/{job_id}")
def get_job_detail(job_id: str, request: Request) -> dict[str, object]:
    jid = _normalize_job_id(job_id)
    user = _require_auth_user(request)
    return get_job_detail_payload(
        jid=jid,
        user=user,
        auth_store=auth_store,
        can_access_job=_can_access_job,
        job_to_api_dict=_job_to_api_dict,
    )


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str, request: Request) -> dict[str, str]:
    jid = _normalize_job_id(job_id)
    user = _require_auth_user(request)
    return cancel_job_payload(
        jid=jid,
        user=user,
        auth_store=auth_store,
        can_access_job=_can_access_job,
        job_worker=job_worker,
        log=log,
    )


@app.post("/jobs/{job_id}/retry")
def retry_job(job_id: str, request: Request) -> dict[str, str]:
    jid = _normalize_job_id(job_id)
    user = _require_auth_user(request)
    return retry_job_payload(
        jid=jid,
        user=user,
        auth_store=auth_store,
        can_access_job=_can_access_job,
        job_worker=job_worker,
        output_dir_root=config.output_dir,
        log=log,
    )


@app.delete("/jobs/{job_id}")
def delete_job_record(job_id: str, request: Request) -> dict[str, str]:
    jid = _normalize_job_id(job_id)
    user = _require_auth_user(request)
    return delete_job_payload(
        jid=jid,
        user=user,
        auth_store=auth_store,
        can_access_job=_can_access_job,
        remove_job_workspace=_remove_job_workspace_for_app,
    )


class BatchDownloadRequest(BaseModel):
    job_ids: list[str]


@app.get("/jobs/{job_id}/download")
def download_job_result(
    job_id: str, request: Request, background_tasks: BackgroundTasks
) -> FileResponse:
    jid = _normalize_job_id(job_id)
    user = _require_auth_user(request)
    return build_single_download_response(
        jid=jid,
        user=user,
        auth_store=auth_store,
        can_access_job=_can_access_job,
        zip_job_output_folder=_zip_job_output_folder,
        short_job_id=short_job_id,
        log=log,
        background_tasks=background_tasks,
    )


@app.post("/jobs/batch-download")
def batch_download_jobs(
    payload: BatchDownloadRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> FileResponse:
    user = _require_auth_user(request)
    return build_batch_download_response(
        raw_ids=list(payload.job_ids or []),
        user=user,
        normalize_job_id=_normalize_job_id,
        auth_store=auth_store,
        can_access_job=_can_access_job,
        zip_job_output_folder=_zip_job_output_folder,
        log=log,
        background_tasks=background_tasks,
    )


@app.post("/convert")
async def convert(request: Request, file: UploadFile = File(...)) -> dict[str, str]:
    return await handle_legacy_convert(
        request=request,
        file=file,
        ingest_uploads_create_job=_ingest_uploads_create_job,
        wait_job_terminal=_wait_job_terminal,
        get_job=auth_store.get_job,
    )


@app.get("/download/{job_id}")
def download_legacy(
    job_id: str, request: Request, background_tasks: BackgroundTasks
) -> FileResponse:
    return download_job_result(job_id, request, background_tasks)


