from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

from auth import AuthStore, AuthUser, JobRecord
from config import AppConfig
from job_worker import JobQueueWorker
from service import ConversionError, ConversionService

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

logging.basicConfig(
    level=logging.DEBUG if config.debug else logging.INFO,
    format="%(asctime)s · %(levelname)-7s · %(name)s · %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
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
        log.info(
            "启动时将 %d 个残留「转换中」任务重置为排队（上次进程异常退出或重启导致）",
            n_reset,
        )
    n = job_worker.requeue_queued_from_db()
    if n:
        log.info("启动时已将 %d 个排队任务重新加入执行队列", n)
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
    else:
        out["progress_percent"] = None
        out["progress_note"] = None
        out["progress_pages_done"] = None
        out["progress_pages_total"] = None

    out["pdf_vl_failed_pages"] = None
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


async def _ingest_upload_create_job(request: Request, file: UploadFile) -> str:
    user = _require_auth_user(request)
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    paths = None
    try:
        service.cleanup_old_jobs()
        service.validate_extension(file.filename)
        paths = service.create_job_paths(file.filename)

        total_size = 0
        with paths.input_file.open("wb") as buffer:
            while True:
                # 客户端主动取消/断开上传连接时，避免写入半成品并创建 job。
                if await request.is_disconnected():
                    raise HTTPException(status_code=499, detail="客户端已取消上传")
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total_size += len(chunk)
                if total_size > config.max_file_size_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"文件过大，最大允许 {config.max_file_size_bytes} 字节",
                    )
                buffer.write(chunk)

            # 上传读取循环退出后再做一次断开检测，避免尾部 race condition。
            if await request.is_disconnected():
                raise HTTPException(status_code=499, detail="客户端已取消上传")

        await file.close()

        auth_store.insert_job(
            job_id=paths.job_id,
            owner_username=user.username,
            role_snapshot=user.role,
            original_filename=file.filename,
            input_file=str(paths.input_file.resolve()),
            output_file=str(paths.output_file.resolve()),
        )
        job_worker.enqueue(paths.job_id)
        log.info(
            "任务已创建并入队 job_id=%s user=%s role=%s file=%s size_bytes=%s",
            paths.job_id,
            user.username,
            user.role,
            file.filename,
            total_size,
        )
        return paths.job_id
    except HTTPException as exc:
        if paths:
            # 上传中断/超限等场景下，清理本次已创建的临时输入/输出目录。
            _remove_job_workspace(paths.job_id)
        raise
    except ConversionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("创建任务失败: %s", exc)
        raise HTTPException(status_code=500, detail="创建任务失败") from exc


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


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


@app.get("/auth/me")
def me(request: Request) -> dict[str, str]:
    user = _require_auth_user(request)
    return {"username": user.username, "role": user.role}


@app.get("/auth/users")
def list_auth_users(request: Request) -> dict[str, list[str]]:
    user = _require_auth_user(request)
    if not _is_admin(user):
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return {"usernames": auth_store.list_usernames()}


@app.post("/jobs")
async def create_job(request: Request, file: UploadFile = File(...)) -> dict[str, object]:
    job_id = await _ingest_upload_create_job(request, file)
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
    page_size: int = Query(20, ge=1, le=100),
) -> dict[str, object]:
    user = _require_auth_user(request)
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
            # 若该任务正在运行，尝试立即终止转换子进程，释放 worker 以拉起下一个排队任务。
            if job.status == "running":
                try:
                    job_worker.cancel(jid)
                except Exception:  # noqa: BLE001
                    pass
            log.info(
                "任务已取消 job_id=%s owner=%s operator=%s file=%s prior_status=%s",
                jid,
                job.owner_username,
                user.username,
                job.original_filename,
                job.status,
            )
            return {"message": "已取消", "status": "cancelled"}
        raise HTTPException(status_code=409, detail="任务状态已变更，请刷新")

    raise HTTPException(status_code=400, detail="当前状态不可取消")


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
    """
    将任务输出目录打成 zip（含 .md 及同目录下的切图、Docling 配图等），
    解压后与 Web 服务端路径一致，相对图片引用可正常显示。
    """
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
    if not job.output_file:
        raise HTTPException(status_code=404, detail="未找到转换结果")
    md_file = Path(job.output_file)
    if not md_file.is_file():
        raise HTTPException(status_code=404, detail="未找到转换结果文件")
    job_dir = md_file.parent
    try:
        zip_path = _zip_job_output_folder(job_dir)
    except Exception as exc:
        log.exception("打包下载失败 job_id=%s dir=%s", jid, job_dir)
        raise HTTPException(status_code=500, detail="打包下载失败") from exc
    background_tasks.add_task(lambda p=zip_path: p.unlink(missing_ok=True))
    return FileResponse(
        path=zip_path,
        media_type="application/zip",
        filename=f"{md_file.stem}.zip",
    )


@app.post("/convert")
async def convert(request: Request, file: UploadFile = File(...)) -> dict[str, str]:
    """兼容旧客户端：创建异步任务并阻塞直到结束，返回原 download_url 形态。"""
    job_id = await _ingest_upload_create_job(request, file)
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
    """兼容旧 download_url：与 /jobs/{id}/download 相同校验。"""
    return download_job_result(job_id, request, background_tasks)
