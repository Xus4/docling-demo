from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

from auth import AuthStore
from config import AppConfig
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

logging.basicConfig(
    level=logging.DEBUG if config.debug else logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
log = logging.getLogger("webapp")

app = FastAPI(title="Docling Demo Web Service", debug=config.debug)
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


def _require_login(request: Request) -> str:
    username = request.session.get("username")
    if not username:
        raise HTTPException(status_code=401, detail="请先登录")
    user = auth_store.get_user(str(username))
    if not user:
        request.session.clear()
        raise HTTPException(status_code=401, detail="登录状态已失效，请重新登录")
    return user.username


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
    username = request.session.get("username")
    if not username:
        raise HTTPException(status_code=401, detail="未登录")
    user = auth_store.get_user(str(username))
    if not user:
        request.session.clear()
        raise HTTPException(status_code=401, detail="未登录")
    return {"username": user.username, "role": user.role}


@app.post("/convert")
async def convert(request: Request, file: UploadFile = File(...)) -> dict[str, str]:
    _require_login(request)
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    try:
        service.cleanup_old_jobs()
        service.validate_extension(file.filename)
        job = service.create_job_paths(file.filename)

        total_size = 0
        with job.input_file.open("wb") as buffer:
            while True:
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

        await file.close()
        service.convert_to_markdown(str(job.input_file), str(job.output_file))
        return {
            "job_id": job.job_id,
            "filename": file.filename,
            "download_url": f"/download/{job.job_id}",
        }
    except HTTPException:
        raise
    except ConversionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("转换失败: %s", exc)
        raise HTTPException(status_code=500, detail="转换失败，请联系管理员") from exc


@app.get("/download/{job_id}")
def download(job_id: str, request: Request) -> FileResponse:
    _require_login(request)
    output_dir = config.output_dir / job_id
    files = list(output_dir.glob("*.md"))
    if not files:
        raise HTTPException(status_code=404, detail="未找到转换结果")
    md_file = files[0]
    return FileResponse(path=md_file, media_type="text/markdown", filename=md_file.name)
