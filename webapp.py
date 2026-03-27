from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config import AppConfig
from service import ConversionError, ConversionService

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"

config = AppConfig.from_env()
config.ensure_dirs()
service = ConversionService(config)

logging.basicConfig(
    level=logging.DEBUG if config.debug else logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
log = logging.getLogger("webapp")

app = FastAPI(title="Docling Demo Web Service", debug=config.debug)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/convert")
async def convert(file: UploadFile = File(...)) -> dict[str, str]:
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
def download(job_id: str) -> FileResponse:
    output_dir = config.output_dir / job_id
    files = list(output_dir.glob("*.md"))
    if not files:
        raise HTTPException(status_code=404, detail="未找到转换结果")
    md_file = files[0]
    return FileResponse(path=md_file, media_type="text/markdown", filename=md_file.name)
