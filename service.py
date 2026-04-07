from __future__ import annotations

import shutil
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from config import AppConfig
from src.converter import IndustrialDocConverter


class ConversionError(Exception):
    pass


@dataclass
class ConversionJobPaths:
    job_id: str
    input_root: Path
    output_root: Path
    input_entry: Path
    output_entry: Path
    is_directory: bool


@dataclass(frozen=True)
class ConvertToMarkdownResult:
    """convert_to_markdown 返回值：输出路径 + pdf-vl 单页失败页码（1-based，无则为空元组）。"""

    output_path: Path
    pdf_vl_failed_pages: tuple[int, ...]


class ConversionService:
    def __init__(self, app_config: AppConfig) -> None:
        self.app_config = app_config
        self.converter = IndustrialDocConverter(
            config=self.app_config.build_converter_config()
        )

    def validate_extension(self, filename: str) -> str:
        ext = Path(filename).suffix.lower().lstrip(".")
        if not ext:
            raise ConversionError("文件缺少扩展名")
        if ext not in self.app_config.allowed_types:
            raise ConversionError(f"不支持的文件类型: .{ext}")
        return ext

    def is_supported_file(self, path: Path) -> bool:
        ext = path.suffix.lower().lstrip(".")
        return bool(ext) and ext in self.app_config.allowed_types

    def create_job_paths(self, entry_name: str, *, is_directory: bool = False) -> ConversionJobPaths:
        job_id = uuid.uuid4().hex
        safe_name = Path(entry_name).name or "upload"
        input_root = self.app_config.input_dir / job_id
        output_root = self.app_config.output_dir / job_id
        input_root.mkdir(parents=True, exist_ok=True)
        output_root.mkdir(parents=True, exist_ok=True)

        if is_directory:
            input_entry = input_root / safe_name
            output_entry = output_root / safe_name
        else:
            ext = Path(safe_name).suffix.lower().lstrip(".")
            stem = Path(safe_name).stem
            if ext in {"png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"}:
                input_entry = input_root / f"{stem}.pdf"
                output_entry = output_root / f"{stem}.md"
            else:
                input_entry = input_root / safe_name
                output_entry = output_root / f"{stem}.md"

        return ConversionJobPaths(
            job_id=job_id,
            input_root=input_root,
            output_root=output_root,
            input_entry=input_entry,
            output_entry=output_entry,
            is_directory=is_directory,
        )

    def save_upload_file(self, source_path: Path, target_path: Path) -> None:
        src_ext = source_path.suffix.lower().lstrip(".")
        dst_ext = target_path.suffix.lower().lstrip(".")
        if dst_ext == "pdf" and src_ext in {"png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"}:
            from PIL import Image

            img = Image.open(str(source_path))
            frames = []
            try:
                while True:
                    frame = img.copy()
                    if frame.mode not in ("RGB", "L"):
                        frame = frame.convert("RGB")
                    frames.append(frame)
                    img.seek(img.tell() + 1)
            except EOFError:
                pass
            if not frames:
                raise ConversionError("图片读取失败")
            first = frames[0]
            rest = frames[1:]
            if rest:
                first.save(
                    str(target_path),
                    "PDF",
                    save_all=True,
                    append_images=rest,
                )
            else:
                first.save(str(target_path), "PDF")
            return

        shutil.copy2(source_path, target_path)

    def convert_to_markdown(
        self,
        input_path: str,
        output_path: str,
        *,
        progress_callback: Callable[[int, int], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> ConvertToMarkdownResult:
        src = Path(input_path).resolve()
        dst = Path(output_path).resolve()
        self.converter.convert_path_to_markdown(
            src,
            dst,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
        )
        failed = self.converter.last_pdf_vl_failed_pages
        pages = tuple(failed) if failed else ()
        return ConvertToMarkdownResult(dst, pages)

    def iter_supported_files(self, input_root: Path):
        for path in sorted(input_root.rglob("*")):
            if not path.is_file():
                continue
            if path.name.startswith("."):
                continue
            if self.is_supported_file(path):
                yield path

    def cleanup_old_jobs(self) -> None:
        if not self.app_config.auto_cleanup:
            return
        max_age_seconds = self.app_config.cleanup_max_age_hours * 3600
        now = time.time()
        self._cleanup_base_dir(self.app_config.input_dir, now, max_age_seconds)
        self._cleanup_base_dir(self.app_config.output_dir, now, max_age_seconds)

    @staticmethod
    def _cleanup_base_dir(base_dir: Path, now: float, max_age_seconds: int) -> None:
        if not base_dir.exists():
            return
        for entry in base_dir.iterdir():
            if not entry.is_dir():
                continue
            age = now - entry.stat().st_mtime
            if age > max_age_seconds:
                shutil.rmtree(entry, ignore_errors=True)
