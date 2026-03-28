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
    input_file: Path
    output_file: Path


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

    def create_job_paths(self, filename: str) -> ConversionJobPaths:
        job_id = uuid.uuid4().hex
        safe_name = Path(filename).name
        input_file = self.app_config.input_dir / job_id / safe_name
        output_file = self.app_config.output_dir / job_id / f"{Path(safe_name).stem}.md"
        input_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        return ConversionJobPaths(job_id=job_id, input_file=input_file, output_file=output_file)

    def save_upload_file(self, source_path: Path, target_path: Path) -> None:
        shutil.copy2(source_path, target_path)

    def convert_to_markdown(
        self,
        input_path: str,
        output_path: str,
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> Path:
        src = Path(input_path).resolve()
        dst = Path(output_path).resolve()
        self.converter.convert_path_to_markdown(
            src, dst, progress_callback=progress_callback
        )
        return dst

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
