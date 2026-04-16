from __future__ import annotations

import logging
import shutil
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from config import AppConfig
from src.logging_utils import log_event
from src.mineru_client import MinerUError, mineru_client_config_from_app, run_mineru_convert
from src.table_semantic.augment import TableCaptionParams, augment_markdown_file
from src.table_semantic.llm_client import OpenAICompatibleConfig

log = logging.getLogger(__name__)


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
    output_path: Path


class ConversionService:
    """上传工作区；解析通过 MinerU HTTP API（默认 MINERU_BASE_URL 见 config）。"""

    def __init__(self, app_config: AppConfig) -> None:
        self.app_config = app_config

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
        backend_override: str | None = None,
        remote_task_id: str | None = None,
        on_remote_task_id: Callable[[str], None] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        semantic_progress_callback: Callable[[int, int, float | None], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
        processing_stage_callback: Callable[[str], None] | None = None,
    ) -> ConvertToMarkdownResult:
        src = Path(input_path).resolve()
        dst = Path(output_path).resolve()
        if not src.is_file():
            raise ConversionError("输入文件不存在")
        if not str(self.app_config.mineru_base_url).strip():
            raise ConversionError(
                "未配置 MINERU_BASE_URL。请在环境变量中设置为 mineru-api 根地址（默认 http://192.168.2.60:8011）。"
            )
        mcfg = mineru_client_config_from_app(self.app_config)
        try:
            run_mineru_convert(
                input_path=src,
                output_path=dst,
                cfg=mcfg,
                backend_override=backend_override,
                resume_task_id=remote_task_id,
                on_remote_task_id=on_remote_task_id,
                progress_callback=progress_callback,
                cancel_check=cancel_check,
                on_processing_stage=processing_stage_callback,
            )
        except MinerUError as exc:
            raise ConversionError(str(exc)) from exc

        if (
            self.app_config.table_semantic_enable
            and self.app_config.table_semantic_base_url.strip()
            and self.app_config.table_semantic_model.strip()
        ):
            if processing_stage_callback is not None:
                processing_stage_callback("semantic_enhance")
            llm_cfg = OpenAICompatibleConfig(
                base_url=self.app_config.table_semantic_base_url,
                api_key=self.app_config.table_semantic_api_key,
                model=self.app_config.table_semantic_model,
                timeout_sec=self.app_config.table_semantic_timeout_sec,
                thinking_enable=self.app_config.table_semantic_thinking_enable,
                max_tokens=self.app_config.table_semantic_max_tokens,
                temperature=self.app_config.table_semantic_temperature,
            )
            base_url = str(self.app_config.table_semantic_base_url).strip()
            log.info(
                f"开始表格语义增强 | "
                f"模型: {self.app_config.table_semantic_model} | "
                f"并发数: {self.app_config.table_semantic_max_concurrency} | "
                f"超时: {self.app_config.table_semantic_timeout_sec}秒"
            )
            try:
                augment_markdown_file(
                    dst,
                    cfg=llm_cfg,
                    max_concurrency=self.app_config.table_semantic_max_concurrency,
                    caption_params=TableCaptionParams(
                        context_before_chars=self.app_config.table_semantic_context_before_chars,
                        context_after_chars=self.app_config.table_semantic_context_after_chars,
                        caption_target_chars=self.app_config.table_semantic_caption_target_chars,
                    ),
                    progress_callback=semantic_progress_callback,
                )
                log.info("表格语义增强完成")
            except Exception as exc:  # noqa: BLE001
                log.error(
                    f"表格语义增强失败 | "
                    f"错误: {type(exc).__name__}: {str(exc)[:200]} | "
                    f"策略: {self.app_config.table_semantic_on_error}"
                )
                if self.app_config.table_semantic_on_error == "fail":
                    raise ConversionError(f"表格语义增强失败: {exc!s}") from exc

        return ConvertToMarkdownResult(dst)

    def iter_supported_files(self, input_root: Path):
        for path in sorted(input_root.rglob("*")):
            if not path.is_file():
                continue
            if path.name.startswith("."):
                continue
            if self.is_supported_file(path):
                yield path
