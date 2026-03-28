from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from dotenv import load_dotenv

if TYPE_CHECKING:
    from src.converter import ConverterConfig

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", override=False)


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_size_to_bytes(size_str: str | None, default_bytes: int) -> int:
    if not size_str:
        return default_bytes

    raw = size_str.strip().upper()
    units = [
        ("GB", 1024 * 1024 * 1024),
        ("MB", 1024 * 1024),
        ("KB", 1024),
        ("B", 1),
    ]

    for unit, factor in units:
        if raw.endswith(unit):
            number = raw[: -len(unit)].strip()
            if not number:
                raise ValueError(f"Invalid size value: {size_str}")
            return int(float(number) * factor)

    return int(raw)


def env_str(key: str, default: str) -> str:
    """读取环境变量字符串；空或未设置时用 default（供 Web/CLI 共用）。"""
    v = os.getenv(key)
    if v is None or not str(v).strip():
        return default
    return str(v).strip()


def env_int(key: str, default: int) -> int:
    v = os.getenv(key, "").strip()
    return int(v) if v else default


def env_float(key: str, default: float) -> float:
    v = os.getenv(key, "").strip()
    return float(v) if v else default


def env_optional_int(key: str, default: int | None) -> int | None:
    """未设置或空字符串时返回 default；否则解析为 int。"""
    v = os.getenv(key, "").strip()
    if not v:
        return default
    return int(v)


def env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None or not str(raw).strip():
        return default
    return _parse_bool(raw, default=default)


def _parse_allowed_types(value: str | None) -> set[str]:
    default_types = {
        "pdf",
        "docx",
        "pptx",
        "html",
        "htm",
        "png",
        "jpg",
        "jpeg",
        "tif",
        "tiff",
        "bmp",
        "webp",
        "xlsx",
    }
    if not value:
        return default_types

    parsed = {
        item.strip().lower().lstrip(".")
        for item in value.split(",")
        if item.strip()
    }
    return parsed or default_types


@dataclass(frozen=True)
class AppConfig:
    max_file_size_bytes: int
    allowed_types: set[str]
    debug: bool
    data_dir: Path
    input_dir: Path
    output_dir: Path
    auto_cleanup: bool
    cleanup_max_age_hours: int
    pdf_vl_primary: bool
    pdf_vl_dpi: float
    pdf_vl_workers: int
    llm_model: str
    llm_base_url: str
    llm_api_key_env: str
    pdf_vl_table_second_pass_max_tables: int
    max_num_pages: int | None
    llm_max_tokens: int | None
    llm_temperature: float
    llm_max_retries: int
    llm_retry_backoff_sec: float
    llm_max_reasoning_tokens: int | None
    llm_table_caption: bool
    llm_table_caption_max_chars: int
    llm_table_caption_max_tables: int
    llm_table_caption_context_lines: int
    pdf_caption_crop_figures: bool
    pdf_caption_crop_max_per_page: int
    llm_enable_thinking: bool
    llm_timeout_sec: float
    llm_empty_content_max_attempts: int
    llm_vl_image_mode: str
    llm_cleanup_max_images: int
    enable_llm_refine: bool
    llm_table_refine: bool
    llm_table_cleanup_max_tables: int
    llm_table_cleanup_max_images_per_table: int
    llm_table_context_lines: int
    llm_allow_rerun: bool
    llm_rerun_max_attempts: int
    pdf_vl_table_second_pass: bool
    pdf_vl_embed_page_images: bool
    auth_db_path: Path
    session_secret: str
    initial_password: str
    auth_admin_username: str
    auth_users: list[str]

    @classmethod
    def from_env(cls) -> "AppConfig":
        data_dir = Path(os.getenv("DATA_DIR", str(ROOT / "data"))).resolve()
        input_dir = Path(os.getenv("INPUT_DIR", str(data_dir / "input"))).resolve()
        output_dir = Path(os.getenv("OUTPUT_DIR", str(data_dir / "output"))).resolve()
        max_file_size_bytes = _parse_size_to_bytes(
            os.getenv("MAX_FILE_SIZE", "20MB"),
            default_bytes=20 * 1024 * 1024,
        )
        cleanup_hours = int(os.getenv("CLEANUP_MAX_AGE_HOURS", "24"))
        max_num_pages_raw = os.getenv("MAX_NUM_PAGES", "").strip()
        llm_max_tokens_raw = os.getenv("LLM_MAX_TOKENS", "").strip()
        llm_base_url_raw = os.getenv("LLM_BASE_URL", "").strip()
        _default_llm_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        return cls(
            max_file_size_bytes=max_file_size_bytes,
            allowed_types=_parse_allowed_types(os.getenv("ALLOWED_TYPES")),
            debug=_parse_bool(os.getenv("DEBUG"), default=False),
            data_dir=data_dir,
            input_dir=input_dir,
            output_dir=output_dir,
            auto_cleanup=_parse_bool(os.getenv("AUTO_CLEANUP"), default=False),
            cleanup_max_age_hours=max(1, cleanup_hours),
            pdf_vl_primary=_parse_bool(os.getenv("PDF_VL_PRIMARY"), default=True),
            pdf_vl_dpi=float(os.getenv("PDF_VL_DPI", "180")),
            pdf_vl_workers=max(1, int(os.getenv("PDF_VL_WORKERS", "10"))),
            llm_model=env_str("LLM_MODEL", "qwen3.5-35b-a3b"),
            llm_base_url=llm_base_url_raw or _default_llm_base_url,
            llm_api_key_env=env_str("LLM_API_KEY_ENV", "DASHSCOPE_API_KEY"),
            pdf_vl_table_second_pass_max_tables=max(
                1, env_int("PDF_VL_TABLE_SECOND_PASS_MAX_TABLES", 5)
            ),
            max_num_pages=(int(max_num_pages_raw) if max_num_pages_raw else None),
            llm_max_tokens=(int(llm_max_tokens_raw) if llm_max_tokens_raw else 16384),
            llm_temperature=env_float("LLM_TEMPERATURE", 0.0),
            llm_max_retries=max(1, env_int("LLM_MAX_RETRIES", 3)),
            llm_retry_backoff_sec=max(0.1, env_float("LLM_RETRY_BACKOFF_SEC", 1.5)),
            llm_max_reasoning_tokens=env_int("LLM_MAX_REASONING_TOKENS", 256),
            llm_table_caption=env_bool("LLM_TABLE_CAPTION", True),
            llm_table_caption_max_chars=max(
                20, env_int("LLM_TABLE_CAPTION_MAX_CHARS", 500)
            ),
            llm_table_caption_max_tables=max(
                0, env_int("LLM_TABLE_CAPTION_MAX_TABLES", 20)
            ),
            llm_table_caption_context_lines=max(
                0, env_int("LLM_TABLE_CAPTION_CONTEXT_LINES", 3)
            ),
            pdf_caption_crop_figures=env_bool("PDF_CAPTION_CROP_FIGURES", True),
            pdf_caption_crop_max_per_page=max(
                1, env_int("PDF_CAPTION_CROP_MAX_PER_PAGE", 4)
            ),
            llm_enable_thinking=env_bool("LLM_ENABLE_THINKING", True),
            llm_timeout_sec=max(30.0, env_float("LLM_TIMEOUT_SEC", 300.0)),
            llm_empty_content_max_attempts=max(
                1, min(10, env_int("LLM_EMPTY_CONTENT_MAX_ATTEMPTS", 3))
            ),
            llm_vl_image_mode=env_str("LLM_VL_IMAGE_MODE", "local_abs"),
            llm_cleanup_max_images=max(1, env_int("LLM_CLEANUP_MAX_IMAGES", 6)),
            enable_llm_refine=env_bool("LLM_ENABLE_REFINE", False),
            llm_table_refine=env_bool("LLM_TABLE_REFINE", False),
            llm_table_cleanup_max_tables=max(
                1, env_int("LLM_TABLE_CLEANUP_MAX_TABLES", 10)
            ),
            llm_table_cleanup_max_images_per_table=max(
                1, env_int("LLM_TABLE_CLEANUP_MAX_IMAGES_PER_TABLE", 6)
            ),
            llm_table_context_lines=max(0, env_int("LLM_TABLE_CONTEXT_LINES", 2)),
            llm_allow_rerun=env_bool("LLM_ALLOW_RERUN", False),
            llm_rerun_max_attempts=max(0, env_int("LLM_RERUN_MAX_ATTEMPTS", 1)),
            pdf_vl_table_second_pass=env_bool("PDF_VL_TABLE_SECOND_PASS", True),
            pdf_vl_embed_page_images=env_bool("PDF_VL_EMBED_PAGE_IMAGES", False),
            auth_db_path=Path(
                os.getenv("AUTH_DB_PATH", str(data_dir / "auth.db"))
            ).resolve(),
            session_secret=os.getenv("SESSION_SECRET", "change-me-in-production"),
            initial_password=os.getenv("INITIAL_PASSWORD", "ChangeMe123!"),
            auth_admin_username=os.getenv("AUTH_ADMIN_USERNAME", "admin"),
            auth_users=[
                x.strip()
                for x in os.getenv(
                    "AUTH_USERS",
                    "user1,user2,user3,user4,user5",
                ).split(",")
                if x.strip()
            ],
        )

    def ensure_dirs(self) -> None:
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_converter_config(self) -> "ConverterConfig":
        from src.converter import ConverterConfig

        vim = self.llm_vl_image_mode.strip().lower()
        if vim not in ("local_abs", "url"):
            vim = "local_abs"
        vim_mode = cast(Literal["local_abs", "url"], vim)

        return ConverterConfig(
            enable_llm_refine=self.enable_llm_refine,
            llm_api_key_env=self.llm_api_key_env,
            pdf_vl_primary=self.pdf_vl_primary,
            pdf_vl_dpi=self.pdf_vl_dpi,
            pdf_vl_workers=self.pdf_vl_workers,
            llm_model=self.llm_model,
            llm_base_url=self.llm_base_url,
            pdf_vl_table_second_pass_max_tables=self.pdf_vl_table_second_pass_max_tables,
            max_num_pages=self.max_num_pages,
            llm_max_tokens=self.llm_max_tokens,
            llm_temperature=self.llm_temperature,
            llm_max_retries=self.llm_max_retries,
            llm_retry_backoff_sec=self.llm_retry_backoff_sec,
            llm_max_reasoning_tokens=self.llm_max_reasoning_tokens,
            llm_table_caption=self.llm_table_caption,
            llm_table_caption_max_chars=self.llm_table_caption_max_chars,
            llm_table_caption_max_tables=self.llm_table_caption_max_tables,
            llm_table_caption_context_lines=self.llm_table_caption_context_lines,
            pdf_caption_crop_figures=self.pdf_caption_crop_figures,
            pdf_caption_crop_max_per_page=self.pdf_caption_crop_max_per_page,
            llm_enable_thinking=self.llm_enable_thinking,
            llm_timeout_sec=self.llm_timeout_sec,
            llm_empty_content_max_attempts=self.llm_empty_content_max_attempts,
            llm_vl_image_mode=vim_mode,
            llm_cleanup_max_images=self.llm_cleanup_max_images,
            llm_table_refine=self.llm_table_refine,
            llm_table_cleanup_max_tables=self.llm_table_cleanup_max_tables,
            llm_table_cleanup_max_images_per_table=self.llm_table_cleanup_max_images_per_table,
            llm_table_context_lines=self.llm_table_context_lines,
            llm_allow_rerun=self.llm_allow_rerun,
            llm_rerun_max_attempts=self.llm_rerun_max_attempts,
            pdf_vl_table_second_pass=self.pdf_vl_table_second_pass,
            pdf_vl_embed_page_images=self.pdf_vl_embed_page_images,
        )
