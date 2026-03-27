from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

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
    pdf_vl_table_second_pass_max_tables: int
    max_num_pages: int | None
    llm_max_tokens: int | None
    llm_temperature: float
    llm_table_caption: bool
    llm_table_caption_max_chars: int
    pdf_caption_crop_figures: bool
    llm_enable_thinking: bool
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
            llm_model=os.getenv("LLM_MODEL", "qwen3.5-35b-a3b"),
            pdf_vl_table_second_pass_max_tables=max(
                1, int(os.getenv("PDF_VL_TABLE_SECOND_PASS_MAX_TABLES", "5"))
            ),
            max_num_pages=(int(max_num_pages_raw) if max_num_pages_raw else None),
            llm_max_tokens=(int(llm_max_tokens_raw) if llm_max_tokens_raw else 16384),
            llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
            llm_table_caption=_parse_bool(os.getenv("LLM_TABLE_CAPTION"), default=True),
            llm_table_caption_max_chars=max(
                20, int(os.getenv("LLM_TABLE_CAPTION_MAX_CHARS", "500"))
            ),
            pdf_caption_crop_figures=_parse_bool(
                os.getenv("PDF_CAPTION_CROP_FIGURES"), default=True
            ),
            llm_enable_thinking=_parse_bool(
                os.getenv("LLM_ENABLE_THINKING"), default=True
            ),
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

        return ConverterConfig(
            pdf_vl_primary=self.pdf_vl_primary,
            pdf_vl_dpi=self.pdf_vl_dpi,
            pdf_vl_workers=self.pdf_vl_workers,
            llm_model=self.llm_model,
            pdf_vl_table_second_pass_max_tables=self.pdf_vl_table_second_pass_max_tables,
            max_num_pages=self.max_num_pages,
            llm_max_tokens=self.llm_max_tokens,
            llm_temperature=self.llm_temperature,
            llm_table_caption=self.llm_table_caption,
            llm_table_caption_max_chars=self.llm_table_caption_max_chars,
            pdf_caption_crop_figures=self.pdf_caption_crop_figures,
            llm_enable_thinking=self.llm_enable_thinking,
        )
