from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

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


def _parse_lang_list(raw: str | None) -> tuple[str, ...]:
    if not raw or not str(raw).strip():
        return ("ch",)
    parts = [x.strip() for x in str(raw).split(",") if x.strip()]
    return tuple(parts) if parts else ("ch",)


def _normalize_mineru_parse_mode(value: str | None) -> Literal["async", "sync"]:
    v = (value or "async").strip().lower()
    if v == "sync":
        return "sync"
    return "async"


def _normalize_db_type(value: str | None) -> str:
    raw = (value or "sqlite").strip().lower()
    if raw in {"sqlite", "mysql"}:
        return raw
    raise ValueError(f"Invalid DB_TYPE: {value!r}. Allowed values: sqlite, mysql")


@dataclass(frozen=True)
class AppConfig:
    """Web 服务、上传目录与认证；文档解析由后续 MinerU API 接入。"""

    max_file_size_bytes: int
    allowed_types: set[str]
    debug: bool
    data_dir: Path
    input_dir: Path
    output_dir: Path
    worker_max_parallel_jobs: int
    db_type: str
    database_url: str
    auth_db_path: Path
    session_secret: str
    access_token_secret: str
    access_token_ttl_sec: int
    initial_password: str
    auth_admin_username: str
    auth_users: list[str]
    oa_auth_enabled: bool
    oa_auth_login_url: str
    oa_auth_tenant_id: str
    oa_auth_tenant_name: str
    oa_auth_remember_me: bool
    oa_auth_verify_ssl: bool
    oa_auth_timeout_sec: float
    oa_auth_origin: str
    oa_auth_referer: str
    oa_auth_user_agent: str
    oa_auth_cookie: str
    oa_auth_trust_env: bool
    # --- MinerU（HTTP 客户端 → mineru-api；字段与官方表单一一对应）---
    mineru_base_url: str  # 服务根 URL；默认内网部署地址
    mineru_api_key: str | None  # 可选 Bearer；无网关鉴权时 None
    mineru_timeout_sec: float  # 单次请求超时
    mineru_poll_interval_sec: float  # async 轮询间隔
    mineru_max_wait_sec: float  # async 等待终态上限
    mineru_verify_ssl: bool  # HTTPS 校验开关
    mineru_parse_mode: Literal["async", "sync"]  # async=/tasks+轮询；sync=/file_parse
    mineru_backend: str  # pipeline / vlm-* / hybrid-* 等
    mineru_parse_method: str  # auto | txt | ocr
    mineru_formula_enable: bool
    mineru_table_enable: bool
    mineru_server_url: str | None  # *-http-client 时 OpenAI 兼容根 URL
    mineru_lang_list: tuple[str, ...]  # 多语言 OCR，多项为多个 lang_list 表单
    mineru_return_md: bool
    mineru_return_middle_json: bool
    mineru_return_model_output: bool
    mineru_return_content_list: bool
    mineru_return_images: bool
    mineru_response_format_zip: bool
    mineru_return_original_file: bool  # 仅 zip 模式有意义
    mineru_start_page_id: int  # PDF 起始页，从 0 起
    mineru_end_page_id: int  # PDF 结束页，从 0 起

    @classmethod
    def from_env(cls) -> "AppConfig":
        data_dir = Path(os.getenv("DATA_DIR", str(ROOT / "data"))).resolve()
        input_dir = Path(os.getenv("INPUT_DIR", str(data_dir / "input"))).resolve()
        output_dir = Path(os.getenv("OUTPUT_DIR", str(data_dir / "output"))).resolve()
        max_file_size_bytes = _parse_size_to_bytes(
            os.getenv("MAX_FILE_SIZE", "20MB"),
            default_bytes=20 * 1024 * 1024,
        )
        _oa_tenant_name_raw = os.getenv("OA_AUTH_TENANT_NAME")
        if _oa_tenant_name_raw is None:
            oa_tenant_name = "    "
        else:
            oa_tenant_name = str(_oa_tenant_name_raw).rstrip("\r\n")
        auth_db_path = Path(os.getenv("AUTH_DB_PATH", str(data_dir / "auth.db"))).resolve()
        db_type = _normalize_db_type(os.getenv("DB_TYPE", "sqlite"))
        if db_type == "sqlite":
            database_url = "sqlite:///" + auth_db_path.as_posix()
        elif db_type == "mysql":
            database_url = os.getenv("DATABASE_URL", "").strip()
            if not database_url:
                mysql_user = env_str("MYSQL_USER", "root")
                mysql_password = os.getenv("MYSQL_PASSWORD", "")
                mysql_host = env_str("MYSQL_HOST", "127.0.0.1")
                mysql_port = env_int("MYSQL_PORT", 3306)
                mysql_db = env_str("MYSQL_DATABASE", "docling_demo")
                if mysql_password:
                    database_url = (
                        f"mysql+pymysql://{mysql_user}:{mysql_password}"
                        f"@{mysql_host}:{mysql_port}/{mysql_db}?charset=utf8mb4"
                    )
                else:
                    database_url = (
                        f"mysql+pymysql://{mysql_user}"
                        f"@{mysql_host}:{mysql_port}/{mysql_db}?charset=utf8mb4"
                    )
        else:
            database_url = "sqlite:///" + auth_db_path.as_posix()
        return cls(
            max_file_size_bytes=max_file_size_bytes,
            allowed_types=_parse_allowed_types(os.getenv("ALLOWED_TYPES")),
            debug=_parse_bool(os.getenv("DEBUG"), default=False),
            data_dir=data_dir,
            input_dir=input_dir,
            output_dir=output_dir,
            worker_max_parallel_jobs=max(
                1,
                env_int("WORKER_MAX_PARALLEL_JOBS", 1),
            ),
            db_type=db_type,
            database_url=database_url,
            auth_db_path=auth_db_path,
            session_secret=os.getenv("SESSION_SECRET", "change-me-in-production"),
            access_token_secret=(
                os.getenv("ACCESS_TOKEN_SECRET", "").strip()
                or os.getenv("SESSION_SECRET", "change-me-in-production")
            ),
            access_token_ttl_sec=max(
                60,
                env_int("ACCESS_TOKEN_TTL_SEC", 86400),
            ),
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
            oa_auth_enabled=env_bool("OA_AUTH_ENABLED", False),
            oa_auth_login_url=os.getenv("OA_AUTH_LOGIN_URL", "").strip(),
            oa_auth_tenant_id=os.getenv("OA_AUTH_TENANT_ID", "1").strip() or "1",
            oa_auth_tenant_name=oa_tenant_name,
            oa_auth_remember_me=env_bool("OA_AUTH_REMEMBER_ME", True),
            oa_auth_verify_ssl=env_bool("OA_AUTH_VERIFY_SSL", False),
            oa_auth_timeout_sec=max(3.0, env_float("OA_AUTH_TIMEOUT_SEC", 15.0)),
            oa_auth_origin=os.getenv("OA_AUTH_ORIGIN", "").strip(),
            oa_auth_referer=os.getenv("OA_AUTH_REFERER", "").strip(),
            oa_auth_user_agent=os.getenv("OA_AUTH_USER_AGENT", "").strip(),
            oa_auth_cookie=os.getenv("OA_AUTH_COOKIE", "").strip(),
            oa_auth_trust_env=env_bool("OA_AUTH_TRUST_ENV", False),
            # --- MinerU：与 mineru-api multipart 表单一致（详见 .env 中长注释）---
            mineru_base_url=env_str(
                "MINERU_BASE_URL",
                "http://192.168.2.60:8011",
            ),
            mineru_api_key=(os.getenv("MINERU_API_KEY") or "").strip() or None,
            mineru_timeout_sec=max(30.0, env_float("MINERU_TIMEOUT_SEC", 300.0)),
            mineru_poll_interval_sec=max(0.2, env_float("MINERU_POLL_INTERVAL_SEC", 1.5)),
            mineru_max_wait_sec=max(60.0, env_float("MINERU_MAX_WAIT_SEC", 3600.0)),
            mineru_verify_ssl=env_bool("MINERU_VERIFY_SSL", True),
            mineru_parse_mode=_normalize_mineru_parse_mode(os.getenv("MINERU_PARSE_MODE")),
            mineru_backend=env_str("MINERU_BACKEND", "hybrid-auto-engine"),
            mineru_parse_method=env_str("MINERU_PARSE_METHOD", "auto"),
            mineru_formula_enable=env_bool("MINERU_FORMULA_ENABLE", True),
            mineru_table_enable=env_bool("MINERU_TABLE_ENABLE", True),
            mineru_server_url=(os.getenv("MINERU_SERVER_URL") or "").strip() or None,
            mineru_lang_list=_parse_lang_list(os.getenv("MINERU_LANG_LIST")),
            mineru_return_md=env_bool("MINERU_RETURN_MD", True),
            mineru_return_middle_json=env_bool("MINERU_RETURN_MIDDLE_JSON", False),
            mineru_return_model_output=env_bool("MINERU_RETURN_MODEL_OUTPUT", False),
            mineru_return_content_list=env_bool("MINERU_RETURN_CONTENT_LIST", False),
            mineru_return_images=env_bool("MINERU_RETURN_IMAGES", False),
            mineru_response_format_zip=env_bool("MINERU_RESPONSE_FORMAT_ZIP", False),
            mineru_return_original_file=env_bool("MINERU_RETURN_ORIGINAL_FILE", False),
            mineru_start_page_id=max(0, env_int("MINERU_START_PAGE_ID", 0)),
            mineru_end_page_id=max(0, env_int("MINERU_END_PAGE_ID", 99999)),
        )

    def ensure_dirs(self) -> None:
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
