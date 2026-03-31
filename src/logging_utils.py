from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any


def _to_log_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).replace("\n", "\\n")


def kv(**fields: Any) -> str:
    parts: list[str] = []
    for k, v in fields.items():
        parts.append(f"{k}={_to_log_value(v)}")
    return " ".join(parts)


def short_job_id(job_id: str) -> str:
    if len(job_id) <= 10:
        return job_id
    return f"{job_id[:8]}…"


def log_event(logger: logging.Logger, level: int, event: str, **fields: Any) -> None:
    payload = kv(event=event, **fields)
    logger.log(level, payload)


def configure_logging(
    *,
    verbose: bool,
    log_file: Path | None = None,
    app: str = "app",
) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    log_event(
        logging.getLogger(app),
        logging.INFO,
        "logger.ready",
        log_level=logging.getLevelName(level),
    )
