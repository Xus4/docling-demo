"""``--pdf-vl-primary`` 下的 CLI 默认值合并（显式传参优先）。"""

from __future__ import annotations

import argparse
import os


def argv_contains_long_option(argv: list[str], flag: str) -> bool:
    """
    判断用户是否在命令行中显式写了该长选项。
    """
    return flag in argv


def apply_pdf_vl_cli_defaults(args: argparse.Namespace, argv: list[str]) -> None:
    """
    ``--pdf-vl-primary`` 下的推荐默认（显式传参优先）；未走 pdf-vl 时不改任何字段。
    """
    if not bool(getattr(args, "pdf_vl_primary", False)):
        return
    av = argv

    def _i(name: str, default: int) -> int:
        v = os.getenv(name, "").strip()
        return int(v) if v else default

    def _f(name: str, default: float) -> float:
        v = os.getenv(name, "").strip()
        return float(v) if v else default

    if not argv_contains_long_option(av, "--pdf-vl-dpi"):
        args.pdf_vl_dpi = _f("PDF_VL_DPI", 180.0)
    if not argv_contains_long_option(av, "--pdf-vl-workers"):
        args.pdf_vl_workers = _i("PDF_VL_WORKERS", 10)
    if not argv_contains_long_option(av, "--pdf-vl-table-second-pass-max-tables"):
        args.pdf_vl_table_second_pass_max_tables = _i(
            "PDF_VL_TABLE_SECOND_PASS_MAX_TABLES", 5
        )
    if not argv_contains_long_option(av, "--llm-temperature"):
        v = os.getenv("LLM_TEMPERATURE", "").strip()
        args.llm_temperature = float(v) if v else 0.0
    if not argv_contains_long_option(av, "--llm-max-tokens"):
        args.llm_max_tokens = _i("LLM_MAX_TOKENS", 16384)
