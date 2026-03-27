"""``--pdf-vl-primary`` 下的 CLI 默认值合并（显式传参优先）。"""

from __future__ import annotations

import argparse


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
    if not argv_contains_long_option(av, "--pdf-vl-dpi"):
        args.pdf_vl_dpi = 180.0
    if not argv_contains_long_option(av, "--pdf-vl-workers"):
        args.pdf_vl_workers = 10
    if not argv_contains_long_option(av, "--pdf-vl-table-second-pass-max-tables"):
        args.pdf_vl_table_second_pass_max_tables = 5
    if not argv_contains_long_option(av, "--llm-temperature"):
        args.llm_temperature = 0.0
    if not argv_contains_long_option(av, "--llm-max-tokens"):
        args.llm_max_tokens = 16384
