"""cli_pdf_vl_defaults：显式传参检测（仅 ``--opt value``）。"""

import os
import unittest
from argparse import Namespace
from unittest.mock import patch

from src.cli_pdf_vl_defaults import apply_pdf_vl_cli_defaults, argv_contains_long_option


class TestArgvContainsLongOption(unittest.TestCase):
    def test_space_form(self) -> None:
        self.assertTrue(
            argv_contains_long_option(
                ["python", "main.py", "--pdf-vl-primary", "--pdf-vl-dpi", "200"],
                "--pdf-vl-dpi",
            )
        )

    def test_absent(self) -> None:
        self.assertFalse(
            argv_contains_long_option(
                ["python", "main.py", "--pdf-vl-primary"],
                "--pdf-vl-dpi",
            )
        )


class TestApplyPdfVlDefaults(unittest.TestCase):
    @patch.dict(
        os.environ,
        {
            "PDF_VL_DPI": "",
            "PDF_VL_WORKERS": "",
            "PDF_VL_TABLE_SECOND_PASS_MAX_TABLES": "",
            "LLM_TEMPERATURE": "",
            "LLM_MAX_TOKENS": "",
        },
        clear=False,
    )
    def test_sets_defaults_when_absent(self) -> None:
        args = Namespace(
            pdf_vl_primary=True,
            pdf_vl_dpi=150.0,
            pdf_vl_workers=1,
            pdf_vl_table_second_pass_max_tables=3,
            llm_temperature=0.2,
            llm_max_tokens=8192,
        )
        apply_pdf_vl_cli_defaults(args, ["main.py", "--pdf-vl-primary"])
        self.assertEqual(args.pdf_vl_dpi, 180.0)
        self.assertEqual(args.pdf_vl_workers, 10)
        self.assertEqual(args.pdf_vl_table_second_pass_max_tables, 0)
        self.assertEqual(args.llm_temperature, 0.0)
        self.assertEqual(args.llm_max_tokens, 16384)


if __name__ == "__main__":
    unittest.main()
