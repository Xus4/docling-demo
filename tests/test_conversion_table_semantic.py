"""阶段 E：ConversionService 在启用配置时调用表格语义增强。"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from config import AppConfig
from src.core.service import ConversionService


class TestConversionTableSemantic(unittest.TestCase):
    def test_augment_called_when_enabled(self) -> None:
        extras = {
            "TABLE_SEMANTIC_ENABLE": "true",
            "TABLE_SEMANTIC_BASE_URL": "http://127.0.0.1:1/v1",
            "TABLE_SEMANTIC_MODEL": "test-model",
        }

        def fake_convert(*_a: object, **kw: object) -> None:
            p = kw["output_path"]
            assert isinstance(p, Path)
            cb = kw.get("on_processing_stage")
            if callable(cb):
                cb("mineru_materialize")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("|a|\n|---|\n|1|\n", encoding="utf-8")

        with mock.patch.dict(os.environ, extras, clear=False):
            cfg = AppConfig.from_env()
        self.assertTrue(cfg.table_semantic_enable)

        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "x.pdf"
            dst = Path(td) / "x.md"
            src.write_bytes(b"%PDF-1.4\n")
            svc = ConversionService(cfg)
            with mock.patch(
                "src.core.service.run_mineru_convert", side_effect=fake_convert
            ):
                with mock.patch("src.core.service.augment_markdown_file") as aug:
                    svc.convert_to_markdown(str(src), str(dst))
            aug.assert_called_once()

if __name__ == "__main__":
    unittest.main()
