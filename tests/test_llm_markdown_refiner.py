import unittest
from typing import Any

from src.llm_markdown_refiner import (
    DoclingMarkdownRefiner,
    _parse_quality_check,
)


class DummyClient:
    def __init__(self, cleanup_return: str, qc_return: str) -> None:
        self._cleanup_return = cleanup_return
        self._qc_return = qc_return
        self.last_messages: list[dict[str, Any]] = []

    def generate_multimodal(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature=None,
        max_tokens=None,
        result_format="message",
    ) -> str:
        self.last_messages = messages
        # very simple routing: first call returns cleanup, second returns qc
        if len(self.last_messages) and "QC_SCORE" not in self._cleanup_return:
            # not used; kept for safety
            pass
        # heuristic: if user content includes "QC_SCORE" request, treat as qc
        joined = "\n".join(str(m.get("content")) for m in messages)
        # 质检请求含统计块；清洗请求不含
        if "【原始统计】" in joined:
            return self._qc_return
        return self._cleanup_return


class TestLlmMarkdownRefiner(unittest.TestCase):
    def test_parse_quality_check(self) -> None:
        text = (
            "QC_SCORE=85\n"
            "NEED_RERUN=no\n"
            "SUGGEST_OCR_QUALITY=balanced\n"
            "SUGGEST_TABLE_ACCURACY=accurate\n"
            "SUGGEST_GENERATE_PAGE_IMAGES=yes\n"
            "SUGGEST_REASON=表格识别正常"
        )
        qc = _parse_quality_check(text)
        self.assertEqual(qc.score, 85)
        self.assertFalse(qc.need_rerun)
        self.assertIsNotNone(qc.suggest)
        assert qc.suggest is not None
        self.assertEqual(qc.suggest.ocr_quality, "balanced")
        self.assertEqual(qc.suggest.table_accuracy, "accurate")
        self.assertTrue(qc.suggest.generate_page_images)

    def test_cleanup_markdown_normalization(self) -> None:
        dummy = DummyClient(
            cleanup_return="```markdown\n![](a.png)\n# H\n```",
            qc_return="QC_SCORE=10\nNEED_RERUN=no\nSUGGEST_OCR_QUALITY=fast\nSUGGEST_TABLE_ACCURACY=fast\nSUGGEST_GENERATE_PAGE_IMAGES=no\nSUGGEST_REASON=none",
        )
        refiner = DoclingMarkdownRefiner(
            client=dummy, model="qwen3-vl-plus", cleanup_max_images=2
        )
        refined = refiner.cleanup_markdown(
            original_markdown="![](a.png)\n",
            markdown_out_path="/tmp/doc.md",
            image_inputs=["/tmp/a.png"],
        )
        self.assertIn("![](a.png)", refined)
        self.assertNotIn("```", refined)

    def test_quality_check_parses_tags(self) -> None:
        qc_text = (
            "QC_SCORE=20\n"
            "NEED_RERUN=yes\n"
            "SUGGEST_OCR_QUALITY=high\n"
            "SUGGEST_TABLE_ACCURACY=accurate\n"
            "SUGGEST_GENERATE_PAGE_IMAGES=no\n"
            "SUGGEST_REASON=OCR疑似失败"
        )
        dummy = DummyClient(
            cleanup_return="![](a.png)\n",
            qc_return=qc_text,
        )
        refiner = DoclingMarkdownRefiner(
            client=dummy, model="qwen3-vl-plus", cleanup_max_images=2
        )
        qc = refiner.quality_check(original_markdown="![](a.png)\n", refined_markdown="![](a.png)\n")
        self.assertTrue(qc.need_rerun)
        self.assertEqual(qc.suggest.ocr_quality, "high")

    def test_cleanup_tables_per_block(self) -> None:
        # 仅验证：table block 会被替换为模型输出，并且骨架不变
        from tempfile import TemporaryDirectory
        from pathlib import Path

        corrected_table = (
            "| A | B |\n"
            "|---|---|\n"
            "| 9 | 2 |\n"
        )

        class DummyTableClient:
            def generate_multimodal(
                self,
                model: str,
                messages: list[dict[str, Any]],
                *,
                temperature=None,
                max_tokens=None,
                result_format="message",
            ) -> str:
                return corrected_table

        dummy = DummyTableClient()
        refiner = DoclingMarkdownRefiner(
            client=dummy,
            model="qwen3-vl-plus",
            cleanup_max_images=2,
        )

        with TemporaryDirectory() as td:
            root = Path(td)
            md_path = root / "out" / "doc.md"
            md_path.parent.mkdir(parents=True, exist_ok=True)
            (md_path.parent / "img").mkdir(parents=True, exist_ok=True)
            (md_path.parent / "img" / "a.png").write_bytes(b"fake")

            original_md = (
                "![](img/a.png)\n"
                "| A | B |\n"
                "|---|---|\n"
                "| 1 | 2 |\n"
            )
            refined = refiner.cleanup_tables_per_block(
                original_markdown=original_md,
                markdown_out_path=md_path,
                vl_image_mode="local_abs",
                cleanup_max_images_per_table=1,
                cleanup_max_tables=10,
                context_lines=0,
            )
            self.assertIn("| 9 | 2 |", refined)
            self.assertIn("![](img/a.png)", refined)


if __name__ == "__main__":
    unittest.main()

