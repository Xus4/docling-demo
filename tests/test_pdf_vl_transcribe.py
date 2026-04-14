import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest.mock import MagicMock

from src.pdf_vl_transcribe import (
    _is_suspicious_table,
    _normalize_gfm_tables,
    _table_cleanup_messages_with_page_image,
    transcribe_pdf_with_vl,
)


class TestPdfVlTranscribe(unittest.TestCase):
    def test_normalize_gfm_tables_aligns_columns(self) -> None:
        raw = (
            "| A | B | C |\n"
            "| --- | --- | --- |\n"
            "| 1 | 2 |\n"
            "| 3 | 4 | 5 | 6 |\n"
        )
        got = _normalize_gfm_tables(raw)
        self.assertIn("| 1 | 2 |  |", got)
        self.assertIn("| 3 | 4 | 5 | 6 |", got)

    def test_transcribe_calls_multimodal_per_page(self) -> None:
        try:
            import fitz
        except ImportError:
            self.skipTest("pymupdf not installed")

        client = MagicMock()
        client.generate_multimodal.return_value = "# 标题\n\n正文"

        doc = fitz.open()
        doc.new_page()
        doc.new_page()
        tmp = NamedTemporaryFile(suffix=".pdf", delete=False)
        tmp.close()
        path = Path(tmp.name)
        try:
            doc.save(str(path))
            doc.close()
            out, failed_pages = transcribe_pdf_with_vl(
                client=client,
                model="qwen-vl-test",
                pdf_path=path,
                dpi=72.0,
                max_pages=2,
                temperature=0.1,
                max_tokens=512,
            )
        finally:
            path.unlink(missing_ok=True)

        self.assertEqual(client.generate_multimodal.call_count, 2)
        self.assertEqual(failed_pages, [])
        self.assertIn("# ", out)
        self.assertIn("第 1 / 2 页", out)

    def test_is_suspicious_table(self) -> None:
        bad = (
            "| A | B | C |\n"
            "| --- | --- | --- |\n"
            "| 1 | 2 |\n"
        )
        good = (
            "| A | B |\n"
            "| --- | --- |\n"
            "| 1 | 2 |\n"
        )
        self.assertTrue(_is_suspicious_table(bad))
        self.assertFalse(_is_suspicious_table(good))

    def test_wide_table_is_suspicious(self) -> None:
        """列数≥8 的核查表走二次校对（与页图对照），减轻右侧勾选漏检。"""
        header = "| " + " | ".join([f"c{i}" for i in range(8)]) + " |"
        sep = "| " + " | ".join(["---"] * 8) + " |"
        row = "| " + " | ".join([str(i) for i in range(8)]) + " |"
        wide = f"{header}\n{sep}\n{row}\n"
        self.assertTrue(_is_suspicious_table(wide))

    def test_table_cleanup_messages_include_page_image(self) -> None:
        img = Path("/tmp/page.png")
        msgs = _table_cleanup_messages_with_page_image(
            original_table_markdown="| a | b |\n| --- | --- |\n| 1 | 2 |\n",
            context_text="ctx",
            page_image_path=img,
        )
        user_msgs = [m for m in msgs if m.get("role") == "user"]
        self.assertEqual(len(user_msgs), 1)
        content = user_msgs[0]["content"]
        self.assertIsInstance(content, list)
        self.assertEqual(content[0].get("image"), str(img))
        self.assertIn("原始表格", content[1].get("text", ""))


if __name__ == "__main__":
    unittest.main()
