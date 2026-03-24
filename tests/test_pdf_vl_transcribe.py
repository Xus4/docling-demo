import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest.mock import MagicMock

from src.pdf_vl_transcribe import transcribe_pdf_with_vl


class TestPdfVlTranscribe(unittest.TestCase):
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
            out = transcribe_pdf_with_vl(
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
        self.assertIn("# ", out)
        self.assertIn("第 1 / 2 页", out)


if __name__ == "__main__":
    unittest.main()
