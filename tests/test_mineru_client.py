import io
import tempfile
import unittest
import zipfile
from pathlib import Path

from src.mineru_client import (
    MinerUError,
    build_multipart_form_fields,
    build_multipart_request_items,
    markdown_from_results_json,
    markdown_from_zip_bytes,
    persist_zip_artifacts,
)


class TestMinerUForm(unittest.TestCase):
    def test_build_form_includes_lang_list_entries(self) -> None:
        form = build_multipart_form_fields(
            backend="hybrid-auto-engine",
            parse_method="auto",
            formula_enable=True,
            table_enable=False,
            server_url=None,
            return_md=True,
            return_middle_json=False,
            return_model_output=False,
            return_content_list=False,
            return_images=False,
            response_format_zip=False,
            return_original_file=False,
            start_page_id=0,
            end_page_id=99,
            lang_list=("ch", "en"),
        )
        keys = [k for k, _ in form]
        self.assertIn("lang_list", keys)
        langs = [v for k, v in form if k == "lang_list"]
        self.assertEqual(langs, ["ch", "en"])
        self.assertIn(("table_enable", "false"), form)

    def test_build_multipart_request_items_stable_shape(self) -> None:
        form = build_multipart_form_fields(
            backend="hybrid-http-client",
            parse_method="auto",
            formula_enable=True,
            table_enable=True,
            server_url="http://127.0.0.1:11434/v1",
            return_md=True,
            return_middle_json=False,
            return_model_output=False,
            return_content_list=False,
            return_images=False,
            response_format_zip=True,
            return_original_file=False,
            start_page_id=0,
            end_page_id=99999,
            lang_list=("ch",),
        )
        items = build_multipart_request_items(
            form_fields=form,
            file_name="a.pdf",
            file_bytes=b"123",
            media_type="application/pdf",
        )
        # 文本字段为 (None, value)，文件字段为 (filename, bytes, media_type)
        self.assertIn(("backend", (None, "hybrid-http-client")), items)
        self.assertIn(("files", ("a.pdf", b"123", "application/pdf")), items)


class TestMinerUMarkdownExtract(unittest.TestCase):
    def test_results_json_by_stem(self) -> None:
        md = markdown_from_results_json(
            {"results": {"doc": {"md_content": "# Hi"}}},
            "doc",
        )
        self.assertEqual(md, "# Hi")

    def test_results_json_single_entry(self) -> None:
        md = markdown_from_results_json(
            {"results": {"x": {"md_content": "# X"}}},
            "other",
        )
        self.assertEqual(md, "# X")

    def test_results_json_missing_raises(self) -> None:
        with self.assertRaises(MinerUError):
            markdown_from_results_json({"results": {}}, "a")

    def test_zip_bytes(self) -> None:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("doc/auto/doc.md", "# Z")
        md = markdown_from_zip_bytes(buf.getvalue(), "doc")
        self.assertEqual(md, "# Z")

    def test_persist_zip_artifacts_extracts_all_files(self) -> None:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("doc/auto/doc.md", "# Main")
            zf.writestr("doc/auto/images/a.png", b"\x89PNG")
            zf.writestr("doc/auto/doc_middle.json", '{"x":1}')
        with tempfile.TemporaryDirectory() as td:
            output_path = Path(td) / "doc.md"
            persist_zip_artifacts(
                zip_bytes=buf.getvalue(),
                output_path=output_path,
                prefer_stem="doc",
            )
            self.assertTrue(output_path.is_file())
            self.assertEqual(output_path.read_text(encoding="utf-8"), "# Main")
            self.assertTrue((Path(td) / "doc/auto/images/a.png").is_file())
            self.assertTrue((Path(td) / "doc/auto/doc_middle.json").is_file())

    def test_persist_zip_artifacts_without_md_raises(self) -> None:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("doc/auto/a.txt", "x")
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(MinerUError):
                persist_zip_artifacts(
                    zip_bytes=buf.getvalue(),
                    output_path=Path(td) / "doc.md",
                    prefer_stem="doc",
                )


if __name__ == "__main__":
    unittest.main()
