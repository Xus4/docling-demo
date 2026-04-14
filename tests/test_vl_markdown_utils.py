import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from src.vl_markdown_utils import (
    extract_markdown_image_refs,
    extract_markdown_image_refs_with_line_index,
    resolve_image_refs,
    extract_markdown_table_blocks,
    summarize_markdown_quality,
    table_column_count,
    validate_table_output_invariants,
    validate_image_refs_invariants,
)


class TestVlMarkdownUtils(unittest.TestCase):
    def test_extract_markdown_image_refs(self) -> None:
        md = """
        # Title
        ![](images/a.png)
        Some text
        ![](images/b.jpg "caption")
        """
        refs = extract_markdown_image_refs(md)
        self.assertEqual(refs, ["images/a.png", "images/b.jpg"])

    def test_resolve_image_refs_local_abs(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            md_dir = root / "out"
            md_dir.mkdir(parents=True, exist_ok=True)
            (md_dir / "img").mkdir(parents=True, exist_ok=True)
            a = md_dir / "img" / "a.png"
            a.write_bytes(b"fake")

            md_path = md_dir / "doc.md"
            md_text = "![](img/a.png)\n"
            refs = extract_markdown_image_refs(md_text)
            resolved = resolve_image_refs(
                refs=refs,
                markdown_out_path=md_path,
                max_images=10,
                image_mode="local_abs",
            )
            self.assertEqual(len(resolved), 1)
            self.assertEqual(resolved[0], str(a.resolve()))

    def test_validate_image_refs_invariants(self) -> None:
        original = "![](a.png)\n![](b.png)\n"
        refined_ok = "![](a.png)\n![](b.png)\n"
        refined_bad = "![](a.png)\n"
        ok, _ = validate_image_refs_invariants(
            original_md=original,
            refined_md=refined_ok,
        )
        self.assertTrue(ok)
        ok2, _ = validate_image_refs_invariants(
            original_md=original,
            refined_md=refined_bad,
        )
        self.assertFalse(ok2)

    def test_extract_markdown_image_refs_with_line_index(self) -> None:
        md = "![](a.png)\ntext\n![](b.png)\n"
        refs = extract_markdown_image_refs_with_line_index(md)
        self.assertEqual(refs, [(0, "a.png"), (2, "b.png")])

    def test_extract_markdown_table_blocks(self) -> None:
        md = (
            "para\n"
            "| A | B |\n"
            "|---|---|\n"
            "| 1 | 2 |\n"
            "tail\n"
        )
        blocks = extract_markdown_table_blocks(md)
        self.assertEqual(len(blocks), 1)
        assert blocks[0].start_line == 1
        assert blocks[0].end_line == 3
        self.assertIn("| A | B |", blocks[0].markdown)

    def test_table_column_count_and_validate(self) -> None:
        original = "| A | B |\n|---|---|\n| 1 | 2 |\n"
        self.assertEqual(table_column_count(original), 2)
        ok, _ = validate_table_output_invariants(original_table=original, refined_table=original)
        self.assertTrue(ok)

        refined_bad = "| A | B |\n|---|---|\n| 1 |\n"
        ok2, _ = validate_table_output_invariants(
            original_table=original,
            refined_table=refined_bad,
        )
        self.assertFalse(ok2)

    def test_extract_html_table_blocks(self) -> None:
        md = (
            "intro\n\n"
            "<table>\n"
            "  <thead><tr><th>A</th><th>B</th></tr></thead>\n"
            "  <tbody><tr><td>1</td><td>2</td></tr></tbody>\n"
            "</table>\n\n"
            "tail\n"
        )
        blocks = extract_markdown_table_blocks(md)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].kind, "html")
        self.assertIn("<table>", blocks[0].markdown)

    def test_html_table_column_count_and_validate(self) -> None:
        original = (
            "<table>\n"
            "  <tr><th>A</th><th colspan=\"2\">B</th></tr>\n"
            "  <tr><td>1</td><td>2</td><td>3</td></tr>\n"
            "</table>\n"
        )
        refined_ok = (
            "<table>\n"
            "  <tr><th>A</th><th colspan=\"2\">B</th></tr>\n"
            "  <tr><td>x</td><td>y</td><td>z</td></tr>\n"
            "</table>\n"
        )
        refined_bad = (
            "<table>\n"
            "  <tr><th>A</th><th>B</th></tr>\n"
            "  <tr><td>x</td><td>y</td></tr>\n"
            "</table>\n"
        )
        self.assertEqual(table_column_count(original), 3)
        ok, _ = validate_table_output_invariants(
            original_table=original,
            refined_table=refined_ok,
        )
        self.assertTrue(ok)
        ok2, _ = validate_table_output_invariants(
            original_table=original,
            refined_table=refined_bad,
        )
        self.assertFalse(ok2)

    def test_summarize_markdown_quality_counts_html_tr_rows(self) -> None:
        md = (
            "<table>\n"
            "  <tr><th>A</th></tr>\n"
            "  <tr><td>1</td></tr>\n"
            "</table>\n"
        )
        stats = summarize_markdown_quality(md)
        self.assertGreaterEqual(stats["table_rows"], 2)


if __name__ == "__main__":
    unittest.main()
