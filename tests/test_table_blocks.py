"""阶段 B：GFM + HTML 表格块扫描。"""

from __future__ import annotations

import unittest

from src.table_semantic.table_blocks import TableBlock, iter_table_blocks


class TestIterTableBlocks(unittest.TestCase):
    def test_gfm_only(self) -> None:
        md = "|a|b|\n|---|---|\n|1|2|\n"
        blocks = list(iter_table_blocks(md))
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].kind, "gfm_table")
        self.assertEqual(blocks[0].raw.strip(), md.strip())

    def test_html_only(self) -> None:
        md = "前文\n<table><tr><td>x</td></tr></table>\n后文"
        blocks = list(iter_table_blocks(md))
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].kind, "html_table")
        self.assertIn("<table>", blocks[0].raw)
        self.assertIn("</table>", blocks[0].raw.lower())

    def test_mixed_order(self) -> None:
        md = "|h|\n|---|\n|v|\n\n<table><tbody><tr><td>t</td></tr></tbody></table>\n"
        blocks = list(iter_table_blocks(md))
        self.assertEqual([b.kind for b in blocks], ["gfm_table", "html_table"])

    def test_skip_inside_fence(self) -> None:
        md = (
            "```\n|a|b|\n|---|---|\n|1|2|\n<table></table>\n```\n"
            "|x|y|\n|---|---|\n|1|2|\n"
        )
        blocks = list(iter_table_blocks(md))
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].kind, "gfm_table")

    def test_html_case_insensitive(self) -> None:
        md = "<TABLE><TR><TD>1</TD></TR></TABLE>"
        blocks = list(iter_table_blocks(md))
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].kind, "html_table")

    def test_adjacent_gfm_tables(self) -> None:
        md = "|a|\n|---|\n|1|\n|b|\n|---|\n|2|\n"
        blocks = list(iter_table_blocks(md))
        self.assertEqual(len(blocks), 2)
        self.assertTrue(all(b.kind == "gfm_table" for b in blocks))

    def test_chinese_around(self) -> None:
        md = "说明文字\n\n|列|\n| --- |\n|值|\n\n结束。\n"
        blocks = list(iter_table_blocks(md))
        self.assertEqual(len(blocks), 1)


if __name__ == "__main__":
    unittest.main()
