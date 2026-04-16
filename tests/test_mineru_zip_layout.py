"""阶段 A：MinerU ZIP 解压目录归一化。"""

from __future__ import annotations

import io
import tempfile
import unittest
import zipfile
from pathlib import Path

from src.mineru_zip_layout import normalize_mineru_zip_layout
from src.mineru_client import persist_zip_artifacts


class TestNormalizeMineruZipLayout(unittest.TestCase):
    def test_flatten_outer_removed(self) -> None:
        """foo.md（外层）+ foo/foo.md + foo/images/a.png → 根下 foo.md、images/a.png，foo/ 消失。"""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            outer = root / "foo.md"
            outer.write_text("outer discard", encoding="utf-8")
            inner = root / "foo"
            inner.mkdir()
            (inner / "foo.md").write_text("inner truth", encoding="utf-8")
            img = inner / "images"
            img.mkdir()
            (img / "a.png").write_bytes(b"\x89PNG")

            self.assertTrue(normalize_mineru_zip_layout(root, "foo"))

            # 外层副本已删除；根下 foo.md 现为内层内容；foo/ 目录已扁平化消失
            self.assertFalse(inner.exists())
            self.assertEqual(outer.read_text(encoding="utf-8"), "inner truth")
            self.assertTrue((root / "images" / "a.png").is_file())

    def test_flatten_without_outer_md(self) -> None:
        """仅有 foo/foo.md + foo/images/ 时仍扁平化，不要求外层 foo.md。"""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            inner = root / "foo"
            inner.mkdir()
            (inner / "foo.md").write_text("only inner", encoding="utf-8")
            (inner / "images").mkdir()
            (inner / "images" / "x.png").write_bytes(b"x")

            self.assertTrue(normalize_mineru_zip_layout(root, "foo"))

            self.assertFalse(inner.exists())
            self.assertEqual((root / "foo.md").read_text(encoding="utf-8"), "only inner")
            self.assertTrue((root / "images" / "x.png").is_file())

    def test_noop_single_top_level_md_only(self) -> None:
        """仅根下 foo.md、无 foo/ 目录时不应改动。"""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            md = root / "foo.md"
            md.write_text("solo", encoding="utf-8")

            self.assertFalse(normalize_mineru_zip_layout(root, "foo"))

            self.assertEqual(md.read_text(encoding="utf-8"), "solo")

    def test_merge_images_when_root_images_exists(self) -> None:
        """out_root 已有 images/ 时与内层 images 合并。"""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "doc.md").write_text("x", encoding="utf-8")
            existing = root / "images"
            existing.mkdir()
            (existing / "old.png").write_bytes(b"old")

            inner = root / "doc"
            inner.mkdir()
            (inner / "doc.md").write_text("inner", encoding="utf-8")
            (inner / "images").mkdir()
            (inner / "images" / "new.png").write_bytes(b"new")

            self.assertTrue(normalize_mineru_zip_layout(root, "doc"))

            self.assertFalse((root / "doc").exists())
            self.assertEqual((root / "doc.md").read_text(encoding="utf-8"), "inner")
            self.assertTrue((root / "images" / "old.png").is_file())
            self.assertTrue((root / "images" / "new.png").is_file())


class TestPersistZipArtifactsWithLayout(unittest.TestCase):
    def test_zip_triggers_normalize(self) -> None:
        """ZIP 含外层 md + 内层目录时，persist 后主 md 在 out_root 且 images 扁平。"""
        with tempfile.TemporaryDirectory() as td:
            out_root = Path(td)
            output_path = out_root / "report.md"
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as zf:
                zf.writestr("report.md", "outer")
                zf.writestr("report/report.md", "# Inner\n![](images/p.png)\n")
                zf.writestr("report/images/p.png", b"PNG")
            persist_zip_artifacts(
                zip_bytes=buf.getvalue(),
                output_path=output_path,
                prefer_stem="report",
            )
            self.assertEqual(output_path.read_text(encoding="utf-8"), "# Inner\n![](images/p.png)\n")
            self.assertTrue((out_root / "images" / "p.png").is_file())
            self.assertFalse((out_root / "report").exists())

    def test_zip_keeps_only_main_md_and_images(self) -> None:
        """persist 后仅保留主 md 与 images 目录，清理中间过程文件。"""
        with tempfile.TemporaryDirectory() as td:
            out_root = Path(td)
            output_path = out_root / "report.md"
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as zf:
                zf.writestr("report.md", "outer")
                zf.writestr("report/report.md", "# Inner\n![](images/p.png)\n")
                zf.writestr("report/images/p.png", b"PNG")
                zf.writestr("report/report_model.json", "{}")
                zf.writestr("report/report_middle.json", "{}")
                zf.writestr("report/report_origin.pdf", b"%PDF")
            persist_zip_artifacts(
                zip_bytes=buf.getvalue(),
                output_path=output_path,
                prefer_stem="report",
            )

            entries = sorted(p.name for p in out_root.iterdir())
            self.assertEqual(entries, ["images", "report.md"])
            self.assertTrue((out_root / "images" / "p.png").is_file())


if __name__ == "__main__":
    unittest.main()
