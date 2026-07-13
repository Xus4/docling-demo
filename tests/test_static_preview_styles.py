from pathlib import Path
import unittest


def _css_rule(css: str, selector: str) -> str:
    marker = selector + " {"
    start = css.index(marker) + len(marker)
    end = css.index("}", start)
    return css[start:end]


class TestStaticPreviewStyles(unittest.TestCase):
    def test_preview_viewer_defines_scrollable_height_boundary(self) -> None:
        css = Path("static/styles.css").read_text(encoding="utf-8")
        rule = _css_rule(css, ".preview-viewer")

        self.assertIn("height: 100%", rule)
        self.assertIn("overflow: auto", rule)

    def test_preview_panes_are_two_columns_on_wide_screens(self) -> None:
        css = Path("static/styles.css").read_text(encoding="utf-8")

        self.assertIn(".preview-panes", css)
        self.assertIn("grid-template-columns: repeat(2, minmax(0, 1fr))", css)


if __name__ == "__main__":
    unittest.main()
