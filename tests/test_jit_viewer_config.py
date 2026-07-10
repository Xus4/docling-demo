from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pdf_preview_uses_native_renderer_to_avoid_canvas_reuse_race():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")

    assert 'pdfRender: "native"' in html


def test_preview_dialog_is_sized_for_reading():
    css = (ROOT / "static" / "styles.css").read_text(encoding="utf-8")

    assert "width: min(1500px, 100%);" in css
    assert "height: min(94vh, 1000px);" in css