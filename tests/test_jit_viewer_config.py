from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pdf_preview_uses_native_renderer_to_avoid_canvas_reuse_race():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")

    assert 'pdfRender: "native"' in html


def test_preview_dialog_is_sized_for_reading():
    css = (ROOT / "static" / "styles.css").read_text(encoding="utf-8")

    assert "width: min(1500px, 100%);" in css
    assert "height: min(94vh, 1000px);" in css

def test_preview_modal_has_source_and_markdown_panes():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")

    assert 'id="sourceViewerContainer"' in html
    assert 'id="outputViewerContainer"' in html
    assert 'data-source-preview-url' in html
    assert 'data-output-preview-url' in html


def test_preview_uses_two_viewers_and_destroys_both_on_close():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")

    assert "sourceViewer = window.JitViewer.createViewer" in html
    assert "outputViewer = window.JitViewer.createViewer" in html
    assert "sourceViewer.destroy" in html
    assert "outputViewer.destroy" in html
