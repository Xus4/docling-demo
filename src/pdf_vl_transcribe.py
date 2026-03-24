"""
方案 A：将 PDF 按页渲染为图片，逐页调用 Qwen-VL 转写为 Markdown（验证用）。

依赖：pymupdf（import fitz）
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time
import tempfile
from pathlib import Path
from typing import Optional

from .dashscope_client import DashScopeClient, build_system_message, build_vl_user_message
from .llm_markdown_refiner import _normalize_markdown_output

_log = logging.getLogger(__name__)

_VL_SYSTEM = (
    "你是文档 OCR 与排版助手。用户将提供 PDF 某一页的渲染图。"
    "请将该页内容转写为 GitHub Flavored Markdown。\n"
    "要求：\n"
    "1) 保留合理的标题层级（#、##、###）。\n"
    "2) 表格使用 Markdown 管道表格（| 列 |）。\n"
    "3) 只输出本页可见内容，不要编造页码或页眉页脚以外的说明。\n"
    "4) 无法辨认的字用 [?] 占位。\n"
    "5) 不要在整个文档外套一层 ```markdown 代码块；可直接输出 Markdown 正文。"
)

_VL_USER_TMPL = (
    "这是 PDF「{name}」的第 {idx} / {total} 页渲染图。"
    "请只转录本页内容。"
)


def transcribe_pdf_with_vl(
    *,
    client: DashScopeClient,
    model: str,
    pdf_path: Path,
    dpi: float = 150.0,
    workers: int = 1,
    max_pages: Optional[int] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """
    逐页渲染 PDF → PNG（临时文件），每页一次 multimodal 调用，拼接为单个 Markdown。
    """
    try:
        import fitz  # pymupdf
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "pdf-vl-primary 需要安装 pymupdf：pip install pymupdf"
        ) from e

    pdf_path = pdf_path.resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(str(pdf_path))

    dpi_f = float(dpi)
    if dpi_f < 72 or dpi_f > 400:
        _log.warning("pdf-vl dpi=%s 超出常见范围 72~400，仍继续", dpi_f)

    workers_i = max(1, int(workers))
    doc = fitz.open(str(pdf_path))
    try:
        n = doc.page_count
        if n <= 0:
            return f"# {pdf_path.name}\n\n（空 PDF）\n"
        limit = n if max_pages is None else min(n, max(1, int(max_pages)))
        parts: list[str] = [f"# {pdf_path.name}\n\n"]

        with tempfile.TemporaryDirectory(prefix="docling_pdf_vl_") as tmp:
            tmp_path = Path(tmp)
            page_pngs: list[Path] = []
            for page_index in range(limit):
                page = doc[page_index]
                pix = page.get_pixmap(dpi=int(round(dpi_f)))
                png_path = tmp_path / f"page_{page_index:05d}.png"
                pix.save(str(png_path))
                page_pngs.append(png_path)

            def _one_page(i: int, image_path: Path) -> tuple[int, str]:
                user_text = _VL_USER_TMPL.format(
                    name=pdf_path.name,
                    idx=i + 1,
                    total=limit,
                )
                messages = [
                    build_system_message(_VL_SYSTEM),
                    build_vl_user_message(text=user_text, image_paths=[str(image_path)]),
                ]
                _log.info("pdf-vl: page %s/%s dpi=%s", i + 1, limit, dpi_f)
                page_md = ""
                raw_last = ""
                for attempt in range(1, 4):
                    raw_last = client.generate_multimodal(
                        model,
                        messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    page_md = _normalize_markdown_output(raw_last).strip()
                    if page_md:
                        break
                    _log.warning(
                        "pdf-vl: page %s/%s 模型输出为空，重试 %s/2",
                        i + 1,
                        limit,
                        attempt,
                    )
                    if attempt < 3:
                        time.sleep(1.5 * attempt)
                if not page_md:
                    snippet = (raw_last or "")[:400].replace("\n", "\\n")
                    _log.error(
                        "pdf-vl: page %s/%s 仍为空；原始片段(前400字符): %s",
                        i + 1,
                        limit,
                        snippet,
                    )
                    page_md = "（本页模型输出为空）"
                return i, page_md

            if workers_i == 1:
                page_md_map: dict[int, str] = {}
                for i, p in enumerate(page_pngs):
                    idx, md = _one_page(i, p)
                    page_md_map[idx] = md
            else:
                _log.info("pdf-vl: concurrent workers=%s pages=%s", workers_i, limit)
                page_md_map = {}
                with ThreadPoolExecutor(max_workers=workers_i) as ex:
                    futs = [ex.submit(_one_page, i, p) for i, p in enumerate(page_pngs)]
                    for fut in as_completed(futs):
                        idx, md = fut.result()
                        page_md_map[idx] = md

            for i in range(limit):
                parts.append(f"## 第 {i + 1} / {limit} 页\n\n")
                parts.append(page_md_map.get(i, "（本页模型输出缺失）"))
                parts.append("\n\n---\n\n")
    finally:
        doc.close()

    return "".join(parts).rstrip() + "\n"
