"""
方案 A：将 PDF 按页渲染为图片，逐页调用 Qwen-VL 转写为 Markdown（验证用）。

依赖：pymupdf（import fitz）
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import re
import time
from pathlib import Path
from typing import Optional

from .dashscope_client import DashScopeClient, build_system_message, build_vl_user_message
from .llm_markdown_refiner import _normalize_markdown_output
from .llm_prompts import build_table_cleanup_messages
from .vl_markdown_utils import (
    extract_markdown_table_blocks,
    table_column_count,
    validate_table_output_invariants,
)

_log = logging.getLogger(__name__)

_VL_SYSTEM = (
    "你是文档 OCR 与排版助手。用户将提供 PDF 某一页的渲染图。"
    "请将该页内容转写为 GitHub Flavored Markdown。\n"
    "要求：\n"
    "1) 保留合理的标题层级（#、##、###）。\n"
    "2) 表格使用 Markdown 管道表格（| 列 |）。\n"
    "3) 只输出本页可见内容，不要编造页码或页眉页脚以外的说明。\n"
    "4) 无法辨认的字用 [?] 占位。\n"
    "5) 不要在整个文档外套一层 ```markdown 代码块；可直接输出 Markdown 正文。\n"
    "6) 表格严格要求：每个数据行列数必须与表头一致；不确定的单元格留空，不要串列或跨列。\n"
    "7) 单元格内若包含竖线字符，请写成 \\|，不要新增列。\n"
    "8) 不要重复自我校验或思考过程，只输出最终 Markdown。\n"
    "9) 模型应答时：可交付正文必须出现在 content；禁止把整页转写只放在推理/思考字段。"
)

_VL_USER_TMPL = (
    "这是 PDF「{name}」的第 {idx} / {total} 页渲染图。"
    "请只转录本页内容。"
)


def _split_gfm_row(line: str) -> list[str]:
    t = line.strip()
    if t.startswith("|"):
        t = t[1:]
    if t.endswith("|"):
        t = t[:-1]
    cells: list[str] = []
    cur: list[str] = []
    escaped = False
    for ch in t:
        if escaped:
            cur.append(ch)
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            cur.append(ch)
            continue
        if ch == "|":
            cells.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    cells.append("".join(cur).strip())
    return cells


def _is_gfm_sep_row(line: str) -> bool:
    cells = _split_gfm_row(line)
    if not cells:
        return False
    for c in cells:
        # --- / :--- / ---: / :---:
        if not re.match(r"^:?-{3,}:?$", c):
            return False
    return True


def _fmt_gfm_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _normalize_gfm_tables(markdown: str) -> str:
    """
    修复模型输出中常见的 GFM 表格列错位：
    - 每个数据行列数对齐到表头列数
    - 少列补空，多列并入最后一列（避免后续列整体右移）
    """
    lines = markdown.splitlines()
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        if (
            i + 1 < n
            and "|" in lines[i]
            and "|" in lines[i + 1]
            and _is_gfm_sep_row(lines[i + 1])
        ):
            header_cells = _split_gfm_row(lines[i])
            ncols = len(header_cells)
            if ncols <= 1:
                out.append(lines[i])
                i += 1
                continue

            # 标准化表头 + 分隔行
            out.append(_fmt_gfm_row(header_cells))
            out.append(_fmt_gfm_row(["---"] * ncols))
            i += 2

            while i < n and "|" in lines[i] and lines[i].strip():
                row_cells = _split_gfm_row(lines[i])
                if len(row_cells) < ncols:
                    row_cells = row_cells + [""] * (ncols - len(row_cells))
                elif len(row_cells) > ncols:
                    row_cells = row_cells[: ncols - 1] + [" | ".join(row_cells[ncols - 1 :])]
                out.append(_fmt_gfm_row(row_cells))
                i += 1
            continue

        out.append(lines[i])
        i += 1
    return "\n".join(out)


def _extract_first_table_block(output_text: str) -> str:
    if not output_text:
        return ""
    lines = output_text.splitlines()
    in_table = False
    buf: list[str] = []
    for l in lines:
        t = l.strip()
        if t.startswith("|") and t.count("|") >= 2:
            if not in_table:
                in_table = True
                buf = []
            buf.append(l)
        else:
            if in_table and buf:
                break
    return "\n".join(buf).strip()


def _is_suspicious_table(tb_md: str) -> bool:
    lines = [l for l in tb_md.splitlines() if l.strip().startswith("|")]
    if len(lines) < 2:
        return False
    ncols = table_column_count(tb_md)
    if ncols <= 1:
        return False
    for row in lines:
        cols = len(_split_gfm_row(row))
        if cols != ncols:
            return True
    # 出现大量空单元格也视为可疑（常见错列后被补空）
    empty_cells = 0
    total_cells = 0
    for row in lines[2:]:
        cells = _split_gfm_row(row)
        total_cells += len(cells)
        empty_cells += sum(1 for c in cells if not c.strip())
    if total_cells > 0 and (empty_cells / total_cells) > 0.4:
        return True
    return False


def _review_suspicious_tables_with_llm(
    *,
    client: DashScopeClient,
    model: str,
    page_md: str,
    temperature: Optional[float],
    max_tokens: Optional[int],
    max_tables: int,
) -> str:
    lines = page_md.splitlines()
    blocks = extract_markdown_table_blocks(page_md)
    if not blocks:
        return page_md

    reviewed = 0
    for tb in blocks:
        if reviewed >= max_tables:
            break
        if not _is_suspicious_table(tb.markdown):
            continue
        context_start = max(0, tb.start_line - 2)
        context_text = "\n".join(lines[context_start : tb.start_line]).strip()
        messages = build_table_cleanup_messages(
            original_table_markdown=tb.markdown,
            context_text=context_text,
        )
        raw = client.generate_multimodal(
            model,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        fixed = _normalize_markdown_output(raw)
        fixed_table = _extract_first_table_block(fixed)
        ok, _details = validate_table_output_invariants(
            original_table=tb.markdown,
            refined_table=fixed_table,
        )
        if not ok:
            continue
        new_lines = fixed_table.splitlines()
        expected_len = tb.end_line - tb.start_line + 1
        if len(new_lines) != expected_len:
            continue
        lines[tb.start_line : tb.end_line + 1] = new_lines
        reviewed += 1
    if reviewed > 0:
        _log.info("pdf-vl: table second-pass reviewed=%s", reviewed)
    return "\n".join(lines)


def transcribe_pdf_with_vl(
    *,
    client: DashScopeClient,
    model: str,
    pdf_path: Path,
    markdown_out_path: Optional[Path] = None,
    dpi: float = 150.0,
    workers: int = 1,
    embed_page_images: bool = False,
    table_second_pass: bool = True,
    table_second_pass_max_tables: int = 3,
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

        page_pngs: list[Path] = []
        image_ref_paths: list[str] = []
        assets_dir: Optional[Path] = None
        if markdown_out_path is not None and embed_page_images:
            mdp = markdown_out_path.resolve()
            assets_dir = mdp.parent / f"{mdp.stem}_pages"
            assets_dir.mkdir(parents=True, exist_ok=True)

        for page_index in range(limit):
            page = doc[page_index]
            pix = page.get_pixmap(dpi=int(round(dpi_f)))
            if assets_dir is not None:
                png_path = assets_dir / f"page_{page_index + 1:04d}.png"
            else:
                # 不嵌入输出时仅用于当前进程调用，放系统临时目录
                import tempfile

                tmp_dir = Path(tempfile.gettempdir()) / "docling_pdf_vl_pages"
                tmp_dir.mkdir(parents=True, exist_ok=True)
                png_path = tmp_dir / f"{pdf_path.stem}.page_{page_index + 1:04d}.png"
            pix.save(str(png_path))
            page_pngs.append(png_path)
            if markdown_out_path is not None and assets_dir is not None:
                rel = png_path.relative_to(markdown_out_path.parent.resolve())
                image_ref_paths.append(rel.as_posix())
            elif assets_dir is not None:
                image_ref_paths.append(png_path.as_posix())

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
            else:
                page_md = _normalize_gfm_tables(page_md)
                if table_second_pass:
                    page_md = _review_suspicious_tables_with_llm(
                        client=client,
                        model=model,
                        page_md=page_md,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        max_tables=max(1, int(table_second_pass_max_tables)),
                    )
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
            img_ref = image_ref_paths[i] if (embed_page_images and i < len(image_ref_paths)) else ""
            if embed_page_images and img_ref:
                parts.append(f"![第{i + 1}页渲染图]({img_ref})\n\n")
            parts.append(page_md_map.get(i, "（本页模型输出缺失）"))
            parts.append("\n\n---\n\n")
    finally:
        doc.close()

    return "".join(parts).rstrip() + "\n"
