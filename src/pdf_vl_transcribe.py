
"""
方案 A：将 PDF 按页渲染为图片，逐页调用 Qwen-VL 转写为 Markdown（验证用）。

依赖：pymupdf（import fitz）
"""
from __future__ import annotations
import difflib
import math
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import re
import tempfile
import time
from dataclasses import dataclass
from PIL import Image
from pathlib import Path
from typing import Callable, Optional

from .dashscope_client import DashScopeClient, build_system_message, build_vl_user_message
from .llm_markdown_refiner import _normalize_markdown_output
from .llm_prompts import INTERNAL_VL_PAGE_REASONING, build_table_cleanup_messages
from .logging_utils import kv, log_event
from .vl_markdown_utils import (
    extract_markdown_table_blocks,
    table_column_count,
    validate_table_output_invariants,
)

_log = logging.getLogger(__name__)

_VL_SYSTEM = (
    "你是文档 OCR 与排版助手。用户将提供 PDF 某一页的渲染图；请转写为 **GitHub Flavored Markdown**。"
    "优先级（冲突时以前者为准）：①忠实还原可见文字与真实表格 ②工程图/照片等不得伪造为表格 ③表格勾选与列对齐 ④图题与位置标注。\n"
    "要求：\n"
    "1) 保留合理的标题层级（#、##、###）；列表层级与编号与页面一致。\n"
    "2) 表格必须使用 **GFM 管道表格**，并严格可渲染：\n"
    "   - 每一行必须以 `|` 开头并以 `|` 结尾（不要省略两端管道）。\n"
    "   - 必须包含分隔行，例如：`| --- | --- |`（每列至少 3 个 `-`，可含 `:` 对齐）。\n"
    "   - 表头行、分隔行、每个数据行的列数必须完全一致；禁止缺列/多列；不确定单元格可留空但必须保留列位。\n"
    "   - 单元格内若出现竖线字符，必须写成 `\\|`，禁止引入额外列。\n"
    "   - 禁止在表格行前加任何缩进（空格/Tab）；否则很多 Markdown 编辑器会把表格当普通文本/源码。\n"
    "   - 表格前后各留一行空行，避免被列表/引用块粘连导致渲染失败。\n"
    "3) 只输出本页可见内容；不编造页眉页脚说明；无法辨认的字用 `[?]` 占位。\n"
    "4) **绝对不要**用 ``` 或 ```markdown 包裹整页输出；也不要输出任何解释/分析文字。\n"
    "5) 只输出最终 Markdown；可交付正文必须写在 content，勿仅放在推理字段。\n"
    "6) 含勾选/对钩（✓、√、手写勾）的表格：逐格核对，尤其最右列与靠边格，避免漏读。\n"
    "7) 插图/照片/曲线图等：在图题行后标注位置，**仅**使用下列枚举之一（勿自造英文短语）："
    "[[position:左上|右上|左下|右下|页面顶部|页面底部|页面中部|页面左侧|页面右侧]]；"
    "示例：图 1.1 标题文字 [[position:页面顶部]]。\n"
    "8) **严禁**把工程示意图、安装图、剖面图、构造图、实物照片等用 Markdown 表格去「概括」；"
    "此类内容用图题 + [[position:...]] 表示存在即可，勿用表格模拟图形结构。\n"
    "9) 多表同页：每张表独立核对列数 N 与分隔行，禁止把两张表的行混在一起。\n"
    f"{INTERNAL_VL_PAGE_REASONING}"
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
    # 列数较多的核查/登记表：右侧勾选列漏检更常见，走二次校对（与页图对照）
    if ncols >= 8:
        return True
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


def _table_cleanup_messages_with_page_image(
    *,
    original_table_markdown: str,
    context_text: str,
    page_image_path: Path,
) -> list[dict]:
    """
    与 Docling 流程中 cleanup_tables_per_block 一致：表格纠错请求附带本页渲染图，
    便于对照最右列/靠边格的勾选符号（二次校对此前未传图，无法补漏）。
    """
    messages = build_table_cleanup_messages(
        original_table_markdown=original_table_markdown,
        context_text=context_text,
    )
    user_text = ""
    for m in messages:
        if m.get("role") == "user":
            user_text = str(m.get("content") or "")
            break
    user_message = {
        "role": "user",
        "content": [{"image": str(page_image_path)}] + [{"text": user_text}],
    }
    return [m for m in messages if m.get("role") != "user"] + [user_message]


def _review_suspicious_tables_with_llm(
    *,
    client: DashScopeClient,
    model: str,
    page_md: str,
    page_image_path: Path,
    temperature: Optional[float],
    max_tokens: Optional[int],
    max_tables: int,
) -> tuple[str, int]:
    lines = page_md.splitlines()
    blocks = extract_markdown_table_blocks(page_md)
    if not blocks:
        return page_md, 0

    reviewed = 0
    for tb in blocks:
        if max_tables > 0 and reviewed >= max_tables:
            break
        if not _is_suspicious_table(tb.markdown):
            continue
        context_start = max(0, tb.start_line - 2)
        context_text = "\n".join(lines[context_start : tb.start_line]).strip()
        messages = _table_cleanup_messages_with_page_image(
            original_table_markdown=tb.markdown,
            context_text=context_text,
            page_image_path=page_image_path,
        )
        raw = client.generate_multimodal(
            model,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            biz_stage="表格纠错",
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
        log_event(_log, logging.INFO, "pdf_vl.table_review.patched", patched=reviewed)
    return "\n".join(lines), reviewed




_CAPTION_RE = re.compile(
    r"^\s*(图\s*[A-Za-z0-9一二三四五六七八九十零\.\-—_（）()]+.*)$"
)

_POSITION_RE = re.compile(r"(?:\[\[?|【|\[\s*\[)\s*position:\s*([^\]】]+?)\s*(?:\]\]?|】|\]\s*\])")


@dataclass(frozen=True)
class CaptionLine:
    line_index: int
    text: str
    position: Optional[str] = None


@dataclass(frozen=True)
class DetectedFigure:
    bbox: tuple[float, float, float, float]
    caption_line: Optional[CaptionLine] = None


def _extract_figure_caption_lines(page_md: str, max_per_page: int = 4) -> list[CaptionLine]:
    out: list[CaptionLine] = []
    lines = page_md.splitlines()
    for i, line in enumerate(lines):
        t = line.strip()
        if not t:
            continue
        if _CAPTION_RE.match(t):
            pos_match = _POSITION_RE.search(t)
            position = pos_match.group(1).strip() if pos_match else None
            clean_text = _POSITION_RE.sub("", t).strip()
            out.append(CaptionLine(line_index=i, text=clean_text, position=position))
            if len(out) >= max_per_page:
                break
    return out


def _open_page_image_size(page_image_path: Path) -> tuple[int, int]:
    with Image.open(page_image_path) as im:
        return im.size


def _crop_single_region_above_caption(
    *,
    page_image_path: Path,
    out_path: Path,
    mode: str,
) -> bool:
    """
    mode:
      - single_full: 单图，取页面中上部大区域
      - left_half: 左半图
      - right_half: 右半图
      - top_half: 上半图
      - bottom_half: 下半图
      - top_left: 左上角
      - top_right: 右上角
      - bottom_left: 左下角
      - bottom_right: 右下角
      - bottom_center: 下部中间
    """
    with Image.open(page_image_path) as im:
        w, h = im.size

        if mode == "single_full":
            top = int(h * 0.10)
            bottom = int(h * 0.75)
            left = int(w * 0.08)
            right = int(w * 0.92)
        elif mode == "left_half":
            top = int(h * 0.10)
            bottom = int(h * 0.90)
            left = int(w * 0.06)
            right = int(w * 0.49)
        elif mode == "right_half":
            top = int(h * 0.10)
            bottom = int(h * 0.90)
            left = int(w * 0.51)
            right = int(w * 0.94)
        elif mode == "top_half":
            top = int(h * 0.08)
            bottom = int(h * 0.55)
            left = int(w * 0.08)
            right = int(w * 0.92)
        elif mode == "bottom_half":
            top = int(h * 0.45)
            bottom = int(h * 0.92)
            left = int(w * 0.08)
            right = int(w * 0.92)
        elif mode == "top_left":
            top = int(h * 0.08)
            bottom = int(h * 0.52)
            left = int(w * 0.06)
            right = int(w * 0.50)
        elif mode == "top_right":
            top = int(h * 0.08)
            bottom = int(h * 0.52)
            left = int(w * 0.50)
            right = int(w * 0.94)
        elif mode == "bottom_left":
            top = int(h * 0.48)
            bottom = int(h * 0.92)
            left = int(w * 0.06)
            right = int(w * 0.50)
        elif mode == "bottom_right":
            top = int(h * 0.48)
            bottom = int(h * 0.92)
            left = int(w * 0.50)
            right = int(w * 0.94)
        elif mode == "bottom_center":
            top = int(h * 0.48)
            bottom = int(h * 0.92)
            left = int(w * 0.25)
            right = int(w * 0.75)
        else:
            return False

        if right - left < 40 or bottom - top < 40:
            return False

        crop = im.crop((left, top, right, bottom))
        crop.save(out_path)
        return True


_POSITION_TO_MODE = {
    "左上": "top_left",
    "右上": "top_right",
    "左下": "bottom_left",
    "右下": "bottom_right",
    "页面顶部": "top_half",
    "页面底部": "bottom_half",
    "页面中部": "single_full",
    "页面左侧": "left_half",
    "页面右侧": "right_half",
    "底部中间": "bottom_center",
    "下中部": "bottom_center",
    "top-left": "top_left",
    "top-right": "top_right",
    "bottom-left": "bottom_left",
    "bottom-right": "bottom_right",
    "top": "top_half",
    "bottom": "bottom_half",
    "left": "left_half",
    "right": "right_half",
    "center": "single_full",
}
def _bbox_area(b: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = b
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _bbox_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = _bbox_area(a) + _bbox_area(b) - inter
    if ua <= 0:
        return 0.0
    return inter / ua


def _bbox_close(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    gap: float = 12.0,
) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (
        ax1 < bx0 - gap or
        bx1 < ax0 - gap or
        ay1 < by0 - gap or
        by1 < ay0 - gap
    )


def _merge_two_bboxes(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    return (
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[2], b[2]),
        max(a[3], b[3]),
    )


def _merge_overlapping_bboxes(
    bboxes: list[tuple[float, float, float, float]],
    iou_threshold: float = 0.35,
    gap: float = 16.0,
) -> list[tuple[float, float, float, float]]:
    if not bboxes:
        return []

    boxes = list(bboxes)
    changed = True
    while changed:
        changed = False
        new_boxes: list[tuple[float, float, float, float]] = []
        used = [False] * len(boxes)

        for i in range(len(boxes)):
            if used[i]:
                continue
            cur = boxes[i]
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                if _bbox_iou(cur, boxes[j]) >= iou_threshold or _bbox_close(cur, boxes[j], gap=gap):
                    cur = _merge_two_bboxes(cur, boxes[j])
                    used[j] = True
                    changed = True
            used[i] = True
            new_boxes.append(cur)
        boxes = new_boxes

    return boxes


def _normalize_and_filter_bboxes(
    bboxes: list[tuple[float, float, float, float]],
    page_width: float,
    page_height: float,
) -> list[tuple[float, float, float, float]]:
    out: list[tuple[float, float, float, float]] = []
    page_area = max(1.0, page_width * page_height)

    for x0, y0, x1, y1 in bboxes:
        if x1 <= x0 or y1 <= y0:
            continue

        w = x1 - x0
        h = y1 - y0
        area = w * h
        ratio = area / page_area

        # 太小的 logo / 图标 / 噪声
        if ratio < 0.006:
            continue

        # 过细长的线条/边框噪声
        aspect = max(w / max(h, 1e-6), h / max(w, 1e-6))
        if aspect > 18:
            continue

        out.append((float(x0), float(y0), float(x1), float(y1)))

    out = _merge_overlapping_bboxes(out, iou_threshold=0.35, gap=max(page_width, page_height) * 0.01)
    out = sorted(out, key=lambda b: (b[1], b[0]))
    return out


def _expand_bbox_px(
    bbox: tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    pad_ratio: float = 0.04,
    min_pad: int = 12,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    px = max(min_pad, int(bw * pad_ratio))
    py = max(min_pad, int(bh * pad_ratio))
    return (
        max(0, x0 - px),
        max(0, y0 - py),
        min(img_w, x1 + px),
        min(img_h, y1 + py),
    )


def _trim_whitespace_pil(
    im: Image.Image,
    bg_threshold: int = 249,
    pad: int = 28,
) -> Image.Image:
    gray = im.convert("L")
    arr = np.array(gray)
    mask = arr < bg_threshold

    if not mask.any():
        return im

    ys, xs = np.where(mask)
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(im.size[0], int(xs.max()) + 1 + pad)
    y1 = min(im.size[1], int(ys.max()) + 1 + pad)

    # 太激进就不 trim
    if (x1 - x0) < im.size[0] * 0.78 or (y1 - y0) < im.size[1] * 0.78:
        return im

    return im.crop((x0, y0, x1, y1))



def detect_candidate_figure_bboxes(
    page_image_path: Path,
    min_area_ratio: float = 0.015,
    max_boxes: int = 12,
) -> list[tuple[int, int, int, int]]:
    """
    在整页扫描图中找多个候选 figure bbox。
    返回像素坐标 [(x0, y0, x1, y1), ...]
    """
    try:
        import cv2  # type: ignore
    except Exception:
        cv2 = None

    with Image.open(page_image_path) as im:
        gray = np.array(im.convert("L"))

    h, w = gray.shape
    page_area = max(1, h * w)

    if cv2 is not None:
        _, th = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)

        # 比之前更保守：不要把上下两张图闭运算粘在一起
        kx = max(3, int(w * 0.006))
        ky = max(3, int(h * 0.004))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes: list[tuple[int, int, int, int]] = []
        for c in contours:
            x, y, ww, hh = cv2.boundingRect(c)
            area = ww * hh
            if area / page_area < min_area_ratio:
                continue
            if ww < 40 or hh < 40:
                continue

            # 过滤纯文字横条
            aspect = max(ww / max(hh, 1), hh / max(ww, 1))
            if aspect > 14:
                continue

            boxes.append((x, y, x + ww, y + hh))

        boxes = _merge_overlapping_bboxes(boxes, iou_threshold=0.18, gap=max(w, h) * 0.006)
        boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
        return boxes[:max_boxes]

    # 无 cv2 兜底
    mask = (gray < 245).astype(np.uint8)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return []
    return [(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)]

def detect_text_bands(
    page_image_path: Path,
    row_ratio_threshold: float = 0.01,
    min_band_height: int = 10,
    merge_gap: int = 8,
) -> list[tuple[int, int]]:
    with Image.open(page_image_path) as im:
        gray = np.array(im.convert("L"))

    h, w = gray.shape
    mask = (gray < 245).astype(np.uint8)
    row_density = mask.sum(axis=1) / max(1, w)

    active = row_density > row_ratio_threshold
    bands: list[tuple[int, int]] = []
    start = None

    for i, v in enumerate(active):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_band_height:
                bands.append((start, i))
            start = None

    if start is not None and h - start >= min_band_height:
        bands.append((start, h))

    if not bands:
        return []

    merged = [bands[0]]
    for y0, y1 in bands[1:]:
        py0, py1 = merged[-1]
        if y0 - py1 <= merge_gap:
            merged[-1] = (py0, y1)
        else:
            merged.append((y0, y1))
    return merged

def match_captions_to_figure_bboxes(
    captions: list[CaptionLine],
    boxes: list[tuple[int, int, int, int]],
    img_w: int,
    img_h: int,
    has_table: bool = True,
) -> list[tuple[CaptionLine, tuple[int, int, int, int]]]:
    if not captions:
        return []

    table_zones = _detect_table_zone(boxes, img_w, img_h, has_table) if boxes else []

    used: set[int] = set()
    pairs: list[tuple[CaptionLine, tuple[int, int, int, int]]] = []
    
    boxes_sorted = sorted(boxes, key=lambda b: (b[1], b[0])) if boxes else []
    n_caps = len(captions)

    # 【关键修复1】动态生成安全边界与期望Y中心点
    def _get_expected_bounds_and_cy(idx: int, pos: str) -> tuple[int, int, float]:
        if "左" in pos or "右" in pos:
            # 左右结构：彻底放开Y轴上下限，避免被腰斩，期望高度在正中
            return 0, img_h, img_h * 0.5
        else:
            # 上下结构：划分Y轴，但大幅增加宽容度(上下各扩 25% 页面高度)，防止大图被切
            upper = int(img_h * idx / n_caps) if idx > 0 else 0
            lower = int(img_h * (idx + 1) / n_caps) if idx < n_caps - 1 else img_h
            return max(0, upper - int(img_h * 0.25)), min(img_h, lower + int(img_h * 0.25)), img_h * (idx + 0.5) / n_caps

    for idx, cap in enumerate(captions):
        best_j = None
        best_score = None
        
        pos = (cap.position or "").strip()
        default_upper, default_lower, expected_cy = _get_expected_bounds_and_cy(idx, pos)

        for j, b in enumerate(boxes_sorted):
            if j in used:
                continue

            x0, y0, x1, y1 = b
            area = (x1 - x0) * (y1 - y0)
            cx = (x0 + x1) / 2
            cy = (y0 + y1) / 2
            bw = x1 - x0
            bh = y1 - y0

            in_table = False
            for tz in table_zones:
                tx0, ty0, tx1, ty1 = tz
                ix0, iy0 = max(x0, tx0), max(y0, ty0)
                ix1, iy1 = min(x1, tx1), min(y1, ty1)
                if ix1 > ix0 and iy1 > iy0:
                    if (ix1 - ix0) * (iy1 - iy0) / max(area, 1) > 0.6:
                        in_table = True
                        break
            if in_table:
                continue

            score = area
            
            # 【关键修复2】强力惩罚偏离期望Y轴高度的框，终结“大图霸权”，解决 2.1和2.2 乱序
            y_dist_ratio = abs(cy - expected_cy) / max(img_h, 1)
            score -= area * y_dist_ratio * 4.0  
            
            # 惩罚拓扑错位
            score -= abs(idx - j) * area * 0.1

            if max(bw / max(bh, 1), bh / max(bw, 1)) > 2.8:
                score -= area * 0.2

            if pos:
                if "上" in pos or "顶部" in pos:
                    if cy < img_h * 0.45: score += area * 1.5
                    elif cy > img_h * 0.6: score -= area * 3.0
                elif "下" in pos or "底部" in pos:
                    if cy > img_h * 0.55: score += area * 1.5
                    elif cy < img_h * 0.4: score -= area * 3.0
                elif "中" in pos:
                    if img_h * 0.25 < cy < img_h * 0.75: score += area * 0.8
                    else: score -= area * 1.0
                
                # 增强左右判定
                if "左" in pos and "右" not in pos:
                    if cx > img_w * 0.55: score -= area * 3.0
                    else: score += area * 1.0
                elif "右" in pos and "左" not in pos:
                    if cx < img_w * 0.45: score -= area * 3.0
                    else: score += area * 1.0

            if best_score is None or score > best_score:
                best_score = score
                best_j = j

        if best_j is not None:
            seed = boxes_sorted[best_j]
            y_upper, y_lower = default_upper, default_lower
            x_left, x_right = 0, img_w

            if "上" in pos or "顶部" in pos: y_lower = min(y_lower, int(img_h * 0.6))
            elif "中" in pos:
                y_upper = max(y_upper, int(img_h * 0.15))
                y_lower = min(y_lower, int(img_h * 0.85))
            
            # 【关键修复3】严格建立左右结构的“横向隔离墙”，防止切图串门
            if "左" in pos and "右" not in pos:
                x_right = int(img_w * 0.53)
            elif "右" in pos and "左" not in pos:
                x_left = int(img_w * 0.47)

            merged = _group_boxes_for_single_figure(
                seed_box=seed, all_boxes=boxes_sorted, img_w=img_w, img_h=img_h, max_extra_boxes=12,
                y_upper_bound=y_upper, y_lower_bound=y_lower,
                x_left_bound=x_left, x_right_bound=x_right,
                table_zones=table_zones,
            )
            used.add(best_j)

            for k, b in enumerate(boxes_sorted):
                if k not in used:
                    bx0, by0, bx1, by1 = b
                    mx0, my0, mx1, my1 = merged
                    ix0, iy0 = max(bx0, mx0), max(by0, my0)
                    ix1, iy1 = min(bx1, mx1), min(by1, my1)
                    if ix1 > ix0 and iy1 > iy0:
                        if (ix1 - ix0) * (iy1 - iy0) / max((bx1 - bx0) * (by1 - by0), 1) > 0.6:
                            used.add(k)

            pairs.append((cap, merged))

    # 兜底逻辑
    matched_caps = {p[0] for p in pairs}
    for idx, cap in enumerate(captions):
        if cap not in matched_caps:
            pos = (cap.position or "").strip()
            margin_x = int(img_w * 0.08)
            _, _, expected_cy = _get_expected_bounds_and_cy(idx, pos)
            
            y0 = max(0, int(expected_cy - img_h * 0.15))
            y1 = min(img_h, int(expected_cy + img_h * 0.15))
            
            if "上" in pos or "顶部" in pos: y0, y1 = int(img_h * 0.05), int(img_h * 0.45)
            elif "下" in pos or "底部" in pos: y0, y1 = int(img_h * 0.55), int(img_h * 0.95)
            
            synthetic_box = (margin_x, y0, img_w - margin_x, min(y1, img_h))
            pairs.append((cap, synthetic_box))

    return pairs





def _norm_caption_text(s: str) -> str:
    s = _clean_position_markers(s or "")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[：:;；,，.。．、]+$", "", s)
    return s.strip()


def _caption_match_score(a: str, b: str) -> float:
    na = _norm_caption_text(a)
    nb = _norm_caption_text(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    if na in nb or nb in na:
        return 0.95
    return difflib.SequenceMatcher(None, na, nb).ratio()


def _expand_bbox_tuple(
    bbox: tuple[int, int, int, int],
    pad_x: int,
    pad_y: int,
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    return (
        max(0, x0 - pad_x),
        max(0, y0 - pad_y),
        min(img_w, x1 + pad_x),
        min(img_h, y1 + pad_y),
    )


def _bbox_union(
    boxes: list[tuple[int, int, int, int]]
) -> tuple[int, int, int, int]:
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )
def _bbox_iou(box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]) -> float:
    """计算两个边界框的交并比 (Intersection over Union)"""
    x0_1, y0_1, x1_1, y1_1 = box1
    x0_2, y0_2, x1_2, y1_2 = box2
    
    ix0 = max(x0_1, x0_2)
    iy0 = max(y0_1, y0_2)
    ix1 = min(x1_1, x1_2)
    iy1 = min(y1_1, y1_2)
    
    iw = max(0, ix1 - ix0)
    ih = max(0, iy1 - iy0)
    inter = iw * ih
    
    area1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    area2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0.0


def _bbox_center(b: tuple[int, int, int, int]) -> tuple[float, float]:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _group_boxes_for_single_figure(
    seed_box: tuple[int, int, int, int],
    all_boxes: list[tuple[int, int, int, int]],
    img_w: int,
    img_h: int,
    max_extra_boxes: int = 8,
    y_upper_bound: int = 0,
    y_lower_bound: int | None = None,
    x_left_bound: int = 0,
    x_right_bound: int | None = None,
    table_zones: list[tuple[int, int, int, int]] | None = None,
) -> tuple[int, int, int, int]:
    group = [seed_box]
    sx0, sy0, sx1, sy1 = seed_box
    scx, scy = _bbox_center(seed_box)
    sw = sx1 - sx0
    sh = sy1 - sy0

    if y_lower_bound is None:
        y_lower_bound = img_h
    if x_right_bound is None:
        x_right_bound = img_w

    # 【关键修复4】增加初始外扩范围，容忍更松散的构图（如俯视图和剖面图间隙）
    cur = _expand_bbox_tuple(
        seed_box,
        pad_x=max(int(img_w * 0.06), int(sw * 0.6)),
        pad_y=max(int(img_h * 0.08), int(sh * 0.3)),
        img_w=img_w,
        img_h=img_h,
    )
    cur = (
        max(cur[0], x_left_bound), 
        max(cur[1], y_upper_bound), 
        min(cur[2], x_right_bound), 
        min(cur[3], y_lower_bound)
    )

    used = {seed_box}

    for _ in range(max_extra_boxes):
        added = False
        cx0, cy0, cx1, cy1 = cur
        for b in all_boxes:
            if b in used:
                continue
            x0, y0, x1, y1 = b
            bw = x1 - x0
            bh = y1 - y0
            bcx, bcy = _bbox_center(b)

            if bcx < x_left_bound or bcx > x_right_bound or bcy < y_upper_bound or bcy > y_lower_bound:
                continue

            if table_zones:
                in_table = False
                for tz in table_zones:
                    tx0, ty0, tx1, ty1 = tz
                    ix0, iy0 = max(x0, tx0), max(y0, ty0)
                    ix1, iy1 = min(x1, tx1), min(y1, ty1)
                    if ix1 > ix0 and iy1 > iy0:
                        if (ix1 - ix0) * (iy1 - iy0) / max(bw * bh, 1) > 0.5:
                            in_table = True
                            break
                if in_table:
                    continue

            in_zone = not (x1 < cx0 or x0 > cx1 or y1 < cy0 or y0 > cy1)
            
            # 【关键修复5】增加垂直方向合并容忍度，跨越空隙将上下两部分抓取到同一个图里
            close_enough = (
                abs(bcx - scx) < max(sw * 1.2, img_w * 0.35)
                and abs(bcy - scy) < max(sh * 1.0, img_h * 0.25)
            )

            if not in_zone and not close_enough:
                continue

            aspect = max(bw / max(bh, 1), bh / max(bw, 1))
            if aspect > 15:
                continue

            group.append(b)
            used.add(b)
            added = True

        if not added:
            break

        union_box = _bbox_union(group)
        ux0, uy0, ux1, uy1 = union_box
        cur = _expand_bbox_tuple(union_box, pad_x=max(30, int((ux1-ux0) * 0.2)), pad_y=max(30, int((uy1-uy0) * 0.15)), img_w=img_w, img_h=img_h)
        cur = (max(cur[0], x_left_bound), max(cur[1], y_upper_bound), min(cur[2], x_right_bound), min(cur[3], y_lower_bound))

    return _bbox_union(group)



def detect_visual_regions_from_page_image(
    page_image_path: Path,
    min_area_ratio: float = 0.015,
    max_regions: int = 8,
) -> list[tuple[int, int, int, int]]:
    """
    在扫描页 PNG 上直接检测主要图形/插图区域。
    返回像素坐标 bbox: [(x0, y0, x1, y1), ...]
    优先使用 cv2；若不可用则退回 numpy 投影法。
    """
    try:
        import cv2  # type: ignore

        img = cv2.imread(str(page_image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return []

        h, w = img.shape[:2]
        page_area = max(1, h * w)

        # 非白区域
        _, th = cv2.threshold(img, 245, 255, cv2.THRESH_BINARY_INV)

        # 连接图线/文字块，得到大内容块
        k1 = max(3, int(min(w, h) * 0.008))
        k2 = max(3, int(min(w, h) * 0.015))
        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (k1, k1))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (k2, k2))
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel1)
        th = cv2.dilate(th, kernel2, iterations=1)

        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions: list[tuple[int, int, int, int]] = []
        for c in contours:
            x, y, ww, hh = cv2.boundingRect(c)
            area = ww * hh
            ratio = area / page_area
            if ratio < min_area_ratio:
                continue
            if ww < 40 or hh < 40:
                continue
            regions.append((x, y, x + ww, y + hh))

        regions = _merge_overlapping_bboxes(regions, iou_threshold=0.20, gap=max(w, h) * 0.01)
        regions = sorted(regions, key=lambda b: (_bbox_area(b) * -1.0, b[1], b[0]))
        return [(int(x0), int(y0), int(x1), int(y1)) for x0, y0, x1, y1 in regions[:max_regions]]

    except Exception:
        pass

    # 无 cv2 的简易兜底
    try:
        with Image.open(page_image_path) as im:
            gray = im.convert("L")
            arr = np.array(gray)
            h, w = arr.shape[:2]
            page_area = max(1, h * w)

            mask = arr < 245
            if not mask.any():
                return []

            row_nonwhite = mask.sum(axis=1)
            col_nonwhite = mask.sum(axis=0)

            row_thresh = max(10, int(w * 0.02))
            col_thresh = max(10, int(h * 0.02))

            row_idx = np.where(row_nonwhite > row_thresh)[0]
            col_idx = np.where(col_nonwhite > col_thresh)[0]
            if len(row_idx) == 0 or len(col_idx) == 0:
                return []

            y0, y1 = int(row_idx.min()), int(row_idx.max()) + 1
            x0, x1 = int(col_idx.min()), int(col_idx.max()) + 1

            area = (x1 - x0) * (y1 - y0)
            if area / page_area < min_area_ratio:
                return []

            return [(x0, y0, x1, y1)]
    except Exception:
        return []

def _detect_table_zone(
    boxes: list[tuple[int, int, int, int]],
    img_w: int,
    img_h: int,
    has_table: bool = True,  # 新增参数，指示页面是否有文本表格
) -> list[tuple[int, int, int, int]]:
    if not has_table or len(boxes) < 6:
        return []

    table_zones = []
    used = set()

    for i, b in enumerate(boxes):
        if i in used:
            continue
        x0, y0, x1, y1 = b
        bw, bh = x1 - x0, y1 - y0
        
        # 【关键修复】单元格不能太小，防止折线图的刻度点、数据点被误判为表格单元格
        if bw < 12 or bh < 12:
            continue
            
        aspect = max(bw / max(bh, 1), bh / max(bw, 1))
        if aspect > 8:
            continue

        peers = [i]
        for j, nb in enumerate(boxes):
            if j == i or j in used:
                continue
            nx0, ny0, nx1, ny1 = nb
            nbh = ny1 - ny0
            # 同一行：y轴高度接近，且纵向对齐
            if abs(nbh - bh) < bh * 0.4 and abs(ny0 - y0) < bh * 0.6 and abs(nx0 - x0) < img_w * 0.8:
                peers.append(j)

        if len(peers) >= 4:
            all_cells = list(peers)
            row_y_set = {boxes[k][1] for k in peers}
            for j, nb in enumerate(boxes):
                if j in all_cells:
                    continue
                ny0 = nb[1]
                nbh = nb[3] - nb[1]
                min_row_y = min(row_y_set)
                max_row_y = max(row_y_set)
                if (ny0 >= min_row_y - bh * 2 and ny0 <= max_row_y + bh * 4 and abs(nbh - bh) < bh * 0.5):
                    all_cells.append(j)
                    row_y_set.add(ny0)

            if len(all_cells) >= 8:
                cell_boxes = [boxes[k] for k in all_cells]
                zone = _bbox_union(cell_boxes)
                table_zones.append(zone)
                for k in all_cells:
                    used.add(k)

    return table_zones


def crop_figures_from_scanned_page_by_visual_regions(
    *,
    page_md: str,
    page_image_path: Path,
    markdown_out_path: Path,
    page_index: int,
    max_per_page: int,
) -> list[tuple[str, str]]:
    captions = _extract_figure_caption_lines(page_md, max_per_page=max_per_page)
    if not captions:
        return []

    figures_dir = markdown_out_path.parent / f"{markdown_out_path.stem}_figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(page_image_path) as im:
        img_w, img_h = im.size

    boxes = detect_candidate_figure_bboxes(
        page_image_path=page_image_path,
        min_area_ratio=0.01,
        max_boxes=max(8, len(captions) * 3),
    )

    if not boxes:
        return crop_figures_by_captions_from_page_image(
            page_md=page_md, page_image_path=page_image_path,
            markdown_out_path=markdown_out_path, page_index=page_index, max_per_page=max_per_page,
        )

    # 【新增】智能判断页面是否存在 Markdown 表格，极大地防止图表被误伤（解决 90 页）
    has_table = ("|" in page_md or "|-" in page_md)

    pairs = match_captions_to_figure_bboxes(
        captions=captions, boxes=boxes, img_w=img_w, img_h=img_h, has_table=has_table
    )

    if not pairs:
        return []

    out: list[tuple[str, str]] = []
    _saved_bboxes: list[tuple[str, tuple[int, int, int, int]]] = []

    for idx, (cap, bbox) in enumerate(pairs, start=1):
        out_path = figures_dir / f"page_{page_index + 1:04d}_fig_{idx:02d}.png"

        # 【关键修复】改回单纯的 IoU 交并比检测，防止 64 页正常兜底框被大框误伤跳过
        overlap_too_much = False
        for _, prev_bbox in _saved_bboxes:
            if _bbox_iou(bbox, prev_bbox) > 0.85:  # 只有两框基本一致时才跳过
                overlap_too_much = True
                break

        if overlap_too_much:
            _log.warning("pdf-vl: page %s fig %s bbox overlaps heavily with previous, skipped", page_index+1, idx)
            continue

        ok = crop_image_by_bbox(
            page_image_path=page_image_path, out_path=out_path, bbox=bbox,
            margin=max(14, int(min(img_w, img_h) * 0.01)),
        )

        if ok:
            rel = out_path.relative_to(markdown_out_path.parent).as_posix()
            out.append((cap.text, rel))
            _saved_bboxes.append((cap.text, bbox))

    return out



def _get_crop_mode_for_position(position: Optional[str]) -> str:
    if not position:
        return "single_full"
    pos = position.strip()
    for key, mode in _POSITION_TO_MODE.items():
        if key in pos or pos in key:
            return mode
    return "single_full"

def _binarize_page_image(page_image_path: Path, threshold: int = 245) -> np.ndarray:
    with Image.open(page_image_path) as im:
        gray = im.convert("L")
        arr = np.array(gray)
    return (arr < threshold).astype(np.uint8)


def _find_text_like_row_bands(
    bin_img: np.ndarray,
    row_ratio_threshold: float = 0.015,
    min_band_height: int = 12,
    merge_gap: int = 10,
) -> list[tuple[int, int]]:
    """
    找页面中横向的“文字/说明带”。
    返回 [(y0, y1), ...]
    """
    h, w = bin_img.shape
    row_density = bin_img.sum(axis=1) / max(1, w)

    active = row_density > row_ratio_threshold
    bands: list[tuple[int, int]] = []

    start = None
    for i, v in enumerate(active):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_band_height:
                bands.append((start, i))
            start = None
    if start is not None and h - start >= min_band_height:
        bands.append((start, h))

    if not bands:
        return []

    merged = [bands[0]]
    for y0, y1 in bands[1:]:
        py0, py1 = merged[-1]
        if y0 - py1 <= merge_gap:
            merged[-1] = (py0, y1)
        else:
            merged.append((y0, y1))
    return merged

def _split_page_into_figure_bands(
    page_image_path: Path,
    caption_count: int,
    top_margin_ratio: float = 0.04,
    bottom_margin_ratio: float = 0.04,
) -> list[tuple[int, int]]:
    """
    根据页面内容投影，把页面切成多个纵向 figure band。
    适合一页上下排多个图的扫描页。
    """
    bin_img = _binarize_page_image(page_image_path)
    h, w = bin_img.shape

    row_density = bin_img.sum(axis=1) / max(1, w)

    top_margin = int(h * top_margin_ratio)
    bottom_margin = int(h * bottom_margin_ratio)

    valid = row_density[top_margin:h - bottom_margin]

    # 用低密度行作为候选分割缝
    low_thresh = max(0.002, np.percentile(valid, 20))
    low_rows = np.where(valid <= low_thresh)[0] + top_margin

    if len(low_rows) == 0 or caption_count <= 1:
        return [(top_margin, h - bottom_margin)]

    # 聚合连续低密度行
    gaps: list[tuple[int, int]] = []
    s = low_rows[0]
    p = low_rows[0]
    for y in low_rows[1:]:
        if y - p <= 6:
            p = y
        else:
            gaps.append((s, p))
            s = y
            p = y
    gaps.append((s, p))

    # 取最像“分隔图与图”的空白带
    gap_centers = [int((a + b) / 2) for a, b in gaps if (b - a) >= 8]

    if not gap_centers:
        return [(top_margin, h - bottom_margin)]

    # 按 caption 数决定要切几刀
    need_splits = caption_count - 1
    if len(gap_centers) > need_splits:
        # 选最分散的缝，避免都挤在一起
        selected = np.linspace(0, len(gap_centers) - 1, need_splits).round().astype(int)
        split_rows = [gap_centers[i] for i in selected]
    else:
        split_rows = gap_centers

    split_rows = sorted(set(split_rows))

    bands: list[tuple[int, int]] = []
    prev = top_margin
    for y in split_rows:
        if y - prev > 40:
            bands.append((prev, y))
            prev = y
    if h - bottom_margin - prev > 40:
        bands.append((prev, h - bottom_margin))

    return bands if bands else [(top_margin, h - bottom_margin)]

def _detect_main_figure_bbox_in_band(
    page_image_path: Path,
    band: tuple[int, int],
    exclude_bottom_text_ratio: float = 0.22,
    min_area_ratio_in_band: float = 0.08,
) -> tuple[int, int, int, int] | None:
    """
    在一个纵向 band 内找主图框。
    会优先排除 band 下方 caption/说明文字区域。
    """
    try:
        import cv2  # type: ignore
    except Exception:
        cv2 = None

    with Image.open(page_image_path) as im:
        gray = np.array(im.convert("L"))
    h, w = gray.shape
    y0, y1 = band
    y0 = max(0, min(y0, h - 1))
    y1 = max(y0 + 1, min(y1, h))

    sub = gray[y0:y1, :]
    sh, sw = sub.shape
    if sh < 30 or sw < 30:
        return None

    # 先切掉 band 底部一部分，尽量排除 caption/构件说明
    cut_bottom = int(sh * (1.0 - exclude_bottom_text_ratio))
    cut_bottom = max(int(sh * 0.55), cut_bottom)
    main = sub[:cut_bottom, :]

    if cv2 is not None:
        _, th = cv2.threshold(main, 245, 255, cv2.THRESH_BINARY_INV)

        kx = max(3, int(sw * 0.01))
        ky = max(3, int(main.shape[0] * 0.01))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        band_area = main.shape[0] * main.shape[1]
        for c in contours:
            x, y, ww, hh = cv2.boundingRect(c)
            area = ww * hh
            if area / max(1, band_area) < min_area_ratio_in_band:
                continue
            if ww < 40 or hh < 40:
                continue
            # 太扁的文字带过滤掉
            aspect = max(ww / max(hh, 1), hh / max(ww, 1))
            if aspect > 12:
                continue
            candidates.append((x, y, x + ww, y + hh))

        if not candidates:
            return None

        # 不选最大面积裸值，选“更像图”的框：更高、更靠上、面积大
        def score(b):
            x0, yy0, x1, yy1 = b
            area = (x1 - x0) * (yy1 - yy0)
            height = yy1 - yy0
            width = x1 - x0
            return area + height * 200 - yy0 * 50 + min(width, sw * 0.9) * 20

        best = max(candidates, key=score)
        bx0, by0, bx1, by1 = best
        return (bx0, y0 + by0, bx1, y0 + by1)

    # 无 cv2 兜底
    mask = (main < 245).astype(np.uint8)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    bx0, bx1 = int(xs.min()), int(xs.max()) + 1
    by0, by1 = int(ys.min()), int(ys.max()) + 1
    return (bx0, y0 + by0, bx1, y0 + by1)

def crop_figures_by_docling_layout(
    *,
    page_md: str,
    page_image_path: Path,
    markdown_out_path: Path,
    page_index: int,
    max_per_page: int,
    pdf_path: Optional[Path] = None,
    dpi: float = 150.0,
) -> list[tuple[str, str]]:
    """
    优先使用 MinerU / PyMuPDF 的图片位置。
    对扫描 PDF：
    - 不再直接按 caption 位置粗裁
    - 而是先做页面视觉区域检测
    """
    captions = _extract_figure_caption_lines(page_md, max_per_page=max_per_page)
    if not captions:
        return []

    figures_dir = markdown_out_path.parent / f"{markdown_out_path.stem}_figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    page_width = 1.0
    page_height = 1.0

    if pdf_path is not None:
        try:
            import fitz
            doc = fitz.open(str(pdf_path))
            try:
                if page_index < doc.page_count:
                    page_width = doc[page_index].rect.width
                    page_height = doc[page_index].rect.height
            finally:
                doc.close()
        except Exception:
            pass

    mineru_bboxes: list[tuple[float, float, float, float]] = []
    pymupdf_bboxes: list[tuple[float, float, float, float]] = []

    if pdf_path is not None:
        mineru_bboxes = get_mineru_image_positions(pdf_path, page_index)
        if not mineru_bboxes:
            pymupdf_bboxes = get_page_image_positions(pdf_path, page_index, dpi)

    image_bboxes = mineru_bboxes if mineru_bboxes else pymupdf_bboxes
    image_bboxes = _normalize_and_filter_bboxes(
        image_bboxes,
        page_width=page_width,
        page_height=page_height,
    )

    is_full_page_scan = False
    if len(image_bboxes) == 1 and page_width > 0 and page_height > 0:
        bbox = image_bboxes[0]
        img_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        page_area = page_width * page_height
        if page_area > 0 and img_area / page_area > 0.82:
            is_full_page_scan = True
            _log.debug(
                "page %s: detected full-page scan (%.1f%%), switching to scanned-page visual mode",
                page_index + 1,
                (img_area / page_area) * 100,
            )

    # 扫描页 / 无可靠 bbox：改走视觉内容块检测
    if not image_bboxes or is_full_page_scan:
        return crop_figures_from_scanned_page_by_visual_regions(
            page_md=page_md,
            page_image_path=page_image_path,
            markdown_out_path=markdown_out_path,
            page_index=page_index,
            max_per_page=max_per_page,
        )

    out: list[tuple[str, str]] = []

    with Image.open(page_image_path) as im:
        img_width, img_height = im.size

    scale_x = img_width / page_width if page_width > 0 else 1.0
    scale_y = img_height / page_height if page_height > 0 else 1.0

    # 把 PDF 坐标转成像素坐标，并扩边
    pixel_boxes: list[tuple[int, int, int, int]] = []
    for bbox in image_bboxes:
        x0, y0, x1, y1 = bbox
        pb = (
            int(x0 * scale_x),
            int(y0 * scale_y),
            int(x1 * scale_x),
            int(y1 * scale_y),
        )
        pb = _expand_bbox_px(pb, img_width, img_height, pad_ratio=0.03, min_pad=max(10, int(min(img_width, img_height) * 0.008)))
        pixel_boxes.append(pb)

    # 不再按 idx 直接硬配，先按 caption 的位置标记做弱匹配
    used = set()

    def match_score(cap: CaptionLine, box: tuple[int, int, int, int]) -> float:
        x0, y0, x1, y1 = box
        area = _bbox_area(box)
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        score = area

        pos = (cap.position or "").strip()
        if pos:
            if "左" in pos and cx < img_width * 0.55:
                score += area * 0.8
            if "右" in pos and cx > img_width * 0.45:
                score += area * 0.8
            if "上" in pos or "顶部" in pos:
                if cy < img_height * 0.60:
                    score += area * 0.5
            if "下" in pos or "底部" in pos:
                if cy > img_height * 0.40:
                    score += area * 0.5
            if "中" in pos and img_height * 0.20 < cy < img_height * 0.80:
                score += area * 0.2

        return score

    for idx, cap in enumerate(captions[:max_per_page], start=1):
        candidates = [(j, b) for j, b in enumerate(pixel_boxes) if j not in used]
        if not candidates:
            break

        candidates = sorted(candidates, key=lambda x: match_score(cap, x[1]), reverse=True)
        best_j, best_bbox = candidates[0]
        used.add(best_j)

        out_path = figures_dir / f"page_{page_index + 1:04d}_fig_{idx:02d}.png"
        ok = crop_image_by_bbox(
            page_image_path=page_image_path,
            out_path=out_path,
            bbox=best_bbox,
            margin=max(10, int(min(img_width, img_height) * 0.008)),
        )
        if ok:
            rel = out_path.relative_to(markdown_out_path.parent).as_posix()
            out.append((cap.text, rel))

    if out:
        return out

    return crop_figures_from_scanned_page_by_visual_regions(
        page_md=page_md,
        page_image_path=page_image_path,
        markdown_out_path=markdown_out_path,
        page_index=page_index,
        max_per_page=max_per_page,
    )



def crop_figures_by_captions_from_page_image(
    *,
    page_md: str,
    page_image_path: Path,
    markdown_out_path: Path,
    page_index: int,
    max_per_page: int,
) -> list[tuple[str, str]]:
    """
    基于 VLM 位置标注和图题顺序分配裁剪区域。
    为每个图题生成一个独立的裁剪区域。
    返回 [(caption_text, relative_image_path), ...]
    """
    captions = _extract_figure_caption_lines(page_md, max_per_page=max_per_page)
    if not captions:
        return []

    figures_dir = markdown_out_path.parent / f"{markdown_out_path.stem}_figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(page_image_path) as im:
        img_width, img_height = im.size

    out: list[tuple[str, str]] = []
    num_captions = len(captions)

    crop_regions = _compute_crop_regions_for_page(
        num_figures=num_captions,
        page_width=img_width,
        page_height=img_height,
        captions=captions,
    )

    for idx, (cap, region) in enumerate(zip(captions, crop_regions)):
        out_path = figures_dir / f"page_{page_index + 1:04d}_fig_{idx + 1:02d}.png"
        ok = _crop_region_from_image(
            page_image_path=page_image_path,
            out_path=out_path,
            region=region,
        )
        if ok:
            rel = out_path.relative_to(markdown_out_path.parent).as_posix()
            out.append((cap.text, rel))

    return out


def _compute_crop_regions_for_page(
    num_figures: int,
    page_width: int,
    page_height: int,
    captions: list[CaptionLine],
) -> list[tuple[int, int, int, int]]:
    """
    根据图题数量和位置标注，计算每个图的裁剪区域 (x0, y0, x1, y1)。
    使用更智能的分区策略，避免重复和遗漏。
    """
    if num_figures == 0:
        return []

    if num_figures == 1:
        cap = captions[0]
        position = cap.position
        if position:
            crop_mode = _get_crop_mode_for_position(position)
            return [_mode_to_region(crop_mode, page_width, page_height)]
        return [_mode_to_region("single_full", page_width, page_height)]

    if num_figures == 2:
        cap0_pos = captions[0].position if captions[0].position else None
        cap1_pos = captions[1].position if captions[1].position else None

        mode0 = _get_crop_mode_for_position(cap0_pos) if cap0_pos else None
        mode1 = _get_crop_mode_for_position(cap1_pos) if cap1_pos else None

        # ★ 修复：若两个 mode 相同或均未提供，判断是左右还是上下
        if mode0 is None and mode1 is None:
            # 无位置标注时，默认上下分（比左右分更常见于竖排扫描文档）
            mode0, mode1 = "top_half", "bottom_half"
        elif mode0 == mode1 or mode0 is None or mode1 is None:
            # 有一个有标注但相同，用位置推断
            ref_mode = mode0 or mode1
            if ref_mode in ("left_half", "right_half", "top_left", "top_right",
                            "bottom_left", "bottom_right"):
                # 明确有左/右信息
                if "left" in ref_mode or "左" in (cap0_pos or ""):
                    mode0, mode1 = "left_half", "right_half"
                else:
                    mode0, mode1 = "top_half", "bottom_half"
            else:
                mode0, mode1 = "top_half", "bottom_half"
        else:
            # 两个都有且不同，直接用
            pass

        return [
            _mode_to_region(mode0, page_width, page_height),
            _mode_to_region(mode1, page_width, page_height),
        ]


    if num_figures == 3:
        positions = []
        for cap in captions:
            pos = cap.position if cap.position else None
            if pos:
                positions.append(_get_crop_mode_for_position(pos))
            else:
                positions.append(None)

        modes = []
        for i, m in enumerate(positions):
            if m and m not in modes:
                modes.append(m)
            else:
                if i == 0:
                    modes.append("top_left")
                elif i == 1:
                    modes.append("top_right")
                else:
                    modes.append("bottom_center")

        return [
            _mode_to_region(modes[0], page_width, page_height),
            _mode_to_region(modes[1], page_width, page_height),
            _mode_to_region(modes[2], page_width, page_height),
        ]

    positions = []
    for cap in captions:
        pos = cap.position if cap.position else None
        if pos:
            positions.append(_get_crop_mode_for_position(pos))
        else:
            positions.append(None)

    modes = []
    used = set()
    fallback_order = ["top_left", "top_right", "bottom_left", "bottom_right", "single_full"]

    for i, m in enumerate(positions):
        if m and m not in used:
            modes.append(m)
            used.add(m)
        else:
            for fb in fallback_order:
                if fb not in used:
                    modes.append(fb)
                    used.add(fb)
                    break
            else:
                modes.append("single_full")

    return [_mode_to_region(m, page_width, page_height) for m in modes]


def _mode_to_region(
    mode: str,
    page_width: int,
    page_height: int,
) -> tuple[int, int, int, int]:
    """
    将裁剪模式转换为像素坐标区域 (x0, y0, x1, y1)。
    对 fallback 使用重叠式区域，宁大勿残，后续再 trim。
    """
    if mode == "single_full":
        left = int(page_width * 0.02)
        top = int(page_height * 0.02)
        right = int(page_width * 0.98)
        bottom = int(page_height * 0.98)

    elif mode == "left_half":
        left = int(page_width * 0.00)
        top = int(page_height * 0.02)
        right = int(page_width * 0.58)
        bottom = int(page_height * 0.98)

    elif mode == "right_half":
        left = int(page_width * 0.42)
        top = int(page_height * 0.02)
        right = int(page_width * 1.00)
        bottom = int(page_height * 0.98)

    elif mode == "top_half":
        left = int(page_width * 0.02)
        top = int(page_height * 0.00)
        right = int(page_width * 0.98)
        bottom = int(page_height * 0.58)

    elif mode == "bottom_half":
        left = int(page_width * 0.02)
        top = int(page_height * 0.42)
        right = int(page_width * 0.98)
        bottom = int(page_height * 1.00)

    elif mode == "top_left":
        left = int(page_width * 0.00)
        top = int(page_height * 0.00)
        right = int(page_width * 0.58)
        bottom = int(page_height * 0.58)

    elif mode == "top_right":
        left = int(page_width * 0.42)
        top = int(page_height * 0.00)
        right = int(page_width * 1.00)
        bottom = int(page_height * 0.58)

    elif mode == "bottom_left":
        left = int(page_width * 0.00)
        top = int(page_height * 0.42)
        right = int(page_width * 0.58)
        bottom = int(page_height * 1.00)

    elif mode == "bottom_right":
        left = int(page_width * 0.42)
        top = int(page_height * 0.42)
        right = int(page_width * 1.00)
        bottom = int(page_height * 1.00)

    elif mode == "bottom_center":
        left = int(page_width * 0.18)
        top = int(page_height * 0.42)
        right = int(page_width * 0.82)
        bottom = int(page_height * 1.00)

    else:
        left = int(page_width * 0.02)
        top = int(page_height * 0.02)
        right = int(page_width * 0.98)
        bottom = int(page_height * 0.98)

    return (left, top, right, bottom)



def _crop_region_from_image(
    *,
    page_image_path: Path,
    out_path: Path,
    region: tuple[int, int, int, int],
) -> bool:
    try:
        with Image.open(page_image_path) as im:
            w, h = im.size
            x0, y0, x1, y1 = region

            x0 = max(0, min(x0, w - 1))
            y0 = max(0, min(y0, h - 1))
            x1 = max(x0 + 1, min(x1, w))
            y1 = max(y0 + 1, min(y1, h))

            if x1 - x0 < 20 or y1 - y0 < 20:
                _log.warning("Crop region too small: %sx%s", x1 - x0, y1 - y0)
                return False

            crop = im.crop((x0, y0, x1, y1))
            crop = _trim_whitespace_pil(crop, bg_threshold=248, pad=18)

            if crop.size[0] < 20 or crop.size[1] < 20:
                return False

            crop.save(out_path)
            return True
    except Exception as e:
        _log.warning("_crop_region_from_image failed: %s", repr(e))
        return False



def _clean_position_markers(markdown: str) -> str:
    return _POSITION_RE.sub("", markdown)


def get_page_image_positions(
    pdf_path: Path,
    page_index: int,
    dpi: float,
) -> list[tuple[float, float, float, float]]:
    """
    使用 PyMuPDF 获取指定页面上所有图片的边界框（原始坐标，非渲染后坐标）。
    返回: [(x0, y0, x1, y1), ...]
    """
    try:
        import fitz
    except ImportError:
        return []

    doc = fitz.open(str(pdf_path))
    try:
        if page_index >= doc.page_count:
            return []
        page = doc[page_index]
        page_width = page.rect.width
        page_height = page.rect.height

        image_list: list[tuple[float, float, float, float]] = []

        for img_info in page.get_image_info():
            if "bbox" in img_info and img_info["bbox"]:
                bbox = img_info["bbox"]
                if all(isinstance(x, (int, float)) for x in bbox):
                    image_list.append(tuple(bbox))

        return image_list
    finally:
        doc.close()


def get_mineru_image_positions(
    pdf_path: Path,
    page_index: int,
) -> list[tuple[float, float, float, float]]:
    """
    使用 Magic-PDF (MinerU) 的布局分析获取页面上图片的精确边界框。
    Magic-PDF 对扫描 PDF 有更好的布局识别能力。
    返回: [(x0, y0, x1, y1), ...] 原始 PDF 坐标
    """
    try:
        from magic_pdf.data.data_reader_writer import FileBasedDataReader
        from magic_pdf.data.dataset import PymuDocDataset
        from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
        from magic_pdf.model.magic_model import MagicModel
        from magic_pdf.config.ocr_content_type import CategoryId
    except ImportError:
        _log.warning(
            "MinerU（magic-pdf）未安装，本页跳过 Magic-PDF 图片定位，将用 PyMuPDF 提取 bbox；"
            "page=%s file=%s",
            page_index + 1,
            pdf_path.name,
        )
        return []

    try:
        reader = FileBasedDataReader("")
        pdf_bits = reader.read(str(pdf_path))
        dataset = PymuDocDataset(pdf_bits)

        # magic-pdf 的 doc_analyze 在不同版本里参数名/位置可能不同：
        # - 有的版本是 doc_analyze(doc=dataset, ...)
        # - 有的版本是 doc_analyze(dataset=dataset, ...) 或 doc_analyze(dataset, ...)
        # 这里做兼容调用，避免因关键字不匹配直接触发整页回退。
        try:
            import inspect

            sig = inspect.signature(doc_analyze)
            params = sig.parameters
            if "doc" in params:
                model_list = doc_analyze(doc=dataset, ocr=True, show_log=False)
            elif "dataset" in params:
                model_list = doc_analyze(dataset=dataset, ocr=True, show_log=False)
            else:
                model_list = doc_analyze(dataset, ocr=True, show_log=False)
        except TypeError:
            # 最后兜底：有些实现只接受位置参数（甚至不支持 ocr/show_log）
            try:
                model_list = doc_analyze(dataset, ocr=True, show_log=False)
            except TypeError:
                model_list = doc_analyze(dataset)

        magic_model = MagicModel(model_list, dataset)

        page_w, page_h = magic_model.get_page_size(page_index)

        img_groups = magic_model.get_imgs_v2(page_index)

        image_bboxes: list[tuple[float, float, float, float]] = []

        for img_group in img_groups:
            if "image_body" in img_group:
                body = img_group["image_body"]
                if isinstance(body, dict) and "bbox" in body:
                    bbox = body["bbox"]
                    if bbox and len(bbox) == 4:
                        x0, y0, x1, y1 = bbox
                        img_area = (x1 - x0) * (y1 - y0)
                        page_area = page_w * page_h
                        if img_area / page_area < 0.90:
                            image_bboxes.append((float(x0), float(y0), float(x1), float(y1)))

        _log.debug(
            "Magic-PDF found %s images on page %s: %s",
            len(image_bboxes),
            page_index + 1,
            image_bboxes,
        )
        return image_bboxes

    except Exception as e:
        _log.warning(
            "MinerU（Magic-PDF）本页图片定位失败，将回退 PyMuPDF；page=%s file=%s err=%s",
            page_index + 1,
            pdf_path.name,
            repr(e),
        )
        return []


def get_docling_image_positions(
    pdf_path: Path,
    page_index: int,
) -> list[tuple[float, float, float, float]]:
    """
    使用 Docling 的布局分析获取页面上图片的精确边界框。
    返回: [(x0, y0, x1, y1), ...]
    注意：Docling 对于扫描PDF无法识别单独图片，此函数只对原生PDF有效。
    """
    try:
        from docling.document_converter import DocumentConverter
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.base_models import InputFormat, PdfFormatOption
    except ImportError:
        _log.warning("Docling not available for image position detection")
        return []

    try:
        pdf_opts = PdfPipelineOptions()
        pdf_opts.generate_picture_images = True
        pdf_opts.generate_page_images = False
        pdf_opts.do_ocr = False
        pdf_opts.do_table_structure = False
        pdf_opts.do_formula_enrichment = False

        conv = DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts),
            },
        )

        result = conv.convert(str(pdf_path), max_num_pages=page_index + 1)
        doc = result.document

        image_bboxes: list[tuple[float, float, float, float]] = []

        page_rect = None
        try:
            if hasattr(doc, 'pages') and page_index < len(doc.pages):
                page_rect = doc.pages[page_index].size
        except Exception:
            pass

        if page_rect:
            page_width = page_rect[0]
            page_height = page_rect[1]
        else:
            return []

        try:
            for element in doc.iter_image_pairs():
                try:
                    if hasattr(element, "image") and element.image:
                        img = element.image
                        if hasattr(img, "bbox") and img.bbox:
                            bbox = img.bbox
                            x0 = float(bbox.l)
                            y0 = float(bbox.t)
                            x1 = float(bbox.r)
                            y1 = float(bbox.b)

                            if x1 > x0 and y1 > y0:
                                img_area = (x1 - x0) * (y1 - y0)
                                page_area = page_width * page_height
                                if img_area / page_area < 0.85:
                                    image_bboxes.append((x0, y0, x1, y1))
                except Exception:
                    continue
        except Exception:
            pass

        _log.debug(
            "Docling found %s images on page %s",
            len(image_bboxes),
            page_index + 1,
        )
        return image_bboxes

    except Exception as e:
        _log.warning("Docling image position detection failed: %s", repr(e))
        return []


def scale_bbox_to_target_dpi(
    bbox: tuple[float, float, float, float],
    page_width: float,
    page_height: float,
    target_dpi: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """
    将图片边界框从原始 PDF 坐标转换为渲染后 PNG 的像素坐标。
    """
    x0, y0, x1, y1 = bbox

    scale_x = (image_width / page_width) if page_width > 0 else 1.0
    scale_y = (image_height / page_height) if page_height > 0 else 1.0

    px0 = int(x0 * scale_x)
    py0 = int(y0 * scale_y)
    px1 = int(x1 * scale_x)
    py1 = int(y1 * scale_y)

    return (px0, py0, px1, py1)


def crop_image_by_bbox(
    *,
    page_image_path: Path,
    out_path: Path,
    bbox: tuple[int, int, int, int],
    margin: int = 10,
) -> bool:
    try:
        with Image.open(page_image_path) as im:
            w, h = im.size
            x0, y0, x1, y1 = bbox

            bw = max(1, x1 - x0)
            bh = max(1, y1 - y0)
            aspect = max(bw / max(bh, 1), bh / max(bw, 1))

            # 基础安全边
            pad_x = max(margin, int(bw * 0.10))
            pad_y = max(margin, int(bh * 0.10))

            # 细长图：朝短边方向额外扩，保住尺寸线/外侧标注
            if bh > bw * 2.2:   # 纵向细长
                pad_x = max(pad_x, int(bw * 0.35))
                pad_y = max(pad_y, int(bh * 0.12))
            elif bw > bh * 2.2: # 横向细长
                pad_x = max(pad_x, int(bw * 0.12))
                pad_y = max(pad_y, int(bh * 0.35))

            # 非常大的图，再给一圈页面级安全边
            pad_x = max(pad_x, int(w * 0.015))
            pad_y = max(pad_y, int(h * 0.015))

            x0 = max(0, x0 - pad_x)
            y0 = max(0, y0 - pad_y)
            x1 = min(w, x1 + pad_x)
            y1 = min(h, y1 + pad_y)

            if x1 - x0 < 20 or y1 - y0 < 20:
                return False

            crop = im.crop((x0, y0, x1, y1))

            # 这里尽量不要再强 trim
            crop = _trim_whitespace_pil(crop, bg_threshold=249, pad=28)

            if crop.size[0] < 20 or crop.size[1] < 20:
                return False

            crop.save(out_path)
            return True
    except Exception as e:
        _log.warning("crop_image_by_bbox failed: %s", repr(e))
        return False





def inject_cropped_figures_into_page_markdown(
    *,
    page_md: str,
    caption_and_refs: list[tuple[str, str]],
) -> str:
    if not caption_and_refs:
        return _clean_position_markers(page_md)

    lines = page_md.splitlines()
    used: set[int] = set()

    for caption_text, ref in reversed(caption_and_refs):
        insert_pos = None
        best_score = 0.0

        for i, line in enumerate(lines):
            if i in used:
                continue
            score = _caption_match_score(line.strip(), caption_text.strip())
            if score > best_score:
                best_score = score
                insert_pos = i

        if insert_pos is None or best_score < 0.72:
            continue

        used.add(insert_pos)
        lines[insert_pos:insert_pos] = [f"![{caption_text}]({ref})", ""]

    return _clean_position_markers("\n".join(lines))


def _markdown_for_failed_vl_page() -> str:
    """单页异常时的正文占位（具体失败页码由任务 result_extra 在前端展示）。"""
    return "（本页转写未成功）\n"


def transcribe_pdf_with_vl(
    *,
    client: DashScopeClient,
    model: str,
    pdf_path: Path,
    markdown_out_path: Optional[Path] = None,
    dpi: float = 150.0,
    workers: int = 1,
    caption_crop_figures: bool = False,
    caption_crop_max_per_page: int = 6,
    table_second_pass: bool = True,
    table_second_pass_max_tables: int = 0,
    max_pages: Optional[int] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    page_markdown_postprocess: Optional[Callable[[str], str]] = None,
) -> tuple[str, list[int]]:
    """
    逐页渲染 PDF → PNG（临时文件），每页一次 multimodal 调用，拼接为单个 Markdown。

    page_markdown_postprocess：单页正文定稿后（含表格纠错、可选插图裁剪）再调用，用于例如逐页插入表格语义说明。
    table_second_pass_max_tables：每页可疑表纠错上限；0 表示不限制。

    返回 (markdown, failed_pages_1based)：后者为转写异常页的 1-based 页码列表（供任务记录与前端提示）。
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
            return f"# {pdf_path.name}\n\n（空 PDF）\n", []
        limit = n if max_pages is None else min(n, max(1, int(max_pages)))
        parts: list[str] = [f"# {pdf_path.name}\n\n"]

        tmp_root = Path(tempfile.gettempdir()) / "docling_pdf_vl_pages"
        tmp_root.mkdir(parents=True, exist_ok=True)

        log_event(
            _log,
            logging.INFO,
            "pdf_vl.run.start",
            file=pdf_path.name,
            total_pages=n,
            process_pages=limit,
            dpi=dpi_f,
            workers=workers_i,
            table_review=bool(table_second_pass),
            crop_figures=bool(caption_crop_figures and markdown_out_path is not None),
            page_md_postprocess=bool(page_markdown_postprocess),
        )

        page_pngs: list[Path] = []
        for page_index in range(limit):
            if cancel_check is not None and cancel_check():
                log_event(_log, logging.INFO, "pdf_vl.render.cancelled", rendered=len(page_pngs), total=limit)
                break
            t0 = time.perf_counter()
            page = doc[page_index]
            pix = page.get_pixmap(dpi=int(round(dpi_f)))
            png_path = tmp_root / f"{pdf_path.stem}.page_{page_index + 1:04d}.png"
            pix.save(str(png_path))
            page_pngs.append(png_path)
            try:
                w = int(getattr(pix, "width", 0) or 0)
                h = int(getattr(pix, "height", 0) or 0)
            except Exception:
                w, h = 0, 0
            size_kb = 0
            try:
                size_kb = int(round(png_path.stat().st_size / 1024.0))
            except Exception:
                size_kb = 0
            _log.info(
                kv(
                    event="pdf_vl.page.rendered",
                    page=f"{page_index + 1}/{limit}",
                    elapsed=f"{time.perf_counter() - t0:.2f}s",
                    size=f"{w}x{h}",
                    kb=size_kb,
                )
            )

        vl_failed_pages_1based: list[int] = []

        def _one_page(i: int, image_path: Path) -> tuple[int, str]:
            if cancel_check is not None and cancel_check():
                return i, "（已取消）\n"
            _log.info(kv(event="pdf_vl.page.start", page=f"{i + 1}/{limit}", dpi=dpi_f, file=pdf_path.name))
            try:
                page_t0 = time.perf_counter()
                img_kb = 0
                try:
                    img_kb = int(round(image_path.stat().st_size / 1024.0))
                except Exception:
                    img_kb = 0
                _log.info(kv(event="pdf_vl.page.image", page=f"{i + 1}/{limit}", kb=img_kb, image=image_path.name))
                user_text = _VL_USER_TMPL.format(
                    name=pdf_path.name,
                    idx=i + 1,
                    total=limit,
                )
                messages = [
                    build_system_message(_VL_SYSTEM),
                    build_vl_user_message(text=user_text, image_paths=[str(image_path)]),
                ]
                page_md = ""
                raw_last = ""
                for attempt in range(1, 4):
                    if cancel_check is not None and cancel_check():
                        return i, "（已取消）\n"
                    raw_last = client.generate_multimodal(
                        model,
                        messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        biz_stage="页转写",
                    )
                    page_md = _normalize_markdown_output(raw_last).strip()
                    if page_md:
                        break
                    _log.warning(kv(event="pdf_vl.page.empty_retry", page=f"{i + 1}/{limit}", attempt=attempt, file=pdf_path.name))
                    if attempt < 3:
                        time.sleep(1.5 * attempt)
                if not page_md:
                    snippet = (raw_last or "")[:400].replace("\n", "\\n")
                    _log.error(kv(event="pdf_vl.page.empty_placeholder", page=f"{i + 1}/{limit}", file=pdf_path.name, snippet=snippet))
                    page_md = "（本页模型输出为空）"
                    _log.info(kv(event="pdf_vl.page.output.placeholder", page=f"{i + 1}/{limit}"))
                else:
                    # 产出快速统计（便于扫读）
                    n_chars_raw = len(page_md)
                    n_lines_raw = page_md.count("\n") + 1 if page_md else 0
                    n_tables_raw = len(extract_markdown_table_blocks(page_md))
                    _log.info(kv(event="pdf_vl.page.output.raw", page=f"{i + 1}/{limit}", lines=n_lines_raw, chars=n_chars_raw, tables=n_tables_raw))
                    page_md = _normalize_gfm_tables(page_md)
                    if table_second_pass:
                        if cancel_check is not None and cancel_check():
                            return i, "（已取消）\n"
                        t_review0 = time.perf_counter()
                        page_md, reviewed_n = _review_suspicious_tables_with_llm(
                            client=client,
                            model=model,
                            page_md=page_md,
                            page_image_path=image_path,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            max_tables=int(table_second_pass_max_tables),
                        )
                        _log.info(kv(event="pdf_vl.page.table_review.done", page=f"{i + 1}/{limit}", patched=reviewed_n, elapsed=f"{time.perf_counter() - t_review0:.2f}s"))

                    if caption_crop_figures and markdown_out_path is not None:
                        if cancel_check is not None and cancel_check():
                            return i, "（已取消）\n"
                        t_crop0 = time.perf_counter()
                        try:
                            caption_and_refs = crop_figures_by_docling_layout(
                                page_md=page_md,
                                page_image_path=image_path,
                                markdown_out_path=markdown_out_path,
                                page_index=i,
                                max_per_page=max(1, int(caption_crop_max_per_page)),
                                pdf_path=pdf_path,
                                dpi=dpi_f,
                            )
                            if caption_and_refs:
                                _log.info(kv(event="pdf_vl.page.figure_crop.patched", page=f"{i + 1}/{limit}", images=len(caption_and_refs), file=pdf_path.name))
                                page_md = inject_cropped_figures_into_page_markdown(
                                    page_md=page_md,
                                    caption_and_refs=caption_and_refs,
                                )
                            _log.info(kv(event="pdf_vl.page.figure_crop.done", page=f"{i + 1}/{limit}", elapsed=f"{time.perf_counter() - t_crop0:.2f}s", images=len(caption_and_refs) if caption_and_refs else 0))
                        except Exception as e:
                            _log.warning(kv(event="pdf_vl.page.figure_crop.failed_ignored", page=f"{i + 1}/{limit}", file=pdf_path.name, err=repr(e)))
                            _log.info(kv(event="pdf_vl.page.figure_crop.ignored", page=f"{i + 1}/{limit}", elapsed=f"{time.perf_counter() - t_crop0:.2f}s"))

                    if page_markdown_postprocess is not None:
                        if cancel_check is not None and cancel_check():
                            return i, "（已取消）\n"
                        t_cap0 = time.perf_counter()
                        page_md = page_markdown_postprocess(page_md)
                        _log.info(
                            kv(
                                event="pdf_vl.page.markdown_postprocess.done",
                                page=f"{i + 1}/{limit}",
                                elapsed=f"{time.perf_counter() - t_cap0:.2f}s",
                            )
                        )

                _log.info(kv(event="pdf_vl.page.done", page=f"{i + 1}/{limit}", elapsed=f"{time.perf_counter() - page_t0:.2f}s", file=pdf_path.name))
                return i, page_md
            except Exception as e:
                vl_failed_pages_1based.append(i + 1)
                _log.error(kv(event="pdf_vl.page.failed_skipped", page=f"{i + 1}/{limit}", file=pdf_path.name, err=repr(e)))
                return i, _markdown_for_failed_vl_page()


        if workers_i == 1:
            page_md_map: dict[int, str] = {}
            for i, p in enumerate(page_pngs):
                if cancel_check is not None and cancel_check():
                    break
                idx, md = _one_page(i, p)
                page_md_map[idx] = md
                if progress_callback is not None:
                    progress_callback(i + 1, limit)
        else:
            log_event(_log, logging.INFO, "pdf_vl.run.parallel", workers=workers_i, pages=limit, file=pdf_path.name)
            page_md_map = {}
            done_pages = 0
            with ThreadPoolExecutor(max_workers=workers_i) as ex:
                it = iter(list(enumerate(page_pngs)))
                in_flight = set()
                # 先投递一批
                while len(in_flight) < workers_i:
                    try:
                        i, p = next(it)
                    except StopIteration:
                        break
                    if cancel_check is not None and cancel_check():
                        break
                    in_flight.add(ex.submit(_one_page, i, p))

                while in_flight:
                    if cancel_check is not None and cancel_check():
                        log_event(_log, logging.INFO, "pdf_vl.parallel.cancelled", in_flight=len(in_flight))
                    for fut in as_completed(list(in_flight), timeout=None):
                        in_flight.remove(fut)
                        idx, md = fut.result()
                        page_md_map[idx] = md
                        done_pages += 1
                        if progress_callback is not None:
                            progress_callback(done_pages, limit)
                        # 继续补充新任务（若未取消）
                        if cancel_check is None or not cancel_check():
                            try:
                                ni, np_ = next(it)
                            except StopIteration:
                                continue
                            in_flight.add(ex.submit(_one_page, ni, np_))
                        # 每完成一个就跳出 for，回到 while 重新检查取消（更快响应）
                        break

        # 若取消：只拼接已完成页，其余输出占位并终止
        cancelled = bool(cancel_check is not None and cancel_check())
        effective_limit = limit
        if cancelled:
            effective_limit = max(page_md_map.keys(), default=-1) + 1
            effective_limit = max(0, min(effective_limit, limit))
        for i in range(effective_limit):
            parts.append(f"## 第 {i + 1} / {limit} 页\n\n")
            parts.append(page_md_map.get(i, "（本页模型输出缺失）"))
            parts.append("\n\n---\n\n")
    finally:
        doc.close()

    return "".join(parts).rstrip() + "\n", sorted(set(vl_failed_pages_1based))