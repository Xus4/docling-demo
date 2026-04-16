"""从 Markdown 文本中扫描 GFM 管道表与 HTML <table> 块（排除围栏代码块）。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator, Literal


@dataclass(frozen=True)
class TableBlock:
    kind: Literal["gfm_table", "html_table"]
    start: int
    end: int
    raw: str


_FENCE_LINE = re.compile(r"^(\s*)(`{3,}|~{3,})")
_GFM_SEP = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$",
)
_HTML_TOKEN = re.compile(r"<table\b|</table\s*>", re.IGNORECASE)


def _fenced_code_ranges(text: str) -> list[tuple[int, int]]:
    """返回 ``(start, end)`` 半开区间列表，覆盖 ``` / ~~~ 围栏代码块（含围栏行）。"""
    lines = text.splitlines(keepends=True)
    ranges: list[tuple[int, int]] = []
    offset = 0
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        m = _FENCE_LINE.match(line)
        if not m:
            offset += len(line)
            i += 1
            continue
        fence = m.group(2)
        ch = fence[0]
        min_len = len(fence)
        start = offset
        offset += len(line)
        i += 1
        while i < n:
            line2 = lines[i]
            m2 = _FENCE_LINE.match(line2)
            if m2 and m2.group(2)[0] == ch and len(m2.group(2)) >= min_len:
                offset += len(line2)
                i += 1
                ranges.append((start, offset))
                break
            offset += len(line2)
            i += 1
        else:
            break
    return ranges


def _in_ranges(pos: int, ranges: list[tuple[int, int]]) -> bool:
    return any(s <= pos < e for s, e in ranges)


def _line_starts_pipe_row(line: str) -> bool:
    s = line.strip()
    if not s or "|" not in s:
        return False
    if s.lstrip().lower().startswith("<"):
        return False
    return True


def _is_gfm_separator_row(line: str) -> bool:
    return bool(_GFM_SEP.match(line))


def _gfm_table_extent(lines: list[str], start_idx: int, char_offsets: list[int]) -> tuple[int, int] | None:
    """若从 start_idx 起为 GFM 表，返回字符半开区间 ``[start, end)``；否则 None。"""
    if start_idx + 1 >= len(lines):
        return None
    if not _line_starts_pipe_row(lines[start_idx]):
        return None
    if not _is_gfm_separator_row(lines[start_idx + 1]):
        return None
    j = start_idx + 2
    while j < len(lines):
        line = lines[j]
        if line.strip() == "":
            break
        if _is_gfm_separator_row(line):
            j -= 1
            break
        if not _line_starts_pipe_row(line):
            break
        j += 1
    last_line_idx = j - 1
    if last_line_idx < start_idx + 1:
        return None
    start_ch = char_offsets[start_idx]
    end_ch = char_offsets[last_line_idx] + len(lines[last_line_idx])
    return start_ch, end_ch


def _find_html_table_end(text: str, table_open_start: int) -> int | None:
    """从 ``<table`` 起点开始，用深度计数匹配对应 ``</table>`` 结束位置（支持嵌套）。"""
    depth = 0
    for m in _HTML_TOKEN.finditer(text, table_open_start):
        tok = m.group(0)
        if tok.lower().startswith("<table"):
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                return m.end()
    return None


def _ranges_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 <= b0 or a0 >= b1)


def iter_table_blocks(text: str) -> Iterator[TableBlock]:
    """按文档顺序产出表格块；``start``/``end`` 为半开字符区间。"""
    fences = _fenced_code_ranges(text)
    lines = text.splitlines(keepends=True)
    char_offsets: list[int] = []
    acc = 0
    for ln in lines:
        char_offsets.append(acc)
        acc += len(ln)

    occupied: list[tuple[int, int]] = []

    def overlaps(a: int, b: int) -> bool:
        return any(_ranges_overlap(a, b, u0, u1) for u0, u1 in occupied)

    def covers_point(p: int) -> bool:
        return any(u0 <= p < u1 for u0, u1 in occupied)

    def mark(a: int, b: int) -> None:
        occupied.append((a, b))

    li = 0
    while li < len(lines):
        pos = char_offsets[li]
        if _in_ranges(pos, fences):
            li += 1
            continue
        ext = _gfm_table_extent(lines, li, char_offsets)
        if ext:
            a, b = ext
            if not overlaps(a, b):
                yield TableBlock(kind="gfm_table", start=a, end=b, raw=text[a:b])
                mark(a, b)
                while li < len(lines) and char_offsets[li] < b:
                    li += 1
                continue
        li += 1

    pos = 0
    while pos < len(text):
        m = _HTML_TOKEN.search(text, pos)
        if m is None:
            break
        if not m.group(0).lower().startswith("<table"):
            pos = m.end()
            continue
        a = m.start()
        if _in_ranges(a, fences) or covers_point(a):
            pos = m.end()
            continue
        end_abs = _find_html_table_end(text, a)
        if end_abs is None:
            pos = m.end()
            continue
        if overlaps(a, end_abs):
            pos = m.end()
            continue
        yield TableBlock(kind="html_table", start=a, end=end_abs, raw=text[a:end_abs])
        mark(a, end_abs)
        pos = end_abs
