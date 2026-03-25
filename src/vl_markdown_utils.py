from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

_log = logging.getLogger(__name__)

_IMAGE_MD_RE = re.compile(
    r"!\[[^\]]*\]\((?P<path>[^)\s]+)(?:\s+\"[^\"]*\")?\)"
)


def extract_markdown_image_refs(markdown_text: str) -> list[str]:
    """
    从 Markdown 中提取 `![](<path>)` 里的 `<path>` 字符串。

    Docling 的 markdown 里通常不带 title（或带 `"<title>"`）。
    """
    if not markdown_text:
        return []
    return [m.group("path") for m in _IMAGE_MD_RE.finditer(markdown_text)]


def extract_markdown_image_refs_with_line_index(
    markdown_text: str,
) -> list[tuple[int, str]]:
    """
    返回 [(line_index, path), ...]，line_index 为 0-based 行号。
    用于把图片“靠近某个 table block”。
    """
    out: list[tuple[int, str]] = []
    if not markdown_text:
        return out
    for i, line in enumerate(markdown_text.splitlines()):
        m = _IMAGE_MD_RE.search(line)
        if not m:
            continue
        out.append((i, m.group("path")))
    return out


@dataclass(frozen=True)
class TableBlock:
    start_line: int  # inclusive, 0-based
    end_line: int  # inclusive, 0-based
    markdown: str


def extract_markdown_table_blocks(markdown_text: str) -> list[TableBlock]:
    """
    粗粒度提取 GitHub 风格的 pipe 表格块。

    规则（稳妥优先）：
    - 遇到连续若干行满足：strip 后以 '|' 开头且包含至少 1 个 '|'
    - 表格块至少 2 行（通常 header + separator）
    - 不尝试解析对齐/宽度，仅用于表格局部纠错与列数校验
    """
    if not markdown_text:
        return []
    lines = markdown_text.splitlines()
    blocks: list[TableBlock] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not (line.startswith("|") and line.count("|") >= 2):
            i += 1
            continue

        start = i
        j = i
        while j < len(lines):
            l = lines[j].strip()
            if l.startswith("|") and l.count("|") >= 2:
                j += 1
                continue
            break

        end = j - 1
        if end - start + 1 >= 2:
            md = "\n".join(lines[start : end + 1])
            blocks.append(TableBlock(start_line=start, end_line=end, markdown=md))
        i = max(j, i + 1)
    return blocks


def table_column_count(table_markdown: str) -> int:
    """
    从第一行估算列数：按 '|' split 去掉两端空串。
    """
    for line in (table_markdown or "").splitlines():
        t = line.strip()
        if t.startswith("|") and t.count("|") >= 2:
            parts = [p.strip() for p in t.split("|")]
            # 常见：第一/最后 split 结果为空串
            if parts and parts[0] == "":
                parts = parts[1:]
            if parts and parts[-1] == "":
                parts = parts[:-1]
            return max(0, len(parts))
    return 0


def validate_table_output_invariants(
    *,
    original_table: str,
    refined_table: str,
) -> tuple[bool, dict[str, object]]:
    """
    保证 refined_table 不改变行/列“骨架”：
    - 列数一致
    - 行数不变（忽略纯空行）
    """
    if not refined_table or not refined_table.strip():
        return False, {"reason": "empty_output"}

    def _row_cols(line: str) -> int:
        t = line.strip()
        if not (t.startswith("|") and t.count("|") >= 2):
            return 0
        parts = [p.strip() for p in t.split("|")]
        if parts and parts[0] == "":
            parts = parts[1:]
        if parts and parts[-1] == "":
            parts = parts[:-1]
        return max(0, len(parts))

    o_lines = [l for l in original_table.splitlines() if l.strip().startswith("|")]
    r_lines = [l for l in refined_table.splitlines() if l.strip().startswith("|")]
    if len(o_lines) != len(r_lines):
        return False, {"reason": "row_count_changed", "orig_rows": len(o_lines), "new_rows": len(r_lines)}

    expected_cols = table_column_count(original_table)
    if expected_cols == 0:
        return False, {"reason": "orig_col_count_invalid", "orig_cols": expected_cols}

    # 逐行检查列数，避免输出丢列导致“看起来还像表格但实际列不一致”
    for idx, line in enumerate(r_lines):
        cols = _row_cols(line)
        if cols != expected_cols:
            return False, {
                "reason": "row_column_count_changed",
                "expected_cols": expected_cols,
                "bad_row_index": idx,
                "bad_row_cols": cols,
            }

    return True, {"orig_rows": len(o_lines), "new_rows": len(r_lines), "cols": expected_cols}


def _looks_like_windows_abs(p: str) -> bool:
    # e.g. C:\x\y.png
    return len(p) >= 3 and p[1] == ":" and (p[2] == "\\" or p[2] == "/")


def resolve_image_refs(
    *,
    refs: Iterable[str],
    markdown_out_path: Path,
    max_images: int,
    image_mode: Literal["local_abs", "url"] = "local_abs",
) -> list[str]:
    """
    将 markdown 图片引用解析为给 Qwen-VL 的输入：
    - local_abs：本地绝对路径（优先）。若 ref 是公网 URL，则原样返回（模型可能仍能访问）。
    - url：若 ref 是本地路径则尝试转为 file://URL（否则跳过）。
    """
    out: list[str] = []
    md_dir = markdown_out_path.parent
    for ref in refs:
        if len(out) >= max_images:
            break
        ref = ref.strip()
        if not ref:
            continue

        if ref.startswith("http://") or ref.startswith("https://"):
            out.append(ref)
            continue

        if ref.startswith("file://"):
            # file://C:/xxx 或 file:///C:/xxx
            local = ref[len("file://") :]
            if image_mode == "url":
                out.append(ref)
            else:
                out.append(local)
            continue

        # local path
        if _looks_like_windows_abs(ref) or ref.startswith("/") or ref.startswith("\\"):
            local_abs = Path(ref).resolve()
        else:
            local_abs = (md_dir / ref).resolve()

        if image_mode == "local_abs":
            if local_abs.exists():
                out.append(str(local_abs))
            else:
                _log.warning("Image ref not found on disk: %s", str(local_abs))
        else:
            # url mode
            if local_abs.exists():
                out.append(local_abs.as_uri())
            else:
                _log.warning("Image ref not found on disk: %s", str(local_abs))
    return out


def validate_image_refs_invariants(
    *,
    original_md: str,
    refined_md: str,
) -> tuple[bool, dict[str, object]]:
    """
    基本不变量：
    - refined_md 中应包含与 original_md 相同数量的图片引用
    - refined_md 中每个 original ref 至少出现一次（以字符串匹配）
    """
    orig_refs = extract_markdown_image_refs(original_md)
    new_refs = extract_markdown_image_refs(refined_md)
    if not orig_refs and not new_refs:
        return True, {"orig_count": 0, "new_count": 0}
    if len(orig_refs) != len(new_refs):
        return False, {"orig_count": len(orig_refs), "new_count": len(new_refs)}

    # count 级别的多重集严格匹配太容易受 Docling/LLM 的格式影响，这里用“包含”检查更稳。
    new_set = set(new_refs)
    missing = [r for r in orig_refs if r not in new_set]
    if missing:
        return False, {"missing_refs": missing[:10], "missing_count": len(missing)}
    return True, {"orig_count": len(orig_refs), "new_count": len(new_refs)}


def summarize_markdown_quality(markdown_text: str) -> dict[str, int]:
    """
    轻量的非 LLM 指标，用于辅助 quality_check。
    注意：这不是最终评价，只是 features。
    """
    if not markdown_text:
        return {"len_chars": 0, "image_refs": 0, "table_rows": 0}
    image_refs = extract_markdown_image_refs(markdown_text)
    # 粗略统计：Markdown 表格行通常包含 '|'，且同时至少出现两列分隔。
    table_rows = 0
    for line in markdown_text.splitlines():
        if "|" in line and line.count("|") >= 2:
            table_rows += 1
    return {"len_chars": len(markdown_text), "image_refs": len(image_refs), "table_rows": table_rows}

