from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Literal, Optional

from .dashscope_client import DashScopeClient, build_system_message, build_vl_user_message
from .llm_prompts import build_cleanup_messages, build_quality_check_messages
from .vl_markdown_utils import (
    TableBlock,
    extract_markdown_image_refs,
    extract_markdown_image_refs_with_line_index,
    extract_markdown_table_blocks,
    resolve_image_refs,
    summarize_markdown_quality,
    table_column_count,
    validate_image_refs_invariants,
    validate_table_output_invariants,
)

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LlmRerunSuggestion:
    ocr_quality: Literal["fast", "balanced", "high"]
    table_accuracy: Literal["fast", "accurate"]
    generate_page_images: bool


@dataclass(frozen=True)
class QualityCheckResult:
    score: int
    need_rerun: bool
    suggest: Optional[LlmRerunSuggestion]
    raw: str


def _normalize_markdown_output(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    if t.startswith("```"):
        # remove a single fence wrapper if present
        # e.g. ```markdown ... ```
        inner = re.sub(r"^```[a-zA-Z0-9_-]*\s*\n?", "", t)
        inner = re.sub(r"\n?```$", "", inner).strip()
        # 避免模型只输出空 fence 时把正文剥成空串
        return inner if inner else t
    return t


def _parse_quality_check(text: str) -> QualityCheckResult:
    raw = text or ""
    # tolerate extra text by searching for known keys
    def _find(key: str) -> Optional[str]:
        m = re.search(rf"{re.escape(key)}=(.*)", raw)
        if not m:
            return None
        return m.group(1).strip()

    score_s = _find("QC_SCORE")
    score = int(score_s) if score_s and score_s.isdigit() else 0

    need_s = (_find("NEED_RERUN") or "no").lower()
    need_rerun = need_s in ("yes", "true", "1", "y")

    ocr_s = (_find("SUGGEST_OCR_QUALITY") or "balanced").lower()
    if ocr_s not in ("fast", "balanced", "high"):
        ocr_s = "balanced"

    table_s = (_find("SUGGEST_TABLE_ACCURACY") or "accurate").lower()
    if table_s not in ("fast", "accurate"):
        table_s = "accurate"

    img_s = (_find("SUGGEST_GENERATE_PAGE_IMAGES") or "no").lower()
    generate_page_images = img_s in ("yes", "true", "1", "y")

    suggest = LlmRerunSuggestion(
        ocr_quality=ocr_s, table_accuracy=table_s, generate_page_images=generate_page_images
    )
    return QualityCheckResult(score=score, need_rerun=need_rerun, suggest=suggest, raw=raw)


class DoclingMarkdownRefiner:
    """
    对 Docling 输出的 Markdown 做 Qwen-VL 清洗/纠错，并用 LLM 质量检查决定是否 rerun。
    """

    def __init__(
        self,
        *,
        client: DashScopeClient,
        model: str,
        cleanup_max_images: int = 6,
        vl_image_mode: Literal["local_abs", "url"] = "local_abs",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        self.client = client
        self.model = model
        self.cleanup_max_images = cleanup_max_images
        self.vl_image_mode = vl_image_mode
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _extract_first_table_block(self, output_text: str) -> str:
        """
        从模型输出里尽量提取一个表格块（连续以 '|' 开头的行）。
        """
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

    def cleanup_markdown(
        self,
        *,
        original_markdown: str,
        markdown_out_path: Any,
        image_inputs: list[str],
    ) -> str:
        start = time.time()
        messages = build_cleanup_messages(original_markdown)

        # build_cleanup_messages 已包含 system+user(纯文本)。
        # 为了让 VL 读图片，我们把 user 的 content 替换为 "images + text" 的 array。
        user_text = ""
        for m in messages:
            if m.get("role") == "user":
                user_text = str(m.get("content") or "")
                break

        user_message = build_vl_user_message(text=user_text, image_paths=image_inputs)
        messages = [m for m in messages if m.get("role") != "user"] + [user_message]

        refined = self._generate_multimodal_with_stage(
            messages=messages,
            biz_stage="文本语义增强",
        )
        refined = _normalize_markdown_output(refined)
        _log.info("LLM cleanup done in %.2fs", time.time() - start)
        return refined

    def quality_check(
        self,
        *,
        original_markdown: str,
        refined_markdown: str,
    ) -> QualityCheckResult:
        original_stats = summarize_markdown_quality(original_markdown)
        refined_stats = summarize_markdown_quality(refined_markdown)

        start = time.time()
        messages = build_quality_check_messages(
            original_markdown=original_markdown,
            refined_markdown=refined_markdown,
            original_stats=original_stats,
            refined_stats=refined_stats,
        )
        # quality_check 只需要文本；不再额外提供图片（降低 cost）。
        # 但仍使用 multimodal endpoint（model='qwen-*-vl'），传纯文本即可。
        qc_raw = self._generate_multimodal_with_stage(
            messages=messages,
            biz_stage="语义质量验收",
        )
        _log.info("LLM quality_check done in %.2fs", time.time() - start)
        return _parse_quality_check(qc_raw)

    @staticmethod
    def prepare_image_inputs(
        *,
        original_md: str,
        markdown_out_path: Any,
        vl_image_mode: Literal["local_abs", "url"],
        cleanup_max_images: int,
    ) -> list[str]:
        refs = extract_markdown_image_refs(original_md)
        return resolve_image_refs(
            refs=refs,
            markdown_out_path=markdown_out_path,
            max_images=cleanup_max_images,
            image_mode=vl_image_mode,
        )

    def cleanup_tables_per_block(
        self,
        *,
        original_markdown: str,
        markdown_out_path: Any,
        vl_image_mode: Literal["local_abs", "url"],
        cleanup_max_images_per_table: int,
        cleanup_max_tables: int,
        context_lines: int = 2,
    ) -> str:
        """
        对每个 Markdown table block 分别调用一次 Qwen-VL，只修正单元格文本。
        失败的 table 会回退到原表格，不会影响其它区域。
        """
        start = time.time()

        table_blocks = extract_markdown_table_blocks(original_markdown)
        if not table_blocks:
            return original_markdown

        # 提取图片及其行号，便于 table 附近喂图（这里用“最近的图片引用”策略）。
        image_refs_with_idx = extract_markdown_image_refs_with_line_index(original_markdown)

        lines = original_markdown.splitlines()
        refined_lines = list(lines)

        refined_tables = 0
        from .llm_prompts import build_table_cleanup_messages

        for tb in table_blocks[: cleanup_max_tables]:
            context_start = max(0, tb.start_line - context_lines)
            context_text = "\n".join(lines[context_start : tb.start_line]).strip()

            # 找最近的图片：优先取 table 前的图片，凑不满再补后面的图片
            preceding = [(idx, ref) for idx, ref in image_refs_with_idx if idx <= tb.start_line]
            following = [(idx, ref) for idx, ref in image_refs_with_idx if idx > tb.start_line]
            chosen_refs: list[str] = []
            for _, ref in reversed(preceding):
                if len(chosen_refs) >= cleanup_max_images_per_table:
                    break
                chosen_refs.append(ref)
            if len(chosen_refs) < cleanup_max_images_per_table:
                for _, ref in following:
                    if len(chosen_refs) >= cleanup_max_images_per_table:
                        break
                    chosen_refs.append(ref)

            image_inputs = resolve_image_refs(
                refs=chosen_refs,
                markdown_out_path=markdown_out_path,
                max_images=cleanup_max_images_per_table,
                image_mode=vl_image_mode,
            )

            # 保留原表格骨架列数，用于输出质量约束
            orig_cols = table_column_count(tb.markdown)
            if orig_cols == 0:
                continue

            messages = build_table_cleanup_messages(
                original_table_markdown=tb.markdown,
                context_text=context_text,
            )
            # 替换 user 为 "images + text" array（使 Qwen-VL 能看图）
            user_text = ""
            for m in messages:
                if m.get("role") == "user":
                    user_text = str(m.get("content") or "")
                    break
            user_message = {
                "role": "user",
                "content": [{"image": p} for p in image_inputs]
                + [{"text": user_text}],
            }
            messages = [m for m in messages if m.get("role") != "user"] + [user_message]

            refined_table_out = self._generate_multimodal_with_stage(
                messages=messages,
                biz_stage="表格语义增强",
            )
            refined_table_out = _normalize_markdown_output(refined_table_out)
            refined_table_only = self._extract_first_table_block(refined_table_out)

            ok, _details = validate_table_output_invariants(
                original_table=tb.markdown,
                refined_table=refined_table_only,
            )
            if not ok:
                _log.warning("Table refine failed; fallback to original table. details=%s", _details)
                continue

            # 替换指定行范围
            new_table_lines = refined_table_only.splitlines()
            expected_len = tb.end_line - tb.start_line + 1
            if len(new_table_lines) != expected_len:
                _log.warning(
                    "Table line count mismatch after refine; expected=%s actual=%s. fallback.",
                    expected_len,
                    len(new_table_lines),
                )
                continue
            refined_lines[tb.start_line : tb.end_line + 1] = new_table_lines
            refined_tables += 1

        refined_md = "\n".join(refined_lines)
        _log.info(
            "LLM per-table cleanup done in %.2fs (tables_refined=%s/%s)",
            time.time() - start,
            refined_tables,
            len(table_blocks),
        )
        return refined_md

    def _generate_multimodal_with_stage(
        self,
        *,
        messages: list[dict[str, Any]],
        biz_stage: str,
    ) -> str:
        """
        兼容旧版/测试桩客户端：若不支持 biz_stage 参数，则自动回退旧调用。
        """
        try:
            return self.client.generate_multimodal(
                self.model,
                messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                biz_stage=biz_stage,
            )
        except TypeError as e:
            if "biz_stage" not in str(e):
                raise
            return self.client.generate_multimodal(
                self.model,
                messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

