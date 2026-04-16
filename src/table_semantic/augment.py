"""在 Markdown 中为表格插入大模型生成的语义块（幂等）。"""

from __future__ import annotations

import logging
import os
import re
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from src.table_semantic.llm_client import (
    ChatCompletionMeta,
    LLMClientError,
    OpenAICompatibleConfig,
    chat_completion_text_with_meta,
)
from src.table_semantic.table_blocks import TableBlock, iter_table_blocks

log = logging.getLogger(__name__)


INTERNAL_TABLE_CAPTION_REASONING = (
    "\n\n【内化核对（严禁写入输出）】"
    "动笔前在脑中：先识别表头与各行/各列语义角色，再按「项目—条件/取值/关系」组织成一段中文；"
    "忌笼统概括；勿把推理过程写出。"
)


@dataclass(frozen=True)
class TableCaptionParams:
    """表格转述：上下文截取与 system 中目标字数提示。"""

    context_before_chars: int = 3000
    context_after_chars: int = 3000
    caption_target_chars: int = 800
    #: 表体 ``raw`` 超过该字符数则跳过语义补充（不调 LLM）。``<= 0`` 表示不限制。
    max_table_chars: int = 2000


DEFAULT_TABLE_CAPTION_PARAMS = TableCaptionParams()


def build_table_caption_messages(
    *,
    table_markdown: str,
    context_text: str,
    max_chars: int = 800,
) -> list[dict[str, str]]:
    system = (
        "你是工业文档表格转述助手：把表格信息转写为**一段连贯中文**，使读者不看表也能把握主要项目与对应关系。"
        "硬性要求："
        "（1）只依据表格与所给上下文中明确可见的信息，不编造、不引申背景；"
        "（2）准确交代行/列/类别之间的对应关系，尤其是「不同类别对应不同要求或状态」；"
        "（3）符号若可在表内或上下文中确定含义，用自然语言写出含义，避免无意义堆砌符号；"
        "（4）涉及必选/可选/适用/不适用、限值等等级时，措辞须与表意一致，避免「均设置」「均包含」等模糊概括；"
        "（5）矩阵表、参数表、清单表、对比表须覆盖主要项目，忌空泛一句带过；"
        "（6）输出为一段中文，无小标题、无项目符号、无代码块；"
        f"（7）总长度不超过约 {max_chars} 字（以表意完整为先，略超可接受时优先保真）。"
        f"{INTERNAL_TABLE_CAPTION_REASONING}"
    )

    user_text = (
        "【表格附近上下文】\n"
        f"{context_text}\n\n"
        "【表格 Markdown】\n"
        f"{table_markdown}\n\n"
        "请将上表转述为一段便于阅读的中文，突出项目与对应关系；不做表外评价或用途阐释。"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_text},
    ]


def _marker_prefix(block: TableBlock) -> str:
    return f'<!-- table-semantic id="t-{block.start}-{block.kind}"'


def _table_raw_exceeds_limit(block: TableBlock, max_table_chars: int) -> bool:
    return max_table_chars > 0 and len(block.raw) > max_table_chars


def _already_augmented(text: str, block: TableBlock) -> bool:
    window = text[block.end : block.end + 1200]
    return _marker_prefix(block) in window


def _slice_context(
    full: str,
    *,
    start: int,
    end: int,
    before_chars: int | None,
    after_chars: int | None,
) -> tuple[str, str]:
    prefix = full[:start]
    suffix = full[end:]
    if before_chars is not None and before_chars >= 0:
        prefix = prefix[-before_chars:]
    if after_chars is not None and after_chars >= 0:
        suffix = suffix[:after_chars]
    return prefix, suffix


def _build_context_text(prefix: str, suffix: str) -> str:
    a, b = prefix.rstrip(), suffix.lstrip()
    if a and b:
        return f"{a}\n\n{b}"
    if a:
        return a
    if b:
        return b
    return "（无：未截取到表格前后文）"


_FENCE_WRAP = re.compile(
    r"^```[a-zA-Z0-9_-]*\s*\r?\n(?P<body>[\s\S]*?)\r?\n```\s*$",
)


def _normalize_caption_output(raw: str) -> str:
    s = raw.strip()
    if not s:
        return ""
    m = _FENCE_WRAP.match(s)
    if m:
        s = m.group("body").strip()
    return s


def _log_preview(text: str, *, max_chars: int = 100) -> str:
    """单行、限长预览，避免日志里换行或过长正文导致显示截断或乱码。"""
    collapsed = " ".join(text.split())
    if len(collapsed) <= max_chars:
        return collapsed
    return collapsed[: max_chars - 1] + "…"


def _not_reported(value: int | None) -> str | int:
    return "not_reported" if value is None else value


def _log_llm_table_outcome(
    *,
    table_index: int,
    table_total: int,
    model: str,
    elapsed_seconds: float,
    usage: ChatCompletionMeta,
    local_prompt_character_count: int,
    local_completion_response_character_count: int,
    local_caption_character_count: int | None,
    result_status: str,
) -> None:
    """One line per model call: full English keys, tokens first, then character counts (Python len)."""
    caption_out = (
        local_caption_character_count
        if local_caption_character_count is not None
        else "not_applicable"
    )
    log.info(
        "table_semantic large_language_model_round_complete "
        "table_index=%s table_count=%s model_name=%s elapsed_seconds=%s "
        "usage_prompt_token_count=%s usage_completion_token_count=%s usage_total_token_count=%s "
        "local_prompt_character_count=%s local_completion_response_character_count=%s "
        "local_caption_character_count=%s "
        "usage_reported_prompt_character_count=%s usage_reported_completion_character_count=%s "
        "result_status=%s",
        table_index,
        table_total,
        model,
        elapsed_seconds,
        _not_reported(usage.prompt_tokens),
        _not_reported(usage.completion_tokens),
        _not_reported(usage.total_tokens),
        local_prompt_character_count,
        local_completion_response_character_count,
        caption_out,
        _not_reported(usage.prompt_chars),
        _not_reported(usage.completion_chars),
        result_status,
    )


def _llm_one_block(
    *,
    full_text: str,
    block: TableBlock,
    cfg: OpenAICompatibleConfig,
    caption_params: TableCaptionParams,
    table_index: int,
    table_total: int,
) -> tuple[int, str] | None:
    prefix, suffix = _slice_context(
        full_text,
        start=block.start,
        end=block.end,
        before_chars=caption_params.context_before_chars,
        after_chars=caption_params.context_after_chars,
    )
    context_text = _build_context_text(prefix, suffix)
    messages = build_table_caption_messages(
        table_markdown=block.raw,
        context_text=context_text,
        max_chars=caption_params.caption_target_chars,
    )
    sys_len = len(messages[0]["content"])
    usr_len = len(messages[1]["content"])
    ctx_len = len(context_text)
    tbl_len = len(block.raw)

    t0 = time.perf_counter()
    raw_content, usage_meta = chat_completion_text_with_meta(cfg=cfg, messages=messages)
    elapsed_sec = round(time.perf_counter() - t0, 3)
    completion_chars_raw = len(raw_content) if isinstance(raw_content, str) else 0

    summary = _normalize_caption_output(raw_content)
    caption_nchars = len(summary) if summary else None
    status = "ok" if summary else "empty"
    prompt_nchars = sys_len + usr_len
    _log_llm_table_outcome(
        table_index=table_index,
        table_total=table_total,
        model=cfg.model,
        elapsed_seconds=elapsed_sec,
        usage=usage_meta,
        local_prompt_character_count=prompt_nchars,
        local_completion_response_character_count=completion_chars_raw,
        local_caption_character_count=caption_nchars,
        result_status=status,
    )
    log.debug(
        "table_semantic large_language_model_message_character_breakdown "
        "table_index=%s table_count=%s "
        "context_section_character_count=%s table_body_character_count=%s "
        "system_message_character_count=%s user_message_character_count=%s",
        table_index,
        table_total,
        ctx_len,
        tbl_len,
        sys_len,
        usr_len,
    )

    if not summary:
        log.warning(
            "table_semantic large_language_model_empty_response "
            "table_index=%s table_count=%s response_preview=%s",
            table_index,
            table_total,
            _log_preview(raw_content or "", max_chars=120),
        )
        return None

    mid = _marker_prefix(block)
    insert = (
        f"\n\n{mid} -->\n"
        f"**表格说明**：{summary}\n"
        f"<!-- /table-semantic -->\n"
    )

    log.debug(
        "table_semantic caption_text_preview table_index=%s table_count=%s preview=%s",
        table_index,
        table_total,
        _log_preview(summary, max_chars=200),
    )

    return block.end, insert


def augment_markdown_text(
    text: str,
    *,
    cfg: OpenAICompatibleConfig,
    max_concurrency: int = 4,
    source: str = "",
    caption_params: TableCaptionParams | None = None,
    progress_callback: Callable[[int, int, float | None], None] | None = None,
) -> str:
    """返回增强后的全文；失败块跳过（抛出由调用方策略处理时可在外层捕获）。"""
    t_all = time.perf_counter()
    cap = caption_params or DEFAULT_TABLE_CAPTION_PARAMS
    blocks = list(iter_table_blocks(text))
    pending_need_llm = [b for b in blocks if not _already_augmented(text, b)]
    skipped_already = len(blocks) - len(pending_need_llm)
    oversized = [
        b for b in pending_need_llm if _table_raw_exceeds_limit(b, cap.max_table_chars)
    ]
    pending = [
        b
        for b in pending_need_llm
        if not _table_raw_exceeds_limit(b, cap.max_table_chars)
    ]
    skipped_oversize = len(oversized)
    workers = max(1, int(max_concurrency))
    pool_workers = min(workers, len(pending)) if pending else 0

    file_hint = Path(source).name if source.strip() else "-"
    log.info(
        "table_semantic batch_started "
        "source_file_name=%s table_count=%s tables_pending_model_calls=%s "
        "tables_skipped_already_augmented=%s tables_skipped_over_size_limit=%s "
        "model_name=%s worker_thread_count=%s",
        file_hint,
        len(blocks),
        len(pending),
        skipped_already,
        skipped_oversize,
        cfg.model,
        workers,
    )
    log.debug(
        "table_semantic configuration_limits "
        "context_before_character_limit=%s context_after_character_limit=%s "
        "caption_target_character_count=%s maximum_table_body_character_count=%s "
        "document_character_count=%s",
        cap.context_before_chars,
        cap.context_after_chars,
        cap.caption_target_chars,
        cap.max_table_chars,
        len(text),
    )
    if oversized:
        log.info(
            "table_semantic tables_skipped_over_size_limit "
            "skipped_table_count=%s maximum_table_body_character_count=%s",
            skipped_oversize,
            cap.max_table_chars,
        )
        for b in oversized:
            log.debug(
                "table_semantic table_skipped_over_size_limit_detail "
                "block_kind=%s table_body_character_count=%s document_character_offset=%s",
                b.kind,
                len(b.raw),
                b.start,
            )
    if progress_callback is not None:
        try:
            progress_callback(0, len(pending), None)
        except Exception:
            pass

    if not pending:
        if pending_need_llm and skipped_oversize == len(pending_need_llm):
            log.info(
                "table_semantic batch_skipped_all_tables_over_size_limit "
                "tables_that_would_need_model_calls=%s maximum_table_body_character_count=%s",
                len(pending_need_llm),
                cap.max_table_chars,
            )
        else:
            log.info("table_semantic batch_no_tables_required_processing")
        return text

    inserts: dict[int, str] = {}

    pending_by_id = {id(b): idx for idx, b in enumerate(pending, start=1)}

    def job(b: TableBlock) -> tuple[int, str] | None:
        idx = int(pending_by_id.get(id(b), 0))
        try:
            return _llm_one_block(
                full_text=text,
                block=b,
                cfg=cfg,
                caption_params=cap,
                table_index=idx,
                table_total=len(pending),
            )
        except (LLMClientError, OSError, ValueError, TypeError, RuntimeError) as exc:
            log.warning(
                "table_semantic large_language_model_call_skipped_due_to_exception "
                "table_index=%s table_count=%s exception_type=%s exception_message=%s",
                idx,
                len(pending),
                type(exc).__name__,
                _log_preview(str(exc), max_chars=160),
            )
            return None

    done_count = 0
    with ThreadPoolExecutor(max_workers=pool_workers) as pool:
        futures = [pool.submit(job, b) for b in pending]
        for fut in as_completed(futures):
            try:
                got = fut.result()
            except (LLMClientError, OSError, ValueError, TypeError, RuntimeError) as exc:
                log.warning(
                    "table_semantic thread_pool_worker_exception_ignored "
                    "exception_type=%s exception_message=%s",
                    type(exc).__name__,
                    _log_preview(str(exc), max_chars=160),
                )
                continue
            if got is None:
                done_count += 1
                if progress_callback is not None:
                    elapsed = time.perf_counter() - t_all
                    avg = elapsed / max(done_count, 1)
                    remain = max(0.0, avg * max(len(pending) - done_count, 0))
                    try:
                        progress_callback(done_count, len(pending), round(remain, 3))
                    except Exception:
                        pass
                continue
            end_pos, snippet = got
            inserts[end_pos] = snippet
            done_count += 1
            if progress_callback is not None:
                elapsed = time.perf_counter() - t_all
                avg = elapsed / max(done_count, 1)
                remain = max(0.0, avg * max(len(pending) - done_count, 0))
                try:
                    progress_callback(done_count, len(pending), round(remain, 3))
                except Exception:
                    pass

    elapsed_sec = round(time.perf_counter() - t_all, 3)
    if not inserts:
        log.info(
            "table_semantic batch_finished inserted_table_count=0 "
            "tables_sent_to_model=%s elapsed_seconds=%s",
            len(pending),
            elapsed_sec,
        )
        return text

    out = text
    for end_pos in sorted(inserts.keys(), reverse=True):
        out = out[:end_pos] + inserts[end_pos] + out[end_pos:]

    log.info(
        "table_semantic batch_finished inserted_table_count=%s tables_sent_to_model=%s elapsed_seconds=%s",
        len(inserts),
        len(pending),
        elapsed_sec,
    )
    log.debug(
        "table_semantic batch_document_character_count_before=%s after=%s",
        len(text),
        len(out),
    )
    return out


_SEMANTIC_BLOCK = re.compile(
    r"\n\n<!--\s*table-semantic\b[^>]*-->.*?<!--\s*/table-semantic\s*-->\n",
    re.DOTALL,
)


def strip_table_semantic_blocks(text: str) -> str:
    """移除本模块插入的语义块（用于测试或重跑前清理）。"""
    return _SEMANTIC_BLOCK.sub("\n", text)


def augment_markdown_file(
    path: Path,
    *,
    cfg: OpenAICompatibleConfig,
    max_concurrency: int = 4,
    caption_params: TableCaptionParams | None = None,
    progress_callback: Callable[[int, int, float | None], None] | None = None,
) -> None:
    """就地原子更新 ``path`` 指向的 UTF-8 Markdown。"""
    src = str(path.resolve())
    t0 = time.perf_counter()
    raw = path.read_text(encoding="utf-8")
    new_text = augment_markdown_text(
        raw,
        cfg=cfg,
        max_concurrency=max_concurrency,
        source=src,
        caption_params=caption_params,
        progress_callback=progress_callback,
    )
    if new_text == raw:
        log.info(
            "table_semantic file_write_skipped_no_content_change file_name=%s",
            path.name,
        )
        return
    fd, tmp = tempfile.mkstemp(
        suffix=".md", prefix="table_sem_", dir=str(path.parent)
    )
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        tmp_path.write_text(new_text, encoding="utf-8")
        tmp_path.replace(path)
        log.info(
            "table_semantic file_write_completed file_name=%s elapsed_seconds=%s",
            path.name,
            round(time.perf_counter() - t0, 3),
        )
    except OSError:
        tmp_path.unlink(missing_ok=True)
        raise
