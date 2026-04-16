"""在 Markdown 中为表格插入大模型生成的语义块（幂等）。"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from src.logging_utils import log_event
from src.table_semantic.llm_client import (
    ChatCompletionMeta,
    LLMClientError,
    OpenAICompatibleConfig,
    chat_completion_json_object_with_meta,
)
from src.table_semantic.table_blocks import TableBlock, iter_table_blocks

log = logging.getLogger(__name__)


def _calc_prompt_chars(messages: list[dict[str, str]]) -> int:
    total = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            total += len(content)
    return total


def _usage_log_fields(meta: ChatCompletionMeta, prompt_chars_estimated: int) -> dict[str, int]:
    fields: dict[str, int] = {"prompt_chars": prompt_chars_estimated}
    if meta.prompt_tokens is not None:
        fields["prompt_tokens"] = meta.prompt_tokens
    if meta.completion_tokens is not None:
        fields["completion_tokens"] = meta.completion_tokens
    if meta.total_tokens is not None:
        fields["total_tokens"] = meta.total_tokens
    if meta.prompt_chars is not None:
        fields["prompt_chars"] = meta.prompt_chars
    if meta.completion_chars is not None:
        fields["completion_chars"] = meta.completion_chars
    return fields

_SYSTEM_PROMPT = (
    "你是文档结构化助手，输出用于 RAG 检索的「表格信息等价转写」。\n"
    "【最高优先级输出规则】\n"
    "1) 你的响应必须是且只能是一个 JSON 对象。\n"
    "2) 不得输出任何 JSON 之外的字符（包括解释、前言、结语、Markdown、```代码块```、注释）。\n"
    "3) 不得使用多段响应，不得附加“说明如下”等文字。\n"
    "4) 若无法完成，也必须返回 JSON 对象，不得返回自然语言段落。\n"
    "【内容规则】\n"
    "1) 只能基于「表格原文」与可选「前后文」推断；前后文与表格可能弱相关，请勿强行关联。\n"
    "2) 禁止编造表格中未出现的数值、实体或结论。\n"
    "3) 输出必须与表格信息等价，不得只做概述，不得遗漏关键字段、条件、单位、时间、范围、阈值、合计/分项、注释与脚注。\n"
    "4) 原表未提供的信息可写“原表未说明”，但不能脑补。\n"
    "【JSON Schema（语义要求）】\n"
    "{\n"
    "  \"equivalent_text\": string,                // 必填：完整等价转写（中文）\n"
    "  \"key_facts\": string[],                    // 可选：按行列关系拆分事实点\n"
    "  \"entities\": string[]                      // 可选：实体词\n"
    "}"
)


def _marker_prefix(block: TableBlock) -> str:
    return f'<!-- table-semantic id="t-{block.start}-{block.kind}"'


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


def _build_user_payload(
    *,
    block: TableBlock,
    context_before: str,
    context_after: str,
) -> str:
    return json.dumps(
        {
            "task": "table_semantic_for_rag",
            "requirement": "请输出与表格信息等价的完整语义文本，不要摘要化",
            "table_kind": block.kind,
            "table_markdown_or_html": block.raw,
            "context_before": context_before,
            "context_after": context_after,
        },
        ensure_ascii=False,
    )


def _extract_equivalent_text(data: dict[str, Any]) -> str:
    for key in ("equivalent_text", "summary", "semantic_text"):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    facts = data.get("key_facts")
    if isinstance(facts, list):
        rows = [str(x).strip() for x in facts if str(x).strip()]
        if rows:
            return "；".join(rows)
    return ""


def _llm_one_block(
    *,
    full_text: str,
    block: TableBlock,
    cfg: OpenAICompatibleConfig,
    context_before_chars: int | None,
    context_after_chars: int | None,
    run_id: str,
    table_index: int,
    table_total: int,
) -> tuple[int, str] | None:
    ctx_b, ctx_a = _slice_context(
        full_text,
        start=block.start,
        end=block.end,
        before_chars=context_before_chars,
        after_chars=context_after_chars,
    )
    user_content = _build_user_payload(
        block=block, context_before=ctx_b, context_after=ctx_a
    )
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    t0 = time.perf_counter()
    log_event(
        log,
        logging.INFO,
        "table_semantic.llm.request",
        zh="开始调用大模型分析表格",
        run_id=run_id,
        table_index=table_index,
        table_total=table_total,
        kind=block.kind,
        span_start=block.start,
        span_end=block.end,
        table_chars=len(block.raw),
        ctx_before_chars=len(ctx_b),
        ctx_after_chars=len(ctx_a),
        user_payload_chars=len(user_content),
        model=cfg.model,
        thinking_enable=cfg.thinking_enable,
    )
    prompt_chars_estimated = _calc_prompt_chars(messages)
    data, usage_meta = chat_completion_json_object_with_meta(cfg=cfg, messages=messages)
    elapsed_sec = round(time.perf_counter() - t0, 3)
    usage_fields = _usage_log_fields(usage_meta, prompt_chars_estimated)
    summary = _extract_equivalent_text(data)
    if not summary:
        log_event(
            log,
            logging.INFO,
            "table_semantic.llm.empty_summary",
            zh="大模型返回为空摘要，跳过该表格",
            run_id=run_id,
            table_index=table_index,
            table_total=table_total,
            kind=block.kind,
            span_start=block.start,
            span_end=block.end,
            elapsed_sec=elapsed_sec,
            model_elapsed_sec=elapsed_sec,
            **usage_fields,
            keys_sample=",".join(sorted(data.keys()))[:200] if isinstance(data, dict) else "",
        )
        return None
    mid = _marker_prefix(block)
    insert = (
        f"\n\n{mid} -->\n"
        f"**表格说明**：{summary}\n"
        f"<!-- /table-semantic -->\n"
    )
    log_event(
        log,
        logging.INFO,
        "table_semantic.llm.ok",
        zh="大模型分析表格完成",
        run_id=run_id,
        table_index=table_index,
        table_total=table_total,
        kind=block.kind,
        span_start=block.start,
        span_end=block.end,
        summary_chars=len(summary),
        insert_chars=len(insert),
        elapsed_sec=elapsed_sec,
        model_elapsed_sec=elapsed_sec,
        **usage_fields,
    )
    return block.end, insert


def augment_markdown_text(
    text: str,
    *,
    cfg: OpenAICompatibleConfig,
    context_before_chars: int | None = None,
    context_after_chars: int | None = None,
    max_concurrency: int = 4,
    source: str = "",
    progress_callback: Callable[[int, int, float | None], None] | None = None,
) -> str:
    """返回增强后的全文；失败块跳过（抛出由调用方策略处理时可在外层捕获）。"""
    t_all = time.perf_counter()
    run_id = uuid.uuid4().hex[:8]
    blocks = list(iter_table_blocks(text))
    pending = [b for b in blocks if not _already_augmented(text, b)]
    skipped = len(blocks) - len(pending)
    workers = max(1, int(max_concurrency))
    pool_workers = min(workers, len(pending)) if pending else 0

    log_event(
        log,
        logging.INFO,
        "table_semantic.text.plan",
        zh="已生成表格语义增强处理计划",
        run_id=run_id,
        source=source or "-",
        doc_chars=len(text),
        tables_total=len(blocks),
        tables_pending=len(pending),
        tables_skipped_augmented=skipped,
        max_concurrency=workers,
        pool_workers=pool_workers,
        context_before_chars=context_before_chars,
        context_after_chars=context_after_chars,
        model=cfg.model,
        thinking_enable=cfg.thinking_enable,
    )
    if progress_callback is not None:
        try:
            progress_callback(0, len(pending), None)
        except Exception:
            pass

    if not pending:
        log_event(
            log,
            logging.INFO,
            "table_semantic.text.skip",
            zh="无需增强，文档中没有待处理表格",
            run_id=run_id,
            source=source or "-",
            reason="no_pending_tables" if blocks else "no_tables_found",
            tables_total=len(blocks),
            elapsed_sec=round(time.perf_counter() - t_all, 3),
        )
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
                context_before_chars=context_before_chars,
                context_after_chars=context_after_chars,
                run_id=run_id,
                table_index=idx,
                table_total=len(pending),
            )
        except (LLMClientError, OSError, ValueError, TypeError, RuntimeError) as exc:
            log_event(
                log,
                logging.WARNING,
                "table_semantic.block.error",
                zh="表格语义增强失败，已跳过该表格",
                run_id=run_id,
                table_index=idx,
                table_total=len(pending),
                source=source or "-",
                kind=b.kind,
                span_start=b.start,
                span_end=b.end,
                table_chars=len(b.raw),
                err_type=type(exc).__name__,
                err=str(exc)[:1200],
            )
            return None

    done_count = 0
    with ThreadPoolExecutor(max_workers=pool_workers) as pool:
        futures = [pool.submit(job, b) for b in pending]
        for fut in as_completed(futures):
            try:
                got = fut.result()
            except (LLMClientError, OSError, ValueError, TypeError, RuntimeError) as exc:
                log_event(
                    log,
                    logging.WARNING,
                    "table_semantic.future.error",
                    zh="并发任务异常，已忽略该结果",
                    run_id=run_id,
                    source=source or "-",
                    err_type=type(exc).__name__,
                    err=str(exc)[:1200],
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
        log_event(
            log,
            logging.INFO,
            "table_semantic.text.no_inserts",
            zh="未生成可插入的语义内容",
            run_id=run_id,
            source=source or "-",
            tables_pending=len(pending),
            elapsed_sec=elapsed_sec,
        )
        return text

    out = text
    for end_pos in sorted(inserts.keys(), reverse=True):
        out = out[:end_pos] + inserts[end_pos] + out[end_pos:]
    log_event(
        log,
        logging.INFO,
        "table_semantic.text.merged",
        zh="语义内容已合并回文档",
        run_id=run_id,
        source=source or "-",
        inserts=len(inserts),
        doc_chars_before=len(text),
        doc_chars_after=len(out),
        elapsed_sec=elapsed_sec,
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
    context_before_chars: int | None = None,
    context_after_chars: int | None = None,
    max_concurrency: int = 4,
    progress_callback: Callable[[int, int, float | None], None] | None = None,
) -> None:
    """就地原子更新 ``path`` 指向的 UTF-8 Markdown。"""
    src = str(path.resolve())
    t0 = time.perf_counter()
    raw = path.read_text(encoding="utf-8")
    log_event(
        log,
        logging.INFO,
        "table_semantic.file.begin",
        zh="开始读取并增强 Markdown 文件",
        path=src,
        bytes=len(raw.encode("utf-8")),
        chars=len(raw),
    )
    new_text = augment_markdown_text(
        raw,
        cfg=cfg,
        context_before_chars=context_before_chars,
        context_after_chars=context_after_chars,
        max_concurrency=max_concurrency,
        source=src,
        progress_callback=progress_callback,
    )
    if new_text == raw:
        log_event(
            log,
            logging.INFO,
            "table_semantic.file.unchanged",
            zh="文件无需改动，未写回",
            path=src,
            elapsed_sec=round(time.perf_counter() - t0, 3),
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
        log_event(
            log,
            logging.INFO,
            "table_semantic.file.wrote",
            zh="增强后的 Markdown 已写入",
            path=src,
            chars_after=len(new_text),
            elapsed_sec=round(time.perf_counter() - t0, 3),
        )
    except OSError:
        tmp_path.unlink(missing_ok=True)
        raise
