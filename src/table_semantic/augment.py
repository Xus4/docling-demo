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
    "你是RAG预处理模块，任务是将Markdown表格（html格式的）进行语义等价转写。\n"

    "【输出约束】\n"
    "1）只输出一个JSON对象，不得包含任何额外内容。\n"
    "2）JSON必须合法可解析。\n"

    "【任务要求】\n"
    "1）按行展开表格，每一行转写为一个完整中文句子。\n"
    "2）每个句子必须包含该行所有列的信息，并明确“列名-值”关系。\n"
    "3）不得遗漏字段，不得跨行合并。\n"
    "4）禁止总结、归纳或改写原意。\n"
    "5）禁止编造信息；空值需说明为“为空”或“未提供”。\n"

    "【输出格式】\n"
    "{\n"
    "  \"equivalent_text\": string\n"
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
) -> str:
    return json.dumps(
        {
            "task": "table_semantic_for_rag",
            "requirement": "请你详细描述表格内容，让我不看表格也能知道表格内是什么信息",
            "table_kind": block.kind,
            "table_markdown_or_html": block.raw,
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
    run_id: str,
    table_index: int,
    table_total: int,
) -> tuple[int, str] | None:
    user_content = _build_user_payload(block=block)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    t0 = time.perf_counter()
    
    # 计算总输入大小
    full_input = json.dumps(messages, ensure_ascii=False)
    
    # 打印输入信息 - 真正的单行，用 | 分隔
    log.info(f"[INPUT] [表格{table_index}/{table_total}] 表格:{len(block.raw)}c | System:{len(_SYSTEM_PROMPT)}c | User:{len(user_content)}c | 总:{len(full_input)}c")
    
    prompt_chars_estimated = _calc_prompt_chars(messages)
    data, usage_meta = chat_completion_json_object_with_meta(cfg=cfg, messages=messages)
    elapsed_sec = round(time.perf_counter() - t0, 3)
    usage_fields = _usage_log_fields(usage_meta, prompt_chars_estimated)
    summary = _extract_equivalent_text(data)
    
    if not summary:
        log.warning(f"[ERROR] [表格{table_index}/{table_total}] 空结果 | {elapsed_sec}s | 返回:{str(data)[:200]}")
        return None
    
    mid = _marker_prefix(block)
    insert = (
        f"\n\n{mid} -->\n"
        f"**表格说明**：{summary}\n"
        f"<!-- /table-semantic -->\n"
    )
    
    # 打印输出信息 - 真正的单行
    log.info(f"[OUTPUT] [表格{table_index}/{table_total}] {elapsed_sec}s | 输入:{len(full_input)}c | 输出:{len(summary)}c | Tokens:{usage_meta.prompt_tokens or '-'}/{usage_meta.completion_tokens or '-'}/{usage_meta.total_tokens or '-'} | {summary[:150]}...")
    
    return block.end, insert


def augment_markdown_text(
    text: str,
    *,
    cfg: OpenAICompatibleConfig,
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

    log.info(
        f"========== 开始表格语义增强 ==========\n"
        f"文档大小: {len(text)}字符\n"
        f"表格总数: {len(blocks)} | 待处理: {len(pending)} | 已跳过: {skipped}\n"
        f"并发数: {workers}\n"
        f"模型: {cfg.model}"
    )
    if progress_callback is not None:
        try:
            progress_callback(0, len(pending), None)
        except Exception:
            pass

    if not pending:
        log.info("无需增强，文档中没有待处理表格")
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
                run_id=run_id,
                table_index=idx,
                table_total=len(pending),
            )
        except (LLMClientError, OSError, ValueError, TypeError, RuntimeError) as exc:
            log.warning(
                f"[表格{idx}/{len(pending)}] 处理失败，已跳过 | "
                f"错误: {type(exc).__name__}: {str(exc)[:200]}"
            )
            return None

    done_count = 0
    with ThreadPoolExecutor(max_workers=pool_workers) as pool:
        futures = [pool.submit(job, b) for b in pending]
        for fut in as_completed(futures):
            try:
                got = fut.result()
            except (LLMClientError, OSError, ValueError, TypeError, RuntimeError) as exc:
                log.warning(f"并发任务异常，已忽略 | 错误: {type(exc).__name__}: {str(exc)[:200]}")
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
        log.info(f"未生成可插入的语义内容 | 耗时: {elapsed_sec}秒")
        return text

    out = text
    for end_pos in sorted(inserts.keys(), reverse=True):
        out = out[:end_pos] + inserts[end_pos] + out[end_pos:]
    
    log.info(
        f"========== 表格语义增强完成 ==========\n"
        f"处理表格: {len(inserts)}/{len(pending)} | "
        f"文档变化: {len(text)} -> {len(out)}字符 | "
        f"总耗时: {elapsed_sec}秒"
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
    progress_callback: Callable[[int, int, float | None], None] | None = None,
) -> None:
    """就地原子更新 ``path`` 指向的 UTF-8 Markdown。"""
    src = str(path.resolve())
    t0 = time.perf_counter()
    raw = path.read_text(encoding="utf-8")
    log.info(f"读取文件: {path.name} | 大小: {len(raw)}字符")
    new_text = augment_markdown_text(
        raw,
        cfg=cfg,
        max_concurrency=max_concurrency,
        source=src,
        progress_callback=progress_callback,
    )
    if new_text == raw:
        log.info("文件无需改动，未写回")
        return
    fd, tmp = tempfile.mkstemp(
        suffix=".md", prefix="table_sem_", dir=str(path.parent)
    )
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        tmp_path.write_text(new_text, encoding="utf-8")
        tmp_path.replace(path)
        log.info(f"文件已更新: {path.name} | 新大小: {len(new_text)}字符 | 耗时: {round(time.perf_counter() - t0, 3)}秒")
    except OSError:
        tmp_path.unlink(missing_ok=True)
        raise
