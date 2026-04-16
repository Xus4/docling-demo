"""在 Markdown 中为表格插入大模型生成的语义块（幂等）。"""

from __future__ import annotations

import json
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


def _log_llm_call_detail(
    *,
    table_index: int,
    table_total: int,
    elapsed_sec: float,
    usage: ChatCompletionMeta,
    prompt_chars: int,
    completion_chars: int,
) -> None:
    """单独一行：大模型调用的 token 与耗时（及请求/响应字数，便于对账）。"""
    pt = usage.prompt_tokens
    ct = usage.completion_tokens
    tt = usage.total_tokens
    log.info(
        "表格语义 | LLM调用 | 表 %s/%s | 耗时=%ss | 输入token=%s 输出token=%s 合计token=%s | "
        "请求约%s字 响应约%s字",
        table_index,
        table_total,
        elapsed_sec,
        pt if pt is not None else "-",
        ct if ct is not None else "-",
        tt if tt is not None else "-",
        prompt_chars,
        completion_chars,
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
    t0 = time.perf_counter()

    full_input = json.dumps(messages, ensure_ascii=False)
    sys_len = len(messages[0]["content"])
    usr_len = len(messages[1]["content"])
    log.debug(
        "表格语义 | 请求明细 | 表 %s/%s | 表体=%s字 上下文=%s字 | system=%s user=%s 合计=%s",
        table_index,
        table_total,
        len(block.raw),
        len(context_text),
        sys_len,
        usr_len,
        len(full_input),
    )

    raw_content, usage_meta = chat_completion_text_with_meta(cfg=cfg, messages=messages)
    elapsed_sec = round(time.perf_counter() - t0, 3)
    prompt_char_count = sys_len + usr_len
    completion_chars_raw = len(raw_content) if isinstance(raw_content, str) else 0
    _log_llm_call_detail(
        table_index=table_index,
        table_total=table_total,
        elapsed_sec=elapsed_sec,
        usage=usage_meta,
        prompt_chars=prompt_char_count,
        completion_chars=completion_chars_raw,
    )

    summary = _normalize_caption_output(raw_content)

    if not summary:
        log.warning(
            "表格语义 | 表 %s/%s | 失败 | 模型返回空或仅空白 | 片段=%s",
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

    log.info(
        "表格语义 | 表 %s/%s | 完成 | 说明=%s字 | %s",
        table_index,
        table_total,
        len(summary),
        _log_preview(summary, max_chars=100),
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
    pending = [b for b in blocks if not _already_augmented(text, b)]
    skipped = len(blocks) - len(pending)
    workers = max(1, int(max_concurrency))
    pool_workers = min(workers, len(pending)) if pending else 0

    file_hint = Path(source).name if source.strip() else "—"
    log.info(
        "表格语义 | 开始 | 文件=%s | 文档=%s字 | 表共%s 待处理%s 已跳过%s | 模型=%s | 并发=%s | "
        "上下文前/后=%s/%s字 目标约%s字",
        file_hint,
        len(text),
        len(blocks),
        len(pending),
        skipped,
        cfg.model,
        workers,
        cap.context_before_chars,
        cap.context_after_chars,
        cap.caption_target_chars,
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
                caption_params=cap,
                table_index=idx,
                table_total=len(pending),
            )
        except (LLMClientError, OSError, ValueError, TypeError, RuntimeError) as exc:
            log.warning(
                "表格语义 | 表 %s/%s | 跳过 | %s: %s",
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
                    "表格语义 | 并发任务异常已忽略 | %s: %s",
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
            "表格语义 | 结束 | 无插入 | 待处理=%s | 总耗时=%ss",
            len(pending),
            elapsed_sec,
        )
        return text

    out = text
    for end_pos in sorted(inserts.keys(), reverse=True):
        out = out[:end_pos] + inserts[end_pos] + out[end_pos:]

    log.info(
        "表格语义 | 结束 | 成功=%s/%s | 文档 %s→%s字 | 总耗时=%ss",
        len(inserts),
        len(pending),
        len(text),
        len(out),
        elapsed_sec,
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
        log.info("表格语义 | 写回 | 跳过 | 无变化 | %s", path.name)
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
            "表格语义 | 写回 | 已保存 | %s | %s字 | 耗时=%ss",
            path.name,
            len(new_text),
            round(time.perf_counter() - t0, 3),
        )
    except OSError:
        tmp_path.unlink(missing_ok=True)
        raise
