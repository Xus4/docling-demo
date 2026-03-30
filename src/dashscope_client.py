from __future__ import annotations

import base64
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import httpx

_log = logging.getLogger(__name__)

# OpenAI 兼容模式本地图片 → data URL（与百炼文档一致）
_EXT_TO_MIME: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}


def _is_openai_compatible_base_url(base_url: str) -> bool:
    """
    是否对多模态请求走 OpenAI 兼容 ``/chat/completions``（而非百炼原生 multimodal 专线路径）。

    - 百炼 OpenAI 兼容：URL 含 ``compatible-mode``。
    - 百炼原生 DashScope：``…/api/v1`` → 走 ``/services/aigc/multimodal-generation/…``。
    - Ollama / vLLM 等：基址常为 ``…/v1``，与 OpenAI 官方一致，应走 ``/v1/chat/completions``。
    """
    u = (base_url or "").lower().rstrip("/")
    if not u:
        return False
    if "compatible-mode" in u:
        return True
    # 百炼原生：…/api/v1 必须用专用接口，不能拼 /chat/completions
    if "dashscope" in u and u.endswith("/api/v1"):
        return False
    # 常见 OpenAI 兼容服务（含 Ollama 默认 http://localhost:11434/v1）
    return u.endswith("/v1")


def _is_likely_ollama_openai_base_url(base_url: str) -> bool:
    """
    判断是否像 Ollama 的 OpenAI 兼容入口（默认端口 11434）。

    Ollama 不识别百炼请求体里的 ``enable_thinking`` / ``max_reasoning_tokens``；
    Qwen3.5 视觉在部分版本会把输出放进 ``message.thinking`` 而 ``content`` 为空，
    需在 JSON 里使用 ``reasoning_effort``（或原生 API 的 ``think``）关闭思考。
    """
    u = (base_url or "").lower().rstrip("/")
    if "dashscope" in u or "compatible-mode" in u:
        return False
    if not u.endswith("/v1"):
        return False
    return "11434" in u


@dataclass(frozen=True)
class DashScopeClientConfig:
    api_key: str
    # 原生：…/api/v1；OpenAI 兼容：…/compatible-mode/v1（见百炼文档）
    base_url: str
    timeout_sec: float = 300.0
    max_retries: int = 3
    retry_backoff_sec: float = 1.5
    # Qwen3.5 / Qwen3-VL：思考走 reasoning_content；百炼用 extra_body.enable_thinking。
    # 默认开启以保障质量；关闭时可省 completion token。
    enable_thinking: bool = True
    # 仅 enable_thinking=True 时生效；关闭思考时请求里会强制 max_reasoning_tokens=0。
    max_reasoning_tokens: Optional[int] = 256
    # OpenAI 兼容：assistant.content 为空（含仅 reasoning_content 有字）时额外重试次数（含首次共 N 次请求）
    empty_content_max_attempts: int = 3
    # True：OpenAI 兼容 /chat/completions 使用 stream，正文与思考链增量均写入 INFO 日志（见 LLM_LOG_STREAM_RESPONSE）
    log_stream_response: bool = False


def _httpx_timeout(timeout_sec: float) -> httpx.Timeout:
    """
    长耗时 LLM 请求（含流式）：read 需覆盖首包延迟（大图预填 / 慢速 27B）与包间隔。
    connect 单独限制，避免把「等首 token」误算成连接阶段；write/pool 与 read 对齐以免流式挂起。
    """
    t = max(30.0, float(timeout_sec))
    c = min(120.0, t)
    return httpx.Timeout(connect=c, read=t, write=t, pool=t)


def _flush_logging_handlers() -> None:
    """尽量立刻刷出日志，便于流式片段实时可见。"""
    for lg in (logging.root, _log):
        for h in getattr(lg, "handlers", []) or []:
            try:
                h.flush()
            except Exception:
                pass
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass


class DashScopeClient:
    _FINAL_OUTPUT_GUARD = (
        "\n\n输出要求（必须遵守）："
        "\n- 最终可用的 Markdown/正文必须写在 assistant 的 content 中，且为非空字符串。"
        "\n- 禁止把可交付正文只写在推理/思考字段（reasoning）；推理字段不能替代 content。"
        "\n- 不要输出「步骤 1」「分析表格结构」等过程说明，直接输出结果。"
        "\n- 不要只输出思考过程；不要解释你如何思考。"
    )

    _RETRY_FORCE_FINAL = (
        "\n\n【重试指令】上一次违反协议：content 为空但推理字段有内容，或正文未写入 content。"
        "\n本次回复必须使 content 为非空最终 Markdown；禁止再用推理字段承载正文；"
        "禁止输出思考过程。"
    )

    """
    Minimal DashScope HTTP client for Qwen 系列（文本/多模态生成）。

    约定响应提取逻辑优先读取：
      output.choices[0].message.content[0].text
    并在结构变化时做降级处理。
    """

    def __init__(self, cfg: DashScopeClientConfig) -> None:
        self.cfg = cfg

    def _openai_chat_completions_payload(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        force_disable_thinking: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": model, "messages": messages}
        if temperature is not None:
            payload["temperature"] = float(temperature)
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
        # 百炼：Python SDK 里写在 extra_body 的字段，最终会并入请求 JSON 顶层。
        # 直接用 httpx 时若只包一层 extra_body，网关可能不展开，导致 enable_thinking 仍走模型默认（易变成思考模式）。
        if force_disable_thinking:
            payload["enable_thinking"] = False
            payload["max_reasoning_tokens"] = 0
        else:
            thinking = bool(self.cfg.enable_thinking)
            payload["enable_thinking"] = thinking
            if thinking:
                if self.cfg.max_reasoning_tokens is not None:
                    payload["max_reasoning_tokens"] = int(self.cfg.max_reasoning_tokens)
            else:
                # 关闭思考时仍可能消耗 reasoning_tokens 导致 content 为空，显式压零
                payload["max_reasoning_tokens"] = 0
        thinking_on = bool(self.cfg.enable_thinking) and not force_disable_thinking
        if _is_likely_ollama_openai_base_url(self.cfg.base_url) and not thinking_on:
            # Ollama /v1/chat/completions：见官方文档 Reasoning/thinking control
            payload["reasoning_effort"] = "none"
        return payload

    @classmethod
    def _append_guard_to_messages(
        cls, messages: list[dict[str, Any]], *, retry_hint: bool = False
    ) -> list[dict[str, Any]]:
        """
        在消息中加入“必须输出最终正文”的约束：
        - 优先追加到 system；
        - 若不存在 system，则追加到最后一个 user 文本。
        """
        guard = cls._FINAL_OUTPUT_GUARD + (cls._RETRY_FORCE_FINAL if retry_hint else "")
        out: list[dict[str, Any]] = [dict(m) for m in messages]

        for i, m in enumerate(out):
            if str(m.get("role") or "") != "system":
                continue
            c = m.get("content")
            if isinstance(c, str):
                if guard not in c:
                    out[i]["content"] = c + guard
                return out

        for i in range(len(out) - 1, -1, -1):
            if str(out[i].get("role") or "") != "user":
                continue
            c = out[i].get("content")
            if isinstance(c, str):
                if guard not in c:
                    out[i]["content"] = c + guard
                return out
            if isinstance(c, list):
                parts = [dict(x) if isinstance(x, dict) else x for x in c]
                for j in range(len(parts) - 1, -1, -1):
                    p = parts[j]
                    if isinstance(p, dict) and isinstance(p.get("text"), str):
                        if guard not in p["text"]:
                            p["text"] = p["text"] + guard
                        out[i]["content"] = parts
                        return out
                parts.append({"type": "text", "text": guard.strip()})
                out[i]["content"] = parts
                return out
        return out

    @staticmethod
    def _endpoint_kind_from_url(url: str) -> str:
        if "chat/completions" in url:
            return "openai-chat-completions"
        if "multimodal-generation" in url:
            return "multimodal-generation"
        if "text-generation" in url:
            return "text-generation"
        return url.rstrip("/").rsplit("/", 1)[-1] or url

    @staticmethod
    def _summarize_messages_for_log(messages: list[dict[str, Any]]) -> str:
        """仅记录条数/角色/文本长度/图片数，避免把正文写入日志。"""
        parts: list[str] = []
        for i, m in enumerate(messages):
            role = str(m.get("role") or "?")
            c = m.get("content")
            if isinstance(c, str):
                parts.append(f"{i}:{role} text_chars={len(c)}")
            elif isinstance(c, list):
                n_img = 0
                n_text = 0
                for x in c:
                    if not isinstance(x, dict):
                        continue
                    if x.get("type") == "image_url":
                        n_img += 1
                    elif x.get("type") == "text":
                        n_text += len(str(x.get("text", "")))
                    elif isinstance(x.get("image"), str):
                        n_img += 1
                    elif isinstance(x.get("text"), str):
                        n_text += len(x["text"])
                parts.append(f"{i}:{role} images={n_img} text_chars={n_text}")
            else:
                parts.append(f"{i}:{role} content_type={type(c).__name__}")
        return "; ".join(parts) if parts else "(empty)"

    @staticmethod
    def _image_ref_to_openai_url(ref: str) -> str:
        ref = ref.strip()
        if ref.startswith(("http://", "https://", "data:")):
            return ref
        p = Path(ref)
        if not p.is_file():
            raise FileNotFoundError(f"LLM image input not found: {ref}")
        data = p.read_bytes()
        b64 = base64.standard_b64encode(data).decode("ascii")
        ext = p.suffix.lower()
        mime = _EXT_TO_MIME.get(ext, "image/png")
        return f"data:{mime};base64,{b64}"

    @classmethod
    def _messages_to_openai_format(
        cls, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """DashScope 原生 message → OpenAI Chat Completions 多模态格式。"""
        out: list[dict[str, Any]] = []
        for m in messages:
            role = str(m.get("role") or "user")
            c = m.get("content")
            if isinstance(c, str):
                out.append({"role": role, "content": c})
            elif isinstance(c, list):
                parts: list[dict[str, Any]] = []
                for item in c:
                    if not isinstance(item, dict):
                        continue
                    if "image" in item and isinstance(item["image"], str):
                        url = cls._image_ref_to_openai_url(item["image"])
                        parts.append(
                            {"type": "image_url", "image_url": {"url": url}}
                        )
                    elif "text" in item:
                        parts.append({"type": "text", "text": str(item["text"])})
                out.append({"role": role, "content": parts})
            else:
                out.append({"role": role, "content": str(c)})
        return out

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }
        kind = self._endpoint_kind_from_url(url)
        model = str(payload.get("model") or "?")
        if "messages" in payload:
            messages = payload.get("messages") or []
            temp = payload.get("temperature")
            max_tok = payload.get("max_tokens")
            rf = "openai"
        else:
            params = payload.get("parameters") or {}
            messages = (payload.get("input") or {}).get("messages") or []
            temp = params.get("temperature")
            max_tok = params.get("max_tokens")
            rf = params.get("result_format")
        msg_summary = self._summarize_messages_for_log(messages)
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                if attempt == 1:
                    _log.info(
                        "LLM call start: kind=%s model=%s temperature=%s max_tokens=%s "
                        "result_format=%s messages=%s",
                        kind,
                        model,
                        temp,
                        max_tok,
                        rf,
                        msg_summary,
                    )
                t0 = time.perf_counter()
                with httpx.Client(timeout=_httpx_timeout(self.cfg.timeout_sec)) as client:
                    resp = client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                elapsed = time.perf_counter() - t0
                usage = data.get("usage")
                request_id = data.get("request_id") or data.get("id")
                _log.info(
                    "LLM call done: kind=%s model=%s elapsed=%.3fs attempt=%s/%s "
                    "usage=%s request_id=%s",
                    kind,
                    model,
                    elapsed,
                    attempt,
                    self.cfg.max_retries,
                    usage,
                    request_id,
                )
                _log.info("LLM 本次交互耗时 %.3f 秒", elapsed)
                return data
            except Exception as e:  # noqa: BLE001 - keep broad for network/runtime
                last_exc = e
                if attempt >= self.cfg.max_retries:
                    raise
                sleep_s = self.cfg.retry_backoff_sec * (2 ** (attempt - 1))
                _log.warning(
                    "DashScope request failed (attempt=%s/%s): %s; sleep %.1fs",
                    attempt,
                    self.cfg.max_retries,
                    repr(e),
                    sleep_s,
                )
                time.sleep(sleep_s)
        # should be unreachable
        assert last_exc is not None
        raise last_exc

    @staticmethod
    def _append_openai_delta_to_buffers(
        delta: dict[str, Any],
        *,
        acc_content: list[str],
        acc_reasoning: list[str],
    ) -> None:
        """从流式 chunk 的 delta 中取出文本并写入缓冲区；正文与思考链均打 INFO（便于实时查看）。"""
        c = delta.get("content")
        if isinstance(c, str) and c:
            acc_content.append(c)
            _log.info("LLM stream out | %s", c)
            _flush_logging_handlers()
        elif isinstance(c, list):
            for item in c:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    t = item["text"]
                    acc_content.append(t)
                    _log.info("LLM stream out | %s", t)
                    _flush_logging_handlers()
        rc = delta.get("reasoning_content")
        if isinstance(rc, str) and rc:
            acc_reasoning.append(rc)
            _log.info("LLM stream reasoning | %s", rc)
            _flush_logging_handlers()
        th = delta.get("thinking")
        if isinstance(th, str) and th:
            acc_reasoning.append(th)
            _log.info("LLM stream thinking | %s", th)
            _flush_logging_handlers()

    def _post_openai_chat_completions_stream(
        self, url: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """
        OpenAI 兼容流式：``stream: true`` + SSE，拼出与单次 JSON 等价的 choices[0].message。
        仅在 ``log_stream_response`` 为 True 时使用。
        """
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        stream_payload = dict(payload)
        stream_payload["stream"] = True

        kind = self._endpoint_kind_from_url(url)
        model = str(payload.get("model") or "?")
        messages = payload.get("messages") or []
        temp = payload.get("temperature")
        max_tok = payload.get("max_tokens")
        msg_summary = self._summarize_messages_for_log(messages)
        rf = "openai"

        last_exc: Optional[Exception] = None
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                if attempt == 1:
                    _log.info(
                        "LLM call start (stream): kind=%s model=%s temperature=%s max_tokens=%s "
                        "result_format=%s messages=%s",
                        kind,
                        model,
                        temp,
                        max_tok,
                        rf,
                        msg_summary,
                    )
                t0 = time.perf_counter()
                acc_content: list[str] = []
                acc_reasoning: list[str] = []
                usage: Optional[dict[str, Any]] = None
                request_id: Optional[str] = None

                with httpx.Client(timeout=_httpx_timeout(self.cfg.timeout_sec)) as client:
                    with client.stream(
                        "POST", url, json=stream_payload, headers=headers
                    ) as resp:
                        resp.raise_for_status()
                        for line in resp.iter_lines():
                            if line is None:
                                continue
                            if isinstance(line, bytes):
                                line = line.decode("utf-8", errors="replace")
                            line = line.strip()
                            if not line or line.startswith(":"):
                                continue
                            if not line.startswith("data: "):
                                continue
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk: dict[str, Any] = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue
                            rid = chunk.get("request_id") or chunk.get("id")
                            if rid:
                                request_id = str(rid)
                            u = chunk.get("usage")
                            if isinstance(u, dict):
                                usage = u
                            choices = chunk.get("choices") or []
                            if not choices:
                                continue
                            ch0 = choices[0] or {}
                            delta = ch0.get("delta")
                            if isinstance(delta, dict) and delta:
                                DashScopeClient._append_openai_delta_to_buffers(
                                    delta,
                                    acc_content=acc_content,
                                    acc_reasoning=acc_reasoning,
                                )
                            # 少数实现仅在最后一包给完整 message（含整段思考链）
                            msg_final = ch0.get("message")
                            if isinstance(msg_final, dict) and msg_final:
                                sub = msg_final.get("content")
                                if isinstance(sub, str) and sub and not acc_content:
                                    acc_content.append(sub)
                                    _log.info("LLM stream out | %s", sub)
                                    _flush_logging_handlers()
                                if not acc_reasoning:
                                    rc_f = msg_final.get("reasoning_content")
                                    if isinstance(rc_f, str) and rc_f:
                                        acc_reasoning.append(rc_f)
                                        _log.info("LLM stream reasoning | %s", rc_f)
                                        _flush_logging_handlers()
                                    th_f = msg_final.get("thinking")
                                    if isinstance(th_f, str) and th_f:
                                        acc_reasoning.append(th_f)
                                        _log.info("LLM stream thinking | %s", th_f)
                                        _flush_logging_handlers()
                elapsed = time.perf_counter() - t0
                full_content = "".join(acc_content)
                msg_out: dict[str, Any] = {"content": full_content}
                if acc_reasoning:
                    msg_out["reasoning_content"] = "".join(acc_reasoning)
                out: dict[str, Any] = {
                    "choices": [
                        {
                            "message": msg_out,
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": usage,
                }
                if request_id:
                    out["id"] = request_id
                    out["request_id"] = request_id
                _log.info(
                    "LLM call done (stream): kind=%s model=%s elapsed=%.3fs attempt=%s/%s "
                    "usage=%s request_id=%s out_chars=%s",
                    kind,
                    model,
                    elapsed,
                    attempt,
                    self.cfg.max_retries,
                    usage,
                    request_id,
                    len(full_content),
                )
                _log.info("LLM 本次交互耗时 %.3f 秒", elapsed)
                return out
            except Exception as e:  # noqa: BLE001
                last_exc = e
                if attempt >= self.cfg.max_retries:
                    raise
                sleep_s = self.cfg.retry_backoff_sec * (2 ** (attempt - 1))
                _log.warning(
                    "DashScope stream request failed (attempt=%s/%s): %s; sleep %.1fs",
                    attempt,
                    self.cfg.max_retries,
                    repr(e),
                    sleep_s,
                )
                time.sleep(sleep_s)
        assert last_exc is not None
        raise last_exc

    @staticmethod
    def _openai_message_to_text(
        msg: dict[str, Any], usage: Optional[dict[str, Any]] = None
    ) -> str:
        """
        OpenAI 兼容：content 可能是 str，也可能是 [{type:text,text:...}]。

        思考模式下正文应在 content；reasoning_content 为链式思考，**不作为** Markdown 正文回填，
        避免把几千字分析灌进输出（见 usage.completion_tokens_details）。
        """
        content = msg.get("content")
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text" and isinstance(item.get("text"), str):
                        parts.append(item["text"])
                    elif isinstance(item.get("text"), str):
                        parts.append(item["text"])
                elif isinstance(item, str):
                    parts.append(item)
            joined = "".join(parts)
            if joined.strip():
                return joined
        elif isinstance(content, str) and content.strip():
            return content

        rc = msg.get("reasoning_content")
        if isinstance(rc, str) and rc.strip():
            ctd = (usage or {}).get("completion_tokens_details") or {}
            _log.warning(
                "Qwen 思考链：content 为空但 reasoning_content 非空 "
                "(text_tokens=%s, reasoning_tokens=%s)。"
                " 请确认请求 JSON 顶层含 enable_thinking=false（OpenAI SDK 对应 extra_body；"
                "本客户端默认关闭）；或提高 max_tokens；勿将 reasoning 当作正文。",
                ctd.get("text_tokens"),
                ctd.get("reasoning_tokens"),
            )
        th = msg.get("thinking")
        if isinstance(th, str) and th.strip():
            _log.warning(
                "Ollama/Qwen：content 为空但 message.thinking 非空（约 %d 字符）。"
                " 百炼字段 enable_thinking/max_reasoning_tokens 对 Ollama 无效；"
                " 已对疑似 Ollama(端口 11434) 自动附加 reasoning_effort=none。"
                " 若仍出现请升级 Ollama（≥0.17.4 相关修复）或换用非思考型模型。",
                len(th),
            )
        return ""

    @staticmethod
    def _native_message_content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list) and content:
            texts: list[str] = []
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    texts.append(part["text"])
            if texts:
                return "\n".join(texts)
            first = content[0]
            if isinstance(first, dict) and isinstance(first.get("text"), str):
                return first["text"]
        return ""

    @staticmethod
    def _extract_text(response_json: dict[str, Any]) -> str:
        # OpenAI 兼容：choices[0].message
        try:
            ch = response_json.get("choices") or []
            if ch:
                msg = ch[0].get("message") or {}
                text = DashScopeClient._openai_message_to_text(
                    msg, usage=response_json.get("usage")
                )
                if text.strip():
                    return text
                fr = ch[0].get("finish_reason")
                if fr or msg:
                    _log.warning(
                        "LLM assistant 文本为空：finish_reason=%s message_keys=%s",
                        fr,
                        list(msg.keys()),
                    )
        except Exception:
            pass
        # DashScope 原生
        try:
            choices = response_json["output"]["choices"]
            msg = choices[0]["message"]
            content = msg.get("content")
            t = DashScopeClient._native_message_content_to_text(content)
            if t.strip():
                return t
        except Exception:
            pass
        # as a final fallback, return pretty json to help debugging
        return str(response_json)

    @staticmethod
    def _extract_openai_assistant_text(response_json: dict[str, Any]) -> str:
        """仅从 OpenAI 兼容 choices[0].message 取正文；失败返回空串（不用整段 JSON 兜底）。"""
        try:
            ch = response_json.get("choices") or []
            if not ch:
                return ""
            msg = (ch[0] or {}).get("message") or {}
            return DashScopeClient._openai_message_to_text(
                msg, usage=response_json.get("usage")
            )
        except Exception:
            return ""

    def _openai_chat_completions_generate(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> str:
        """
        OpenAI 兼容 /chat/completions：带输出约束与「空 content」重试。

        Qwen3.5 等模型可能把全文写进 reasoning_content；此处禁止接受该结果，
        在 empty_content_max_attempts 次内强制关闭思考并重试，仍失败则抛错。
        """
        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        n = max(1, min(10, int(self.cfg.empty_content_max_attempts)))
        last_json: Optional[dict[str, Any]] = None

        for attempt in range(n):
            retry_hint = attempt > 0
            guarded = self._append_guard_to_messages(messages, retry_hint=retry_hint)
            om = self._messages_to_openai_format(guarded)
            force_off = attempt > 0
            temp = temperature
            if attempt > 0:
                if temp is None:
                    temp = 0.0
                else:
                    temp = min(float(temp), 0.2)

            payload = self._openai_chat_completions_payload(
                model=model,
                messages=om,
                temperature=temp,
                max_tokens=max_tokens,
                force_disable_thinking=force_off,
            )
            if self.cfg.log_stream_response:
                response_json = self._post_openai_chat_completions_stream(url, payload)
            else:
                response_json = self._post_json(url, payload)
            last_json = response_json
            text = self._extract_openai_assistant_text(response_json)
            if text.strip():
                return text

            rid = response_json.get("id") or response_json.get("request_id")
            ro = self._is_reasoning_only_openai_response(response_json)
            _log.warning(
                "OpenAI 兼容接口无可用正文: attempt=%s/%s reasoning_only=%s request_id=%s",
                attempt + 1,
                n,
                ro,
                rid,
            )

        last_id = (last_json or {}).get("id") or (last_json or {}).get("request_id")
        raise RuntimeError(
            "DashScope OpenAI 兼容接口在 "
            f"{n} 次尝试后 assistant.content 仍为空（reasoning_content 不计入正文）。"
            f" last_request_id={last_id!r}"
        )

    @staticmethod
    def _is_reasoning_only_openai_response(response_json: dict[str, Any]) -> bool:
        """判断是否出现了 reasoning_content 非空但 content 为空的响应。"""
        try:
            ch = response_json.get("choices") or []
            if not ch:
                return False
            msg = ch[0].get("message") or {}
            content = msg.get("content")
            has_content = False
            if isinstance(content, str):
                has_content = bool(content.strip())
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, str) and item.strip():
                        has_content = True
                        break
                    if (
                        isinstance(item, dict)
                        and isinstance(item.get("text"), str)
                        and item["text"].strip()
                    ):
                        has_content = True
                        break
            rc = msg.get("reasoning_content")
            has_reasoning = isinstance(rc, str) and bool(rc.strip())
            th = msg.get("thinking")
            has_thinking = isinstance(th, str) and bool(th.strip())
            return (has_reasoning or has_thinking) and (not has_content)
        except Exception:
            return False

    def generate_multimodal(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        result_format: str = "message",
    ) -> str:
        base = self.cfg.base_url.rstrip("/")
        if _is_openai_compatible_base_url(base):
            return self._openai_chat_completions_generate(
                model,
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        url = base + "/services/aigc/multimodal-generation/generation"
        payload = {
            "model": model,
            "input": {"messages": messages},
            "parameters": {"result_format": result_format},
        }
        if temperature is not None:
            payload["parameters"]["temperature"] = float(temperature)
        if max_tokens is not None:
            payload["parameters"]["max_tokens"] = int(max_tokens)
        response_json = self._post_json(url, payload)
        return self._extract_text(response_json)


def build_vl_user_message(
    *,
    text: str,
    image_paths: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    构造适用于 Qwen-VL 的 user message。

    - image_paths：本地绝对路径或公网 URL
    - text：需要模型理解/纠错的内容
    """
    if not image_paths:
        return {"role": "user", "content": text}
    content: list[dict[str, str] | str] = [{"image": p} for p in image_paths] + [{"text": text}]
    # docs 允许 content 为 array；这里用 [{"image": ...}, {"text": ...}]。
    return {"role": "user", "content": content}


def build_system_message(content: str) -> dict[str, Any]:
    return {"role": "system", "content": content}

