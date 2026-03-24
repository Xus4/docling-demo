from __future__ import annotations

import base64
import logging
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
    return "compatible-mode" in (base_url or "").lower()


@dataclass(frozen=True)
class DashScopeClientConfig:
    api_key: str
    # 原生：…/api/v1；OpenAI 兼容：…/compatible-mode/v1（见百炼文档）
    base_url: str
    timeout_sec: float = 180.0
    max_retries: int = 3
    retry_backoff_sec: float = 1.5
    # Qwen3.5 / Qwen3-VL：思考模式会把正文放在 reasoning_content，content 可能为空。
    # 百炼文档：通过 extra_body.enable_thinking 控制；文档/转写默认关闭思考。
    enable_thinking: bool = False


class DashScopeClient:
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
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": model, "messages": messages}
        if temperature is not None:
            payload["temperature"] = float(temperature)
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
        # 百炼：Python SDK 里写在 extra_body 的字段，最终会并入请求 JSON 顶层。
        # 直接用 httpx 时若只包一层 extra_body，网关可能不展开，导致 enable_thinking 仍走模型默认（易变成思考模式）。
        payload["enable_thinking"] = bool(self.cfg.enable_thinking)
        return payload

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
                with httpx.Client(timeout=self.cfg.timeout_sec) as client:
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

    def generate_text(
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
            url = base + "/chat/completions"
            om = self._messages_to_openai_format(messages)
            payload = self._openai_chat_completions_payload(
                model=model,
                messages=om,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            response_json = self._post_json(url, payload)
            return self._extract_text(response_json)
        url = base + "/services/aigc/text-generation/generation"
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
            url = base + "/chat/completions"
            om = self._messages_to_openai_format(messages)
            payload = self._openai_chat_completions_payload(
                model=model,
                messages=om,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            response_json = self._post_json(url, payload)
            return self._extract_text(response_json)
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

