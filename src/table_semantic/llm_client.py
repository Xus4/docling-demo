"""OpenAI-compatible Chat Completions 客户端（httpx）。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx


class LLMClientError(Exception):
    """调用大模型网关失败或响应无法解析。"""


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    base_url: str
    api_key: str | None
    model: str
    timeout_sec: float
    thinking_enable: bool = False
    max_tokens: int | None = None  # 最大生成 token 数
    temperature: float | None = None  # 温度参数 (0-2)


@dataclass(frozen=True)
class ChatCompletionMeta:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    prompt_chars: int | None = None
    completion_chars: int | None = None


def _normalize_base(base: str) -> str:
    b = base.rstrip("/")
    if b.endswith("/v1"):
        return b
    return f"{b}/v1"


def _apply_thinking_to_body(
    body: dict[str, Any], cfg: OpenAICompatibleConfig
) -> None:
    """Ollama ``/v1/chat/completions`` 等：顶层 ``thinking`` 与配置一致，true/false 均显式传入。"""
    body["thinking"] = bool(cfg.thinking_enable)


def _apply_generation_params(
    body: dict[str, Any], cfg: OpenAICompatibleConfig
) -> None:
    """应用生成参数到请求体（仅非 None 值）。"""
    if cfg.max_tokens is not None:
        body["max_tokens"] = cfg.max_tokens
    if cfg.temperature is not None:
        body["temperature"] = cfg.temperature


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _extract_usage_meta(data: dict[str, Any], content: str) -> ChatCompletionMeta:
    usage = data.get("usage")
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    prompt_chars: int | None = None
    if isinstance(usage, dict):
        prompt_tokens = _safe_int(usage.get("prompt_tokens"))
        completion_tokens = _safe_int(usage.get("completion_tokens"))
        total_tokens = _safe_int(usage.get("total_tokens"))
        prompt_chars = _safe_int(usage.get("prompt_characters"))
        if prompt_chars is None:
            prompt_chars = _safe_int(usage.get("input_characters"))
    return ChatCompletionMeta(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        prompt_chars=prompt_chars,
        completion_chars=len(content),
    )


def chat_completion_text(
    *,
    cfg: OpenAICompatibleConfig,
    messages: list[dict[str, str]],
    json_mode: bool = False,
    http_client: httpx.Client | None = None,
) -> str:
    """POST ``/v1/chat/completions``，返回 ``choices[0].message.content`` 字符串。"""
    url = f"{_normalize_base(cfg.base_url)}/chat/completions"
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if cfg.api_key and cfg.api_key.strip():
        headers["Authorization"] = f"Bearer {cfg.api_key.strip()}"

    body: dict[str, Any] = {
        "model": cfg.model,
        "messages": messages,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}

    _apply_thinking_to_body(body, cfg)
    _apply_generation_params(body, cfg)

    timeout = httpx.Timeout(cfg.timeout_sec, connect=min(30.0, cfg.timeout_sec))
    owns_client = http_client is None
    client = http_client or httpx.Client(timeout=timeout)
    try:
        resp = client.post(url, headers=headers, json=body)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise LLMClientError(f"HTTP {resp.status_code}: {resp.text[:2000]}") from exc
        except httpx.RequestError as exc:
            raise LLMClientError(str(exc)) from exc
    finally:
        if owns_client:
            client.close()

    try:
        data = resp.json()
    except json.JSONDecodeError as exc:
        raise LLMClientError("响应非 JSON") from exc

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMClientError(f"响应缺少 choices: {data!r}"[:2000])
    msg = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = msg.get("content") if isinstance(msg, dict) else None
    if not isinstance(content, str):
        raise LLMClientError(f"响应缺少 message.content: {data!r}"[:2000])
    return content


def chat_completion_text_with_meta(
    *,
    cfg: OpenAICompatibleConfig,
    messages: list[dict[str, str]],
    json_mode: bool = False,
    http_client: httpx.Client | None = None,
) -> tuple[str, ChatCompletionMeta]:
    """POST ``/v1/chat/completions``，返回内容与 usage 元数据（若网关提供）。"""
    url = f"{_normalize_base(cfg.base_url)}/chat/completions"
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if cfg.api_key and cfg.api_key.strip():
        headers["Authorization"] = f"Bearer {cfg.api_key.strip()}"

    body: dict[str, Any] = {
        "model": cfg.model,
        "messages": messages,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    _apply_thinking_to_body(body, cfg)
    _apply_generation_params(body, cfg)

    timeout = httpx.Timeout(cfg.timeout_sec, connect=min(30.0, cfg.timeout_sec))
    owns_client = http_client is None
    client = http_client or httpx.Client(timeout=timeout)
    try:
        resp = client.post(url, headers=headers, json=body)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise LLMClientError(f"HTTP {resp.status_code}: {resp.text[:2000]}") from exc
        except httpx.RequestError as exc:
            raise LLMClientError(str(exc)) from exc
    finally:
        if owns_client:
            client.close()

    try:
        data = resp.json()
    except json.JSONDecodeError as exc:
        raise LLMClientError("响应非 JSON") from exc

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMClientError(f"响应缺少 choices: {data!r}"[:2000])
    msg = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = msg.get("content") if isinstance(msg, dict) else None
    if not isinstance(content, str):
        raise LLMClientError(f"响应缺少 message.content: {data!r}"[:2000])
    return content, _extract_usage_meta(data, content)


def chat_completion_json_object(
    *,
    cfg: OpenAICompatibleConfig,
    messages: list[dict[str, str]],
    http_client: httpx.Client | None = None,
) -> dict[str, Any]:
    raw = chat_completion_text(
        cfg=cfg, messages=messages, json_mode=True, http_client=http_client
    )
    try:
        out = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMClientError(f"模型返回非 JSON: {raw[:2000]}") from exc
    if not isinstance(out, dict):
        raise LLMClientError("模型 JSON 根类型须为 object")
    return out


def chat_completion_json_object_with_meta(
    *,
    cfg: OpenAICompatibleConfig,
    messages: list[dict[str, str]],
    http_client: httpx.Client | None = None,
) -> tuple[dict[str, Any], ChatCompletionMeta]:
    raw, meta = chat_completion_text_with_meta(
        cfg=cfg, messages=messages, json_mode=True, http_client=http_client
    )
    try:
        out = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMClientError(f"模型返回非 JSON: {raw[:2000]}") from exc
    if not isinstance(out, dict):
        raise LLMClientError("模型 JSON 根类型须为 object")
    return out, meta
