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
