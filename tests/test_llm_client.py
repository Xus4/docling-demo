"""阶段 C：OpenAI-compatible 客户端。"""

from __future__ import annotations

import json
import unittest

import httpx

from src.table_semantic.llm_client import (
    LLMClientError,
    OpenAICompatibleConfig,
    chat_completion_json_object,
    chat_completion_text,
)


class TestOpenAICompatibleClient(unittest.TestCase):
    def test_chat_completion_text_ok(self) -> None:
        payload = {"choices": [{"message": {"content": "hello"}}]}

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertIn("/v1/chat/completions", str(request.url))
            auth = request.headers.get("authorization") or request.headers.get("Authorization")
            self.assertEqual(auth, "Bearer k")
            return httpx.Response(200, json=payload)

        transport = httpx.MockTransport(handler)
        cfg = OpenAICompatibleConfig(
            base_url="http://example.com",
            api_key="k",
            model="m",
            timeout_sec=10.0,
        )
        with httpx.Client(transport=transport) as client:
            out = chat_completion_text(
                cfg=cfg,
                messages=[{"role": "user", "content": "x"}],
                http_client=client,
            )
        self.assertEqual(out, "hello")

    def test_http_error_raises(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="boom")

        cfg = OpenAICompatibleConfig(
            base_url="http://x.com/v1",
            api_key=None,
            model="m",
            timeout_sec=5.0,
        )
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            with self.assertRaises(LLMClientError):
                chat_completion_text(
                    cfg=cfg,
                    messages=[{"role": "user", "content": "a"}],
                    http_client=client,
                )

    def test_connect_error_on_post_becomes_llm_client_error(self) -> None:
        """POST 阶段抛出的 RequestError（非 HTTPStatusError）须包装为 LLMClientError。"""

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("simulated network failure", request=request)

        cfg = OpenAICompatibleConfig(
            base_url="http://x.com/v1",
            api_key=None,
            model="m",
            timeout_sec=5.0,
        )
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            with self.assertRaises(LLMClientError) as ctx:
                chat_completion_text(
                    cfg=cfg,
                    messages=[{"role": "user", "content": "a"}],
                    http_client=client,
                )
        self.assertIn("simulated network failure", str(ctx.exception))

    def test_chat_completion_json_object(self) -> None:
        payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({"summary": "ok", "entities": []}),
                    }
                },
            ]
        }

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=payload)

        cfg = OpenAICompatibleConfig(
            base_url="http://x.com",
            api_key="",
            model="m",
            timeout_sec=5.0,
        )
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            data = chat_completion_json_object(
                cfg=cfg,
                messages=[{"role": "user", "content": "{}"}],
                http_client=client,
            )
        self.assertEqual(data.get("summary"), "ok")

    def test_chat_completion_thinking_ollama_top_level_boolean(self) -> None:
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content.decode())
            return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

        cfg = OpenAICompatibleConfig(
            base_url="http://example.com",
            api_key="k",
            model="m",
            timeout_sec=10.0,
            thinking_enable=True,
        )
        transport = httpx.MockTransport(handler)
        with httpx.Client(transport=transport) as client:
            chat_completion_json_object(
                cfg=cfg,
                messages=[{"role": "user", "content": "{}"}],
                http_client=client,
            )
        body = captured["body"]
        self.assertIsInstance(body, dict)
        assert isinstance(body, dict)
        self.assertIs(body.get("thinking"), True)
        self.assertEqual(body.get("response_format"), {"type": "json_object"})

    def test_chat_completion_thinking_false_explicit_from_config(self) -> None:
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content.decode())
            return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

        cfg = OpenAICompatibleConfig(
            base_url="http://example.com",
            api_key="k",
            model="m",
            timeout_sec=10.0,
            thinking_enable=False,
        )
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            chat_completion_json_object(
                cfg=cfg,
                messages=[{"role": "user", "content": "{}"}],
                http_client=client,
            )
        body = captured["body"]
        assert isinstance(body, dict)
        self.assertIs(body.get("thinking"), False)
        self.assertNotIn("enable_thinking", body)
        self.assertNotIn("chat_template_kwargs", body)

    def test_generation_params_in_request(self) -> None:
        """测试生成参数是否正确传递到请求体"""
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content.decode())
            return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

        cfg = OpenAICompatibleConfig(
            base_url="http://example.com",
            api_key="k",
            model="m",
            timeout_sec=10.0,
            max_tokens=4000,
            temperature=0.7,
        )
        transport = httpx.MockTransport(handler)
        with httpx.Client(transport=transport) as client:
            chat_completion_json_object(
                cfg=cfg,
                messages=[{"role": "user", "content": "test"}],
                http_client=client,
            )
        body = captured["body"]
        assert isinstance(body, dict)
        self.assertEqual(body.get("max_tokens"), 4000)
        self.assertEqual(body.get("temperature"), 0.7)

    def test_none_generation_params_not_in_request(self) -> None:
        """测试 None 值参数不会出现在请求体中"""
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content.decode())
            return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

        cfg = OpenAICompatibleConfig(
            base_url="http://example.com",
            api_key="k",
            model="m",
            timeout_sec=10.0,
        )
        transport = httpx.MockTransport(handler)
        with httpx.Client(transport=transport) as client:
            chat_completion_json_object(
                cfg=cfg,
                messages=[{"role": "user", "content": "test"}],
                http_client=client,
            )
        body = captured["body"]
        assert isinstance(body, dict)
        self.assertNotIn("max_tokens", body)
        self.assertNotIn("temperature", body)


if __name__ == "__main__":
    unittest.main()
