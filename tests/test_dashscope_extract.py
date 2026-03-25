import unittest
from unittest.mock import patch

from src.dashscope_client import (
    DashScopeClient,
    DashScopeClientConfig,
    _is_openai_compatible_base_url,
)


class TestDashScopeExtract(unittest.TestCase):
    def test_openai_ignores_reasoning_when_content_empty(self) -> None:
        msg = {
            "role": "assistant",
            "content": "",
            "reasoning_content": "long chain of thought...",
        }
        usage = {
            "completion_tokens_details": {
                "reasoning_tokens": 100,
                "text_tokens": 0,
            }
        }
        out = DashScopeClient._openai_message_to_text(msg, usage=usage)
        self.assertEqual(out, "")

    def test_openai_list_content(self) -> None:
        msg = {
            "role": "assistant",
            "content": [{"type": "text", "text": "# Hi"}],
        }
        out = DashScopeClient._openai_message_to_text(msg, usage=None)
        self.assertEqual(out, "# Hi")

    def test_chat_payload_enable_thinking_top_level(self) -> None:
        cfg = DashScopeClientConfig(
            api_key="k",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            enable_thinking=True,
            max_reasoning_tokens=128,
        )
        c = DashScopeClient(cfg)
        p = c._openai_chat_completions_payload(
            model="qwen3.5-plus",
            messages=[],
            temperature=0.0,
            max_tokens=100,
        )
        self.assertIs(p.get("enable_thinking"), True)
        self.assertEqual(p.get("max_reasoning_tokens"), 128)
        self.assertNotIn("extra_body", p)

    def test_chat_payload_thinking_off_zero_reasoning(self) -> None:
        cfg = DashScopeClientConfig(
            api_key="k",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            enable_thinking=False,
            max_reasoning_tokens=256,
        )
        c = DashScopeClient(cfg)
        p = c._openai_chat_completions_payload(
            model="qwen3.5-plus",
            messages=[],
            temperature=0.0,
            max_tokens=100,
        )
        self.assertIs(p.get("enable_thinking"), False)
        self.assertEqual(p.get("max_reasoning_tokens"), 0)

    def test_chat_payload_force_disable_thinking(self) -> None:
        cfg = DashScopeClientConfig(
            api_key="k",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            enable_thinking=True,
            max_reasoning_tokens=512,
        )
        c = DashScopeClient(cfg)
        p = c._openai_chat_completions_payload(
            model="qwen",
            messages=[],
            temperature=None,
            max_tokens=10,
            force_disable_thinking=True,
        )
        self.assertIs(p.get("enable_thinking"), False)
        self.assertEqual(p.get("max_reasoning_tokens"), 0)

    def test_openai_generate_raises_after_empty_attempts(self) -> None:
        cfg = DashScopeClientConfig(
            api_key="k",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            empty_content_max_attempts=2,
        )
        c = DashScopeClient(cfg)
        empty = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "only here",
                    }
                }
            ],
            "id": "req-empty",
        }
        with patch.object(c, "_post_json", return_value=empty) as post:
            with self.assertRaises(RuntimeError) as ctx:
                c._openai_chat_completions_generate(
                    "m",
                    [{"role": "user", "content": "hi"}],
                    temperature=0.1,
                    max_tokens=50,
                )
        self.assertIn("仍为空", str(ctx.exception))
        self.assertEqual(post.call_count, 2)

    def test_reasoning_only_response_detection(self) -> None:
        resp = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "thoughts...",
                    }
                }
            ]
        }
        self.assertTrue(DashScopeClient._is_reasoning_only_openai_response(resp))

    def test_append_guard_to_messages_on_system(self) -> None:
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u"},
        ]
        out = DashScopeClient._append_guard_to_messages(msgs)
        self.assertNotEqual(out[0]["content"], "sys")
        self.assertIn("content", out[0]["content"])
        self.assertIn("禁止把可交付正文只写在推理", out[0]["content"])

    def test_openai_compat_url_detection_ollama(self) -> None:
        self.assertTrue(
            _is_openai_compatible_base_url("http://localhost:11434/v1")
        )
        self.assertTrue(_is_openai_compatible_base_url("http://127.0.0.1:11434/v1/"))

    def test_openai_compat_url_detection_dashscope_native(self) -> None:
        self.assertFalse(
            _is_openai_compatible_base_url(
                "https://dashscope.aliyuncs.com/api/v1"
            )
        )

    def test_openai_compat_url_detection_dashscope_openai(self) -> None:
        self.assertTrue(
            _is_openai_compatible_base_url(
                "https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
        )


if __name__ == "__main__":
    unittest.main()
