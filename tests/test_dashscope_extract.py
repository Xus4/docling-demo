import unittest

from src.dashscope_client import DashScopeClient, DashScopeClientConfig


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
        self.assertIn("必须在最终答案通道输出", out[0]["content"])


if __name__ == "__main__":
    unittest.main()
