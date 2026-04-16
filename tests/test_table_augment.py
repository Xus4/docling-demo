"""阶段 D：表格语义插入与幂等。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.table_semantic.augment import augment_markdown_file, augment_markdown_text
from src.table_semantic.llm_client import ChatCompletionMeta, OpenAICompatibleConfig


class TestAugmentMarkdownText(unittest.TestCase):
    def test_inserts_and_idempotent(self) -> None:
        md = "前文\n|a|\n|---|\n|1|\n后文\n"
        cfg = OpenAICompatibleConfig(
            base_url="http://unused",
            api_key=None,
            model="m",
            timeout_sec=5.0,
        )

        def fake_json(*_a: object, **_k: object) -> tuple[dict[str, object], ChatCompletionMeta]:
            return {"summary": "这是表格语义。"}, ChatCompletionMeta()

        with mock.patch(
            "src.table_semantic.augment.chat_completion_json_object_with_meta",
            side_effect=fake_json,
        ):
            out1 = augment_markdown_text(
                md, cfg=cfg, max_concurrency=1
            )
        self.assertIn("这是表格语义", out1)
        self.assertIn("table-semantic", out1)

        with mock.patch(
            "src.table_semantic.augment.chat_completion_json_object_with_meta",
        ) as p2:
            out2 = augment_markdown_text(
                out1, cfg=cfg, max_concurrency=1
            )
        self.assertEqual(out1, out2)
        p2.assert_not_called()

    def test_with_generation_params(self) -> None:
        """测试带有生成参数的配置"""
        md = "|a|\n|---|\n|1|\n"
        cfg = OpenAICompatibleConfig(
            base_url="http://unused",
            api_key=None,
            model="m",
            timeout_sec=5.0,
            max_tokens=2000,
            temperature=0.7,
        )

        def fake_json(*_a: object, **_k: object) -> tuple[dict[str, object], ChatCompletionMeta]:
            return {"summary": "带参数的语义"}, ChatCompletionMeta()

        with mock.patch(
            "src.table_semantic.augment.chat_completion_json_object_with_meta",
            side_effect=fake_json,
        ):
            out = augment_markdown_text(md, cfg=cfg, max_concurrency=1)
        self.assertIn("带参数的语义", out)


class TestAugmentMarkdownFile(unittest.TestCase):
    def test_file_roundtrip(self) -> None:
        md = "|x|\n|---|\n|y|\n"
        cfg = OpenAICompatibleConfig(
            base_url="http://unused",
            api_key=None,
            model="m",
            timeout_sec=5.0,
        )

        def fake_json(*_a: object, **_k: object) -> tuple[dict[str, object], ChatCompletionMeta]:
            return {"summary": "文件内语义"}, ChatCompletionMeta()

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "t.md"
            p.write_text(md, encoding="utf-8")
            with mock.patch(
                "src.table_semantic.augment.chat_completion_json_object_with_meta",
                side_effect=fake_json,
            ):
                augment_markdown_file(p, cfg=cfg, max_concurrency=1)
            body = p.read_text(encoding="utf-8")
            self.assertIn("文件内语义", body)


if __name__ == "__main__":
    unittest.main()
