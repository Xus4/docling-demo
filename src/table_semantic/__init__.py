"""表格语义增强：表格块扫描与后续 LLM 增强（分阶段实现）。"""

from src.table_semantic.augment import (
    DEFAULT_TABLE_CAPTION_PARAMS,
    TableCaptionParams,
    augment_markdown_file,
    augment_markdown_text,
    build_table_caption_messages,
)
from src.table_semantic.llm_client import OpenAICompatibleConfig, chat_completion_json_object
from src.table_semantic.table_blocks import TableBlock, iter_table_blocks

__all__ = [
    "TableBlock",
    "iter_table_blocks",
    "OpenAICompatibleConfig",
    "chat_completion_json_object",
    "TableCaptionParams",
    "DEFAULT_TABLE_CAPTION_PARAMS",
    "build_table_caption_messages",
    "augment_markdown_text",
    "augment_markdown_file",
]
