from __future__ import annotations

from pathlib import Path
from typing import Literal


def build_cleanup_messages(original_markdown: str) -> list[dict]:
    """
    返回 Qwen-VL 的 messages，用于“只输出最终 Markdown”的清洗/纠错。
    """
    system = (
        "你是文档 OCR 与版面纠错助手，优先保证**可读性、结构与引用完整**，其次才是措辞润色。"
        "在尽量保持原有结构（标题层级、段落边界、表格骨架含表头与分隔行、列表层级）的前提下，"
        "修复明显 OCR 错字、断行/空格异常、数字与单位粘连、标点与全半角错误。"
        "不确定处保持原样，不编造事实、不补充原文没有的信息。"
        "图片引用：所有 `![](...)` 路径必须逐字保留，不得删改、不得增删引用行。"
        "代码与公式：若存在 fenced 代码块或行内代码，仅修正块内明显错字，勿删除整块；勿把正文误包进代码围栏。"
        "输出：只返回纠错后的 Markdown 正文，不要前言后语，不要用 ``` 代码块包裹全文。"
    )

    user_text = (
        "请基于以下 Docling 生成的 Markdown 纠错，并结合所附页面/插图（若有）核对表格、标题与段落含义。\n\n"
        "【原始 Markdown 开始】\n"
        f"{original_markdown}\n"
        "【原始 Markdown 结束】\n"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_text},
    ]


def build_image_caption_messages(
    *,
    image_path: Path,
    context_text: str,
    max_chars: int = 0,
    image_mode: str = "local_abs",
) -> list[dict]:
    if image_mode == "url":
        image_ref = str(image_path)
    else:
        image_ref = str(image_path.resolve())

    length_rule = (
        f"单段输出，总字数不超过 {max_chars}。"
        if max_chars and max_chars > 0
        else "单段输出，信息完整优先，不必为压缩而省略关键可见信息。"
    )

    prompt = f"""你是工业文档图像理解助手。根据图片与下列上下文，写一段**可检索的语义补充**（便于读者理解该图在文档中的作用）。

要求（全部遵守）：
1. 使用中文。
2. 只写图像中可确认的内容；看不清的不臆测。
3. 先判断类型（如：照片、流程图、曲线/图表、表格截图、界面、装置/结构示意图等），再写实质信息。
4. 图内文字可概括要点，不要机械全文 OCR。
5. 结合上下文说明该图在文中的角色或强调点。
6. {length_rule}
7. 不要标题、不要编号列表、不要前缀如「图像补充：」。
8. 信息极少时客观描述可见部分即可。

上下文：
{context_text or "无"}
"""

    return [
        {
            "role": "user",
            "content": [
                {"image": image_ref},
                {"text": prompt},
            ],
        }
    ]


def build_quality_check_messages(
    *,
    original_markdown: str,
    refined_markdown: str,
    original_stats: dict[str, int],
    refined_stats: dict[str, int],
    pipeline_mode: Literal["docling", "pdf_vl"] = "docling",
) -> list[dict]:
    """
    返回用于 quality_check 的 messages，要求模型输出固定格式以便程序解析。
    """
    if pipeline_mode == "pdf_vl":
        system = (
            "你是文档质量评估助手。当前管线为：**视觉模型已生成整篇 Markdown，本次 refined 为在其上的润色/纠错**。"
            "请比较 original 与 refined，判断润色是否更可靠、是否仍有关键缺陷。"
            "本模式**不会**重新运行 Docling；请勿建议调整 Docling 流水线。"
            "因此 NEED_RERUN 应恒为 no（表示「不需要再跑 Docling」）；refined 若明显劣于 original，在 QC_SCORE 与 SUGGEST_REASON 中说明即可。"
            "只输出下列键值块，不要解释、不要使用 Markdown。"
        )
    else:
        system = (
            "你是文档质量评估助手。比较 Docling 原始 Markdown 与 LLM 纠错后的版本，判断 refined 是否更可靠，"
            "以及是否值得**提高 Docling 再转换成本**（rerun）以换取更好版面/OCR/表格。"
            "只输出下列键值块，不要解释、不要使用 Markdown。"
        )

    user_text = (
        "评分时请综合以下维度（0–100 分，高分表示 refined 更值得作为最终稿）：\n"
        "- 结构保持：标题层级、列表、表格行数、图片引用数量是否异常丢失或破坏。\n"
        "- 文本可信：错字与 OCR 问题是否改善，是否引入明显编造或文不对题。\n"
        "- 表格与数据：若存在表格，行列是否一致、勾选/符号是否更可信（若无法判断则保守给分）。\n\n"
        "【原始统计】"
        f" len_chars={original_stats.get('len_chars', 0)} "
        f" image_refs={original_stats.get('image_refs', 0)} "
        f" table_rows={original_stats.get('table_rows', 0)}\n"
        "【refine统计】"
        f" len_chars={refined_stats.get('len_chars', 0)} "
        f" image_refs={refined_stats.get('image_refs', 0)} "
        f" table_rows={refined_stats.get('table_rows', 0)}\n\n"
        "【原始 Markdown 开始】\n"
        f"{original_markdown}\n"
        "【原始 Markdown 结束】\n\n"
        "【refined Markdown 开始】\n"
        f"{refined_markdown}\n"
        "【refined Markdown 结束】\n\n"
        "输出格式（严格按键名输出，每行一条，无其它文字）：\n"
        "QC_SCORE=<0-100>\n"
        "NEED_RERUN=<yes|no>\n"
        "SUGGEST_OCR_QUALITY=<fast|balanced|high>\n"
        "SUGGEST_TABLE_ACCURACY=<fast|accurate>\n"
        "SUGGEST_GENERATE_PAGE_IMAGES=<yes|no>\n"
        "SUGGEST_REASON=<一句话，越短越好>\n"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_text},
    ]


def build_table_cleanup_messages(
    *,
    original_table_markdown: str,
    context_text: str,
) -> list[dict]:
    """
    构造 Qwen-VL 用于“表格单元格纠错”的 messages。
    输出必须是“只包含一个表格”的 Markdown（不返回解释/不返回代码块）。
    """
    system = (
        "你是文档表格纠错助手，目标是在**不改表格形状**的前提下修复 OCR 导致的单元格错字、漏字、符号（含 ✓、√、手写勾）。"
        "行数、列数、表头与 GFM 分隔行（| --- |）必须与输入一致；禁止增删行/列或合并单元格。"
        "若消息中含页面截图，请对照图像逐格核对，优先检查最右列与靠边单元格。"
        "只输出该表格的 Markdown 管道表正文，不要代码围栏，不要解释。"
    )

    user_text = (
        "【表格纠错上下文】\n"
        f"{context_text}\n\n"
        "【原始表格 Markdown】\n"
        f"{original_table_markdown}\n\n"
        "请输出修正后的**唯一**表格 Markdown（与输入同形）。"
        "若附有页面图像，请对照核对每一格，勿漏读右侧勾选。"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_text},
    ]


def build_table_caption_messages(
    *,
    table_markdown: str,
    context_text: str,
    max_chars: int = 800,
) -> list[dict]:
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
