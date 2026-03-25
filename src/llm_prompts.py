from __future__ import annotations


def build_cleanup_messages(original_markdown: str) -> list[dict]:
    """
    返回 Qwen-VL 的 messages，用于“只输出最终 Markdown”的清洗/纠错。
    """
    system = (
        "你是文档OCR/版面纠错助手。你的任务是：在尽量保持原有结构（标题层级、段落边界、表格骨架、列表结构）的前提下，"
        "修复明显的OCR错字、断行/空格异常、数字/单位粘连、标点错误等。"
        "如果你不确定某处内容，请保持原样，不要猜测新增内容。"
        "特别要求：原文里的所有图片引用 `![](<path>)`（包含括号内的路径）必须逐字保留，"
        "不得删除、不得改写路径、不得增删图片引用。"
        "输出时只能返回纠错后的 Markdown，不要返回任何解释、不要用代码块包裹。"
    )

    # 将原始 markdown 作为可被模型读到的文本块。注意模型可能会改写；约束见 system。
    user_text = (
        "请基于以下 Docling 生成的 Markdown 进行纠错，并结合你看到的图片理解其内容（用于确认表格/标题/段落）。\n\n"
        "【原始 Markdown 开始】\n"
        f"{original_markdown}\n"
        "【原始 Markdown 结束】\n"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_text},
    ]


def build_quality_check_messages(
    *,
    original_markdown: str,
    refined_markdown: str,
    original_stats: dict[str, int],
    refined_stats: dict[str, int],
) -> list[dict]:
    """
    返回用于 quality_check 的 messages，要求模型输出固定格式以便程序解析。
    """
    system = (
        "你是文档质量评估助手。你需要判断 refined_markdown 是否比原始版本更可靠，"
        "并决定是否需要对 Docling 进行 rerun（例如提升 OCR 精度或表格识别设置）。"
        "你必须遵循输出格式要求，只输出一段纯文本标签块（不输出解释、不输出 Markdown）。"
    )

    user_text = (
        "请比较以下两份 Markdown（原始 vs refine），并输出质量判定。\n\n"
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
        "输出格式（严格按键名输出，换行即分隔）：\n"
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
        "你是文档表格纠错助手。你的目标是修复 OCR 导致的单元格错字/漏字。"
        "必须严格保持表格骨架：行数和列数必须与输入完全一致；"
        "只允许修改单元格文本内容，禁止新增/删除行或列。"
        "若用户消息中附有页面截图，请对照图像逐格核对，尤其最右侧列与靠边线格内的勾选/对钩（✓、√），补全漏识的符号。"
        "输出只能包含修正后的原表格 Markdown，不要输出任何解释或包裹在代码块。"
    )

    user_text = (
        "【表格纠错上下文】\n"
        f"{context_text}\n\n"
        "【原始表格 Markdown】\n"
        f"{original_table_markdown}\n\n"
        "请输出修正后的表格 Markdown。"
        "（若本消息含页面图像，请对照图像核对每一格，勿漏读右侧勾选。）"
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
        "你是工业文档表格转述助手。"
        "你的任务不是总结、评价、解释背景，也不是写空泛说明，"
        "而是把表格中的行、列、项目及其对应关系，准确转写成便于阅读的自然语言。"
        "要求："
        "1. 输出必须严格依据表格和附近上下文中明确可见的信息，不得编造、补充表外知识或加入个人判断。"
        "2. 要准确转述表中各项目与各列对象之间的对应关系，尤其要保留“不同类别对应不同要求/状态/取值”的信息。"
        "3. 如果表格中出现符号、标记或简称，且其含义可由表内或上下文明确判断，应将其翻译为文字表述；不要直接照抄符号。"
        "4. 对表示要求等级、是否必选、是否可选、是否适用这类信息时，要按原表含义转述，如“应监测项目”“可监测项目”“适用”“不适用”，不得含混写成“均设置”“均包含”等会混淆原义的说法。"
        "5. 优先采用连续行文直接转述，可使用“包括……”“其中……”“在A中为……，在B中为……”“……分别为……”等句式。"
        "6. 允许用一句简短引导句概括表格结构，如“表中列出了……”，但不得写空泛套话，不得解释用途、意义或背景。"
        "7. 如果表格本质上是矩阵对应表、参数表、限值表、清单表、分类对比表，应逐项交代主要项目及其对应内容，不要只做笼统概括。"
        "8. 输出应让读者在不看表的情况下，直接获得表中的主要项目及其对应内容。"
        "9. 输出为一段简洁、连贯的中文，不要代码块，不要小标题，不要项目符号。"
        f"10. 总长度不超过 {max_chars} 个中文字符。"
    )

    user_text = (
        "【表格附近上下文】\n"
        f"{context_text}\n\n"
        "【表格 Markdown】\n"
        f"{table_markdown}\n\n"
        "请将这个表格准确转述成一段便于阅读的中文文字。"
        "不要总结表外信息，不要解释用途，不要做评价。"
        "要把表中的主要项目、分类以及它们之间的对应关系写清楚；"
        "如果表中符号表示“必须/可选/适用/不适用”等含义，请用文字准确表达，不要直接输出符号。"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_text},
    ]



