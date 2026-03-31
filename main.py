"""
Batch entry: mirror ``data/input/**`` into ``data/output/**`` as Markdown.

多模型对比：在 ``--output-dir`` 下按模型分子目录（``--output-by-model``），例如
``data/output/qwen3-vl-plus/...`` 与 ``data/output/qwen3.5-plus/...``。

常用（PDF 按页 VL 转写，百炼；模型与 DPI/并发等已有 CLI 默认值，可按需省略）::

    python main.py --pdf-vl-primary --output-by-model --pdf-caption-crop-figures --max-files 1 --max-num-pages 5
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", override=False)

from config import env_bool, env_float, env_int, env_str
DEFAULT_LOG_FILE = ROOT / "data" / "output" / "convert.log"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

from src.cli_pdf_vl_defaults import apply_pdf_vl_cli_defaults
from src.converter import ConverterConfig, IndustrialDocConverter


def _sanitize_model_dir(name: str) -> str:
    """将模型名转为可用作目录名的片段（跨平台安全）。"""
    s = str(name).strip()
    for bad in '/\\:*?"<>|':
        s = s.replace(bad, "_")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"_+", "_", s).strip("._")
    return s[:120] if s else "model"


def _effective_output_dir(args: argparse.Namespace, output_root: Path) -> Path:
    """
    若开启 --output-by-model：在 output_root 下按模型名（或纯 Docling 基线 docling/）分子目录。
    """
    if not bool(getattr(args, "output_by_model", False)):
        return output_root
    if args.enable_llm or args.pdf_vl_primary:
        return output_root / _sanitize_model_dir(str(args.llm_model))
    return output_root / "docling"


def _configure_logging(log_file: Path | None, verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s · %(levelname)-7s · %(name)s · %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Industrial batch conversion: mixed formats → Markdown (Docling + pandas/XLSX)."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "data" / "input",
        help="原始文件根目录（内含 pdf/word/excel/ppt/images 等子目录）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "output",
        help="Markdown 与抽取图片输出根目录；配合 --output-by-model 时作为父目录，其下按模型名再分子目录",
    )
    parser.add_argument(
        "--output-by-model",
        action="store_true",
        help=(
            "启用后在 --output-dir 下按模型分类输出："
            "使用 --enable-llm 或 --pdf-vl-primary 时子目录名为 --llm-model 的净化名；"
            "否则为子目录 docling（仅 Docling/表格等，未走大模型）。便于多模型质量对比。"
        ),
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=DEFAULT_LOG_FILE,
        help="日志文件路径",
    )
    parser.add_argument(
        "--no-log-file",
        action="store_true",
        help="不写文件日志（仅控制台）",
    )
    parser.add_argument("--no-ocr", action="store_true", help="关闭 OCR（扫描件/图片将效果较差）")
    parser.add_argument(
        "--ocr-engine",
        choices=["easyocr", "tesseract"],
        default="easyocr",
        help="OCR 后端（需已安装对应依赖与语言包）",
    )
    parser.add_argument(
        "--formula",
        action="store_true",
        help="启用公式/LaTeX 富集（加载 CodeFormula 模型，内存与耗时常显著增加；大 PDF 建议勿开）",
    )
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help="低内存模式：强制关闭公式、缩放<=1、不生成整页/嵌入图、表格用快速模式",
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="扫描版 PDF：按 --ocr-quality 分档压低/提高渲染分辨率，并收紧管线并发；仍 OOM 可加 --no-tables 或 --ocr-quality fast",
    )
    parser.add_argument(
        "--ocr-quality",
        choices=["fast", "balanced", "high"],
        default="balanced",
        metavar="MODE",
        help="识别质量：fast=省内存；balanced=默认；high=更高分辨率+Table ACCURATE+更宽松 OCR 阈值（扫描件可试 high，更吃显存/内存）",
    )
    parser.add_argument(
        "--scan-max-scale",
        type=float,
        default=None,
        metavar="F",
        help="仅 --scan：覆盖档位默认的页面渲染倍率上限（如 1.25~1.5 提高清晰度，内存/显存不足勿用）",
    )
    parser.add_argument(
        "--ocr-confidence",
        type=float,
        default=None,
        metavar="P",
        help="覆盖 EasyOCR 置信度阈值 0~1；越低越不易漏字，可能多噪（默认按 --ocr-quality）",
    )
    parser.add_argument(
        "--ocr-bitmap-threshold",
        type=float,
        default=None,
        metavar="P",
        help="EasyOCR 位图占页面积比例阈值（默认 0.05；略降如 0.02~0.03 可让更多小块参与 OCR）",
    )
    parser.add_argument(
        "--no-tables",
        action="store_true",
        help="禁用表格结构识别（TableFormer），显著降内存，表格效果会变差",
    )
    parser.add_argument(
        "--rich-images",
        action="store_true",
        help="高分辨率整页/配图导出（images_scale=2 + 生成整页图，内存占用大）",
    )
    parser.add_argument(
        "--images-scale",
        type=float,
        default=1.0,
        metavar="F",
        help="页面/图像缩放系数（默认 1.0；与 --rich-images 互斥时以低内存/高画质标志为准）",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="最多处理多少个文件；0 表示不限制",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="若目标 .md 已存在则跳过该文件，不重复转换",
    )
    parser.add_argument(
        "--max-file-size",
        type=int,
        default=0,
        help="单文件最大字节数；0 表示使用 Docling 默认",
    )
    parser.add_argument(
        "--max-num-pages",
        type=int,
        default=0,
        help="PDF/PPTX 等最大页数；0 表示不限制",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="调试日志")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        metavar="NAME",
        help="Docling 模型推理设备：auto | cpu | cuda | cuda:N | mps（默认 cuda；无 NVIDIA 驱动或仅用 CPU 版 PyTorch 时请用 auto/cpu）",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="强制 CPU（等价于 --device cpu）",
    )
    parser.add_argument(
        "--pipeline-concurrency",
        choices=["default", "low", "minimal"],
        default=None,
        metavar="MODE",
        help="PDF 线程管线并发：default=Docling 默认；low=batch≤2、队列≤16；minimal=batch=1、队列≤2。"
        "不传且使用 --scan / --low-memory 时自动为 minimal",
    )
    parser.add_argument(
        "--no-escape-dimension-asterisks",
        action="store_true",
        help="关闭导出后对「数字*数字」类尺寸写法的 * 转义（默认开启，避免 2.0*2.0*2 被 Markdown 当成强调）",
    )
    parser.add_argument(
        "--keep-page-header-footer",
        action="store_true",
        help="保留 Docling 识别为页眉/页脚的文本块（默认在导出 Markdown 时排除）",
    )

    # ---- LLM post-process（DashScope/Qwen-VL）----
    parser.add_argument(
        "--enable-llm",
        action="store_true",
        help="启用百炼千问（Qwen-VL）对 Docling 输出 Markdown 的清洗/纠错。",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=env_str("LLM_MODEL", "qwen3.5-35b-a3b"),
        help="千问模型名（pdf-vl / LLM 共用；默认与项目常用配置一致）。",
    )
    parser.add_argument(
        "--llm-base-url",
        type=str,
        default=env_str(
            "LLM_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        metavar="URL",
        help=(
            "百炼 API 根地址。国内 OpenAI 兼容模式默认：…/compatible-mode/v1；"
            "原生 DashScope：…/api/v1；国际兼容：…dashscope-intl…/compatible-mode/v1。"
        ),
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=env_float("LLM_TEMPERATURE", 0.2),
        help="LLM 采样温度（--pdf-vl-primary 且未显式传参时默认为 0）。",
    )
    parser.add_argument(
        "--llm-max-tokens",
        type=int,
        default=env_int("LLM_MAX_TOKENS", 8192),
        help=(
            "LLM max_tokens（单次 completion 上限）。--pdf-vl-primary 且未显式传参时默认为 16384。"
        ),
    )
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument(
        "--llm-enable-thinking",
        dest="llm_enable_thinking",
        action="store_true",
        help=(
            "启用 Qwen3.5 等模型的思考模式（默认已开启，利于质量；"
            "正文应走 content，思考走 reasoning_content，会额外消耗 completion token）。"
        ),
    )
    thinking_group.add_argument(
        "--llm-disable-thinking",
        dest="llm_enable_thinking",
        action="store_false",
        help="关闭思考模式（节省 token，可能降低复杂版式/表格表现）。",
    )
    parser.set_defaults(llm_enable_thinking=env_bool("LLM_ENABLE_THINKING", True))
    parser.add_argument(
        "--llm-empty-content-retries",
        type=int,
        default=env_int("LLM_EMPTY_CONTENT_MAX_ATTEMPTS", 3),
        metavar="N",
        help=(
            "OpenAI 兼容接口下 content 为空（含仅 reasoning 有字）时最多请求次数（默认 3）；"
            "用尽仍空则报错，不接受空正文。"
        ),
    )
    parser.add_argument(
        "--llm-log-stream-response",
        dest="llm_log_stream_response",
        action=argparse.BooleanOptionalAction,
        default=env_bool("LLM_LOG_STREAM_RESPONSE", False),
        help=(
            "OpenAI 兼容 /chat/completions 使用流式响应，并在日志中实时打印模型输出（"
            "正文为 LLM stream out，思考链为 LLM stream reasoning/thinking）。需 LLM_BASE_URL 为兼容地址。"
        ),
    )
    parser.add_argument(
        "--llm-allow-rerun",
        action="store_true",
        help="允许根据质量检查结果对 Docling 进行最多 1 次 rerun（提升 OCR/表格精度）。",
    )
    parser.add_argument(
        "--llm-vl-image-mode",
        type=str,
        choices=["local_abs", "url"],
        default=env_str("LLM_VL_IMAGE_MODE", "local_abs"),
        help="给 Qwen-VL 的图片输入方式：local_abs 或 url。",
    )

    parser.add_argument(
        "--llm-table-refine",
        action="store_true",
        help="表格为主时更有效：对每个 Markdown table block 单独调用 Qwen-VL 修正单元格错字/漏字。",
    )
    parser.add_argument(
        "--llm-table-max-tables",
        type=int,
        default=env_int("LLM_TABLE_CLEANUP_MAX_TABLES", 10),
        help="LLM 表格局部纠错最多处理多少个 table block。",
    )
    parser.add_argument(
        "--llm-table-max-images-per-table",
        type=int,
        default=env_int("LLM_TABLE_CLEANUP_MAX_IMAGES_PER_TABLE", 6),
        help="每个 table block 最多喂给 Qwen-VL 的图片张数（从 Markdown 中提取的 image refs 里选取）。",
    )
    parser.add_argument(
        "--llm-table-context-lines",
        type=int,
        default=env_int("LLM_TABLE_CONTEXT_LINES", 2),
        help="每个 table block 前最多取多少行作为上下文文本。",
    )

    # ---- PDF 方案 A：按页 VL 转写（验证）----
    parser.add_argument(
        "--pdf-vl-primary",
        action="store_true",
        help=(
            "仅对 PDF：按页渲染为图片，逐页调用 Qwen-VL 转写为 Markdown（需可用的 API 凭证，默认读环境变量配置）。"
            "不经过 Docling；与 --enable-llm 可同时用于对整篇结果再清洗。"
        ),
    )
    parser.add_argument(
        "--pdf-vl-dpi",
        type=float,
        default=env_float("PDF_VL_DPI", 180.0),
        metavar="DPI",
        help="pdf-vl-primary 渲染 DPI（约 100~200；未显式传参时默认 180）。",
    )
    parser.add_argument(
        "--pdf-vl-workers",
        type=int,
        default=env_int("PDF_VL_WORKERS", 10),
        metavar="N",
        help="pdf-vl-primary 页级并发（默认 10；可按配额下调）。",
    )
    parser.add_argument(
        "--no-pdf-vl-table-second-pass",
        action="store_true",
        help="关闭 pdf-vl 的可疑表格二次 LLM 校对（默认开启）。",
    )
    parser.add_argument(
        "--pdf-vl-table-second-pass-max-tables",
        type=int,
        default=env_int("PDF_VL_TABLE_SECOND_PASS_MAX_TABLES", 5),
        metavar="N",
        help="每页最多二次校对多少个可疑表格（默认 5）。",
    )

    parser.add_argument(
        "--llm-table-caption",
        dest="llm_table_caption",
        action="store_true",
        default=True,
        help="为每个 Markdown 表格生成表格信息转述，并插入到表格下方。",
    )
    parser.add_argument(
        "--no-llm-table-caption",
        dest="llm_table_caption",
        action="store_false",
        help="关闭表格信息转述。",
    )

    parser.add_argument(
        "--llm-table-caption-max-tables",
        type=int,
        default=100,
        help="每个文档最多为多少个表格生成语义补偿。",
    )
    parser.add_argument(
        "--llm-table-caption-max-chars",
        type=int,
        default=5000,
        help="每个表格说明的最大字符数。",
    )
    parser.add_argument(
        "--llm-table-caption-context-lines",
        type=int,
        default=env_int("LLM_TABLE_CAPTION_CONTEXT_LINES", 3),
        help="生成表格说明时，表格前后各带多少行上下文。",
    )

    parser.add_argument(
        "--pdf-caption-crop-figures",
        action="store_true",
        help="在 pdf-vl-primary 模式下，按 Markdown 中识别到的图题，从扫描页整页渲染图中裁出局部插图并写入 Markdown",
    )
    parser.add_argument(
        "--pdf-caption-crop-max-per-page",
        type=int,
        default=env_int("PDF_CAPTION_CROP_MAX_PER_PAGE", 4),
        metavar="N",
        help="每页最多按图题裁出多少张图（默认 4）",
    )

    #图片语义补偿
    parser.add_argument(
        "--llm-image-caption",
        action="store_true",
        help="对 Markdown 中切出的图片做语义补充，并写回 Markdown。",
    )
    parser.add_argument(
        "--llm-image-caption-max-images",
        type=int,
        default=0,
        help="每个文档最多处理多少张图片；0 表示不限制。",
    )
    parser.add_argument(
        "--llm-image-caption-max-chars",
        type=int,
        default=env_int("LLM_IMAGE_CAPTION_MAX_CHARS", 0),
        help="每张图片语义补充最大字符数；0 表示不主动截断。",
    )
    parser.add_argument(
        "--llm-image-caption-context-lines",
        type=int,
        default=3,
        help="生成图片语义补充时，图片前后各取多少行上下文。",
    )



    args = parser.parse_args()
    apply_pdf_vl_cli_defaults(args, sys.argv)

    output_root = args.output_dir.resolve()
    output_dir = _effective_output_dir(args, output_root)
    if not args.no_log_file:
        if args.output_by_model and args.log_file.resolve() == DEFAULT_LOG_FILE.resolve():
            log_path: Path | None = output_dir / "convert.log"
        else:
            log_path = args.log_file
    else:
        log_path = None
    _configure_logging(log_path, args.verbose)
    log = logging.getLogger("main")

    input_dir: Path = args.input_dir.resolve()
    if not input_dir.is_dir():
        log.error("输入目录不存在: %s", input_dir)
        return 2

    if args.output_by_model:
        log.info("按模型分类输出：父目录=%s，本趟写入=%s", output_root, output_dir)

    device = "cpu" if args.cpu else args.device.strip()
    if device.lower().startswith("cuda"):
        try:
            import torch

            ver = getattr(torch, "__version__", "")
            if "+cpu" in ver:
                # 常见：pip 默认装成 torch x.x.x+cpu，CUDA 永远不可用
                log.warning(
                    "当前 PyTorch 为 CPU 构建（%s）。请先卸载再安装带 CUDA 的轮子，例如：\n"
                    "  python -m pip uninstall -y torch torchvision torchaudio\n"
                    "  python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128\n"
                    "RTX 50 系（Blackwell）建议 cu128 + PyTorch 2.7+。详见 README「GPU / PyTorch」。",
                    ver,
                )
            if not torch.cuda.is_available():
                log.warning(
                    "已请求 CUDA，但 torch.cuda.is_available()=False。"
                    "若已安装 GPU 版 PyTorch，请检查 NVIDIA 驱动；或改用 --device auto / --cpu。"
                )
            else:
                cap = torch.cuda.get_device_capability(0)
                name = torch.cuda.get_device_name(0)
                log.info(
                    "PyTorch CUDA：%s | 驱动侧 CUDA %s | 计算能力 sm_%d%d",
                    name,
                    torch.version.cuda,
                    cap[0],
                    cap[1],
                )
                if cap[0] >= 12:
                    log.info(
                        "提示：Blackwell（sm_12x）等显卡请使用 PyTorch 2.7+ 且 cu128 及以上；"
                        "若仍报架构不兼容或 kernel 错误，请按 README 升级/换 nightly。"
                    )
        except ImportError:
            log.warning("未安装 torch，无法检测 CUDA；若 Docling 报错请检查 PyTorch 安装。")

    if args.scan and args.no_ocr:
        log.warning(
            "已指定 --scan（扫描件），但同时使用 --no-ocr：若无文字层，输出可能几乎为空。"
        )

    ocr_quality = args.ocr_quality
    if args.low_memory and ocr_quality == "high":
        log.warning("低内存模式会强制将 OCR 质量降为 fast（忽略 --ocr-quality high）")
        ocr_quality = "fast"

    rich = args.rich_images and not args.low_memory and not args.scan
    if args.pipeline_concurrency is not None:
        pipe_c = args.pipeline_concurrency
    elif args.scan or args.low_memory:
        pipe_c = "minimal"
    else:
        pipe_c = "default"

    if args.ocr_confidence is not None and not (0.0 <= args.ocr_confidence <= 1.0):
        log.error("--ocr-confidence 须在 0~1 之间")
        return 2
    if args.ocr_bitmap_threshold is not None and not (
        0.0 < args.ocr_bitmap_threshold <= 1.0
    ):
        log.error("--ocr-bitmap-threshold 须在 (0,1] 之间")
        return 2
    if args.scan_max_scale is not None and not (0.2 <= args.scan_max_scale <= 2.0):
        log.error("--scan-max-scale 建议在 0.2~2.0 之间")
        return 2
    if args.scan_max_scale is not None and not args.scan:
        log.warning(
            "--scan-max-scale 仅在 --scan 时生效；当前未使用 --scan，该参数将被忽略"
        )

    if args.pdf_vl_primary and args.pdf_vl_dpi < 72:
        log.warning("--pdf-vl-dpi 过低可能影响识别，建议 100~200")
    if args.pdf_vl_primary and args.pdf_vl_dpi > 400:
        log.warning("--pdf-vl-dpi 过高可能导致请求过大或超时，建议 72~200")
    if args.pdf_vl_workers < 1:
        log.error("--pdf-vl-workers 必须 >= 1")
        return 2
    if args.pdf_vl_workers > 32:
        log.warning("--pdf-vl-workers 过高，建议 <= 10（特殊情况下不超过 32）")
    if args.pdf_vl_table_second_pass_max_tables < 1:
        log.error("--pdf-vl-table-second-pass-max-tables 必须 >= 1")
        return 2
    if not (1 <= int(args.llm_empty_content_retries) <= 10):
        log.error("--llm-empty-content-retries 须在 1~10 之间")
        return 2

    if args.pdf_vl_primary:
        log.info(
            "pdf-vl-primary：dpi=%s workers=%s table_2nd_pass_max=%s llm_model=%s "
            "llm_temperature=%s llm_max_tokens=%s",
            args.pdf_vl_dpi,
            args.pdf_vl_workers,
            args.pdf_vl_table_second_pass_max_tables,
            args.llm_model,
            args.llm_temperature,
            args.llm_max_tokens,
        )

    cfg = ConverterConfig(
        enable_ocr=not args.no_ocr,
        ocr_engine=args.ocr_engine,
        enable_formula_enrichment=bool(args.formula) and not args.low_memory,
        low_memory=args.low_memory,
        scan_pdf_mode=args.scan,
        do_table_structure=not args.no_tables,
        pipeline_concurrency=pipe_c,
        ocr_quality=ocr_quality,
        scan_max_images_scale=args.scan_max_scale,
        easyocr_confidence=args.ocr_confidence,
        easyocr_bitmap_area_threshold=args.ocr_bitmap_threshold,
        markdown_escape_dimension_asterisks=not args.no_escape_dimension_asterisks,
        markdown_exclude_page_header_footer=not args.keep_page_header_footer,
        images_scale=2.0 if rich else args.images_scale,
        generate_page_images=rich,
        generate_picture_images=not args.low_memory,
        max_file_size=args.max_file_size or None,
        max_num_pages=args.max_num_pages or None,
        accelerator_device=device,

        enable_llm_refine=bool(args.enable_llm),
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
        llm_temperature=float(args.llm_temperature),
        llm_max_tokens=args.llm_max_tokens,
        llm_enable_thinking=bool(args.llm_enable_thinking),
        llm_empty_content_max_attempts=int(args.llm_empty_content_retries),
        llm_log_stream_response=bool(args.llm_log_stream_response),
        llm_allow_rerun=bool(args.llm_allow_rerun),
        llm_vl_image_mode=str(args.llm_vl_image_mode),

        #图片语义补偿
        llm_image_caption=bool(args.llm_image_caption),
        llm_image_caption_max_images=int(args.llm_image_caption_max_images),
        llm_image_caption_max_chars=int(args.llm_image_caption_max_chars),
        llm_image_caption_context_lines=int(args.llm_image_caption_context_lines),


        llm_table_refine=bool(args.llm_table_refine),
        llm_table_cleanup_max_tables=int(args.llm_table_max_tables),
        llm_table_cleanup_max_images_per_table=int(
            args.llm_table_max_images_per_table
        ),
        llm_table_context_lines=int(args.llm_table_context_lines),


        llm_table_caption=bool(args.llm_table_caption),
        llm_table_caption_max_tables=int(args.llm_table_caption_max_tables),
        llm_table_caption_max_chars=int(args.llm_table_caption_max_chars),
        llm_table_caption_context_lines=int(args.llm_table_caption_context_lines),

        llm_api_key_env=env_str("LLM_API_KEY_ENV", "DASHSCOPE_API_KEY"),
        llm_max_retries=env_int("LLM_MAX_RETRIES", 3),
        llm_retry_backoff_sec=env_float("LLM_RETRY_BACKOFF_SEC", 1.5),
        llm_max_reasoning_tokens=env_int("LLM_MAX_REASONING_TOKENS", 256),
        llm_cleanup_max_images=env_int("LLM_CLEANUP_MAX_IMAGES", 6),
        llm_rerun_max_attempts=env_int("LLM_RERUN_MAX_ATTEMPTS", 1),

        pdf_vl_primary=bool(args.pdf_vl_primary),
        pdf_vl_dpi=float(args.pdf_vl_dpi),
        pdf_vl_workers=int(args.pdf_vl_workers),
        pdf_vl_table_second_pass=not bool(args.no_pdf_vl_table_second_pass),
        pdf_vl_table_second_pass_max_tables=int(args.pdf_vl_table_second_pass_max_tables),
        pdf_caption_crop_figures=bool(args.pdf_caption_crop_figures),
        pdf_caption_crop_max_per_page=int(args.pdf_caption_crop_max_per_page),

    )
    converter = IndustrialDocConverter(cfg)

    files = list(IndustrialDocConverter.iter_supported_files(input_dir))
    if args.max_files and args.max_files > 0:
        files = files[: args.max_files]

    if not files:
        log.warning("未在 %s 下找到可转换文件（pdf/docx/pptx/html/xlsx/常见图片）", input_dir)
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    ok, failed, skipped = 0, 0, 0
    total = len(files)

    for i, src in enumerate(files, start=1):
        rel = src.relative_to(input_dir)
        md_out = (output_dir / rel).with_suffix(".md")
        pct = 100.0 * i / total
        log.info("[%s/%s] (%.1f%%) %s", i, total, pct, src)
        if args.skip_existing and md_out.is_file():
            log.info("         (跳过，已存在: %s)", md_out)
            skipped += 1
            continue
        try:
            converter.convert_path_to_markdown(src, md_out)
            log.info("         => %s", md_out)
            ok += 1
        except Exception:
            if args.enable_llm:
                log.warning(
                    "LLM 未执行：该文件在 Docling 转换阶段失败（--enable-llm 仅在成功生成 .md 之后调用大模型）。"
                )
            log.exception("         !! 失败: %s", src)
            failed += 1
            converter.invalidate_converter_cache()

    log.info(
        "完成: 成功 %s, 失败 %s, 跳过 %s, 输出目录 %s",
        ok,
        failed,
        skipped,
        output_dir,
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
