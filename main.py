"""
Batch entry: mirror ``data/input/**`` into ``data/output/**`` as Markdown.

一键验证（项目根目录）::

    python main.py --input-dir ./data/input --output-dir ./data/output

仅试转前 1 个文件::

    python main.py --max-files 1
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

from src.converter import ConverterConfig, IndustrialDocConverter


def _argv_contains_flag(argv: list[str], flag: str) -> bool:
    return flag in argv


def apply_cli_profile(args: argparse.Namespace, argv: list[str]) -> None:
    """
    预设「稳定 + 中文表格」管线：降低 preprocess/OOM 概率，并抬高 OCR/TableFormer 档位。

    若用户在命令行显式写了同名参数（如 --ocr-quality fast），则不覆盖。
    """
    prof = getattr(args, "profile", None) or "default"
    if prof == "default":
        return
    av = argv
    if prof in ("stable-chinese", "stable-chinese-llm"):
        has_low_mem = _argv_contains_flag(av, "--low-memory")
        if not has_low_mem:
            if not _argv_contains_flag(av, "--scan"):
                args.scan = True
            if not _argv_contains_flag(av, "--pipeline-concurrency"):
                args.pipeline_concurrency = "minimal"
            if not _argv_contains_flag(av, "--ocr-quality"):
                args.ocr_quality = "high"
            if not _argv_contains_flag(av, "--ocr-bitmap-threshold"):
                args.ocr_bitmap_threshold = 0.03
        else:
            if not _argv_contains_flag(av, "--pipeline-concurrency"):
                args.pipeline_concurrency = "minimal"
    if prof == "stable-chinese-llm":
        if not _argv_contains_flag(av, "--enable-llm"):
            args.enable_llm = True
        if not _argv_contains_flag(av, "--llm-table-refine"):
            args.llm_table_refine = True


def apply_model_tuning(args: argparse.Namespace, argv: list[str]) -> None:
    """
    针对 qwen3.5-plus + pdf-vl-primary 的默认调优（显式传参优先）。
    """
    if not bool(getattr(args, "pdf_vl_primary", False)):
        return
    model_name = str(getattr(args, "llm_model", "")).lower()
    if "qwen3.5-plus" not in model_name:
        return

    if not _argv_contains_flag(argv, "--llm-temperature"):
        args.llm_temperature = 0.0
    if not _argv_contains_flag(argv, "--llm-max-tokens"):
        # 模型单次最大输出约 64K tokens；页级转写适当抬高，减少长页/表格截断
        args.llm_max_tokens = 16384
    if not _argv_contains_flag(argv, "--pdf-vl-workers"):
        # RPM 很高，可并发；默认给到 10（用户可按网速/配额下调）
        args.pdf_vl_workers = 10


def _configure_logging(log_file: Path | None, verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Industrial batch conversion: mixed formats → Markdown (Docling + pandas/XLSX)."
    )
    parser.add_argument(
        "--profile",
        choices=["default", "stable-chinese", "stable-chinese-llm"],
        default="default",
        help=(
            "运行预设：stable-chinese=更稳（扫描模式 + 管线 minimal + OCR high + 小块位图 OCR），"
            "适合扫描件/表格截图类 PDF；stable-chinese-llm=同上并开启 --enable-llm 与 --llm-table-refine（需 "
            "DASHSCOPE_API_KEY）。显式传入的同名参数优先。"
        ),
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
        help="Markdown 与抽取图片输出根目录",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=ROOT / "data" / "output" / "convert.log",
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

    # ---- LLM post-process（DashScope/Qwen-VL）----
    parser.add_argument(
        "--enable-llm",
        action="store_true",
        help="启用百炼千问（Qwen-VL）对 Docling 输出 Markdown 的清洗/纠错。",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="qwen3-vl-plus",
        help="千问模型名（如 qwen3-vl-plus）。",
    )
    parser.add_argument(
        "--llm-base-url",
        type=str,
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        metavar="URL",
        help=(
            "百炼 API 根地址。国内 OpenAI 兼容模式默认：…/compatible-mode/v1；"
            "原生 DashScope：…/api/v1；国际兼容：…dashscope-intl…/compatible-mode/v1。"
        ),
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=0.2,
        help="LLM 采样温度。",
    )
    parser.add_argument(
        "--llm-max-tokens",
        type=int,
        default=8192,
        help=(
            "LLM max_tokens（单次 completion 上限）。qwen3.5-plus 等约可到 64K；"
            "Docling 后清洗默认 8K；与 --pdf-vl-primary 且 qwen3.5-plus 时自动调到 16K（未显式传参时）。"
        ),
    )
    parser.add_argument(
        "--llm-enable-thinking",
        action="store_true",
        help=(
            "启用 Qwen3.5 等模型的思考模式（正文走 content，思考走 reasoning_content，"
            "会额外消耗 completion token；文档转写默认关闭）。"
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
        default="local_abs",
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
        default=10,
        help="LLM 表格局部纠错最多处理多少个 table block。",
    )
    parser.add_argument(
        "--llm-table-max-images-per-table",
        type=int,
        default=6,
        help="每个 table block 最多喂给 Qwen-VL 的图片张数（从 Markdown 中提取的 image refs 里选取）。",
    )
    parser.add_argument(
        "--llm-table-context-lines",
        type=int,
        default=2,
        help="每个 table block 前最多取多少行作为上下文文本。",
    )

    # ---- PDF 方案 A：按页 VL 转写（验证）----
    parser.add_argument(
        "--pdf-vl-primary",
        action="store_true",
        help=(
            "仅对 PDF：按页渲染为图片，逐页调用 Qwen-VL 转写为 Markdown（需 DASHSCOPE_API_KEY）。"
            "不经过 Docling；与 --enable-llm 可同时用于对整篇结果再清洗。"
        ),
    )
    parser.add_argument(
        "--pdf-vl-dpi",
        type=float,
        default=150.0,
        metavar="DPI",
        help="pdf-vl-primary 渲染 DPI（约 100~200；越高越清晰但越慢、请求体越大）。",
    )
    parser.add_argument(
        "--pdf-vl-workers",
        type=int,
        default=1,
        metavar="N",
        help="pdf-vl-primary 页级并发调用数（建议 1~10；qwen3.5-plus 可尝试 10）。",
    )

    args = parser.parse_args()
    apply_cli_profile(args, sys.argv)
    apply_model_tuning(args, sys.argv)

    log_path: Path | None = None if args.no_log_file else args.log_file
    _configure_logging(log_path, args.verbose)
    log = logging.getLogger("main")

    input_dir: Path = args.input_dir.resolve()
    output_dir: Path = args.output_dir.resolve()
    if not input_dir.is_dir():
        log.error("输入目录不存在: %s", input_dir)
        return 2

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

    if args.profile != "default":
        log.info(
            "CLI profile=%s：scan=%s pipeline_concurrency=%s ocr_quality=%s "
            "enable_llm=%s llm_table_refine=%s pdf_vl_primary=%s pdf_vl_workers=%s "
            "llm_temperature=%s llm_max_tokens=%s",
            args.profile,
            args.scan,
            args.pipeline_concurrency,
            args.ocr_quality,
            args.enable_llm,
            args.llm_table_refine,
            args.pdf_vl_primary,
            args.pdf_vl_workers,
            args.llm_temperature,
            args.llm_max_tokens,
        )
    if args.profile in ("stable-chinese", "stable-chinese-llm") and args.low_memory:
        log.warning(
            "已使用 --low-memory：OCR 会被强制为 fast，中文表格精度可能不如 profile 预期；"
            "若仍 OOM 可再加 --no-tables。"
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
        llm_allow_rerun=bool(args.llm_allow_rerun),
        llm_vl_image_mode=str(args.llm_vl_image_mode),

        llm_table_refine=bool(args.llm_table_refine),
        llm_table_cleanup_max_tables=int(args.llm_table_max_tables),
        llm_table_cleanup_max_images_per_table=int(
            args.llm_table_max_images_per_table
        ),
        llm_table_context_lines=int(args.llm_table_context_lines),

        pdf_vl_primary=bool(args.pdf_vl_primary),
        pdf_vl_dpi=float(args.pdf_vl_dpi),
        pdf_vl_workers=int(args.pdf_vl_workers),
    )
    converter = IndustrialDocConverter(cfg)

    files = list(IndustrialDocConverter.iter_supported_files(input_dir))
    if args.max_files and args.max_files > 0:
        files = files[: args.max_files]

    if not files:
        log.warning("未在 %s 下找到可转换文件（pdf/docx/pptx/html/xlsx/常见图片）", input_dir)
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    ok, failed = 0, 0
    total = len(files)

    for i, src in enumerate(files, start=1):
        rel = src.relative_to(input_dir)
        md_out = (output_dir / rel).with_suffix(".md")
        pct = 100.0 * i / total
        log.info("[%s/%s] (%.1f%%) %s", i, total, pct, src)
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

    log.info("完成: 成功 %s, 失败 %s, 输出目录 %s", ok, failed, output_dir)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
