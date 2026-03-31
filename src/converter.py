from __future__ import annotations

import logging
import re
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterable, Literal, Optional, Union

import pandas as pd
from PIL import Image

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    PdfPipelineOptions,
    TableStructureOptions,
    TesseractOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption, WordFormatOption
from docling_core.types.doc import ImageRefMode
from docling_core.types.doc.document import DOCUMENT_TOKENS_EXPORT_LABELS
from docling_core.types.doc.labels import DocItemLabel

try:
    from docling.datamodel.pipeline_options import TableFormerMode
except ImportError:  # pragma: no cover - older docling
    TableFormerMode = None  # type: ignore[misc, assignment]

_log = logging.getLogger(__name__)

# Docling 默认 Markdown 导出标签，但排除版面模型识别的页眉/页脚（仍保留正文里的 section_header/title 等）
_MARKDOWN_LABELS_WITHOUT_PAGE_HEADER_FOOTER: set[DocItemLabel] = (
    DOCUMENT_TOKENS_EXPORT_LABELS
    - {DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER}
)

# Extensions routed to Docling (paginated / office / web / image)
_DOCLING_SUFFIXES = {
    ".pdf",
    ".docx",
    ".pptx",
    ".html",
    ".htm",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
}

# Excel handled explicitly via pandas (per-sheet Markdown tables)
_XLSX_SUFFIXES = {".xlsx"}
_MARKDOWN_IMAGE_RE = re.compile(
    r'!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)'
)
_FIGURE_TITLE_RE = re.compile(
    r"^\s*(图\s*[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*(?:-\d+)?(?:[：: \t].*)?)\s*$"
)

_FIGURE_ID_RE = re.compile(
    r"(图\s*[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*(?:-\d+)?)"
)



# EasyOCR: ISO 639-1；Tesseract: ISO 639-2 / tesseract 语言名
_EASYOCR_TO_TESSERACT_LANG = {
    "en": "eng",
    "ch_sim": "chi_sim",
    "zh": "chi_sim",
    "chi_sim": "chi_sim",
}


def _tesseract_lang_list(codes: list[str]) -> list[str]:
    out: list[str] = []
    for c in codes:
        key = c.strip().lower()
        out.append(_EASYOCR_TO_TESSERACT_LANG.get(key, c))
    return out


@dataclass
class ConverterConfig:
    """Runtime tuning for Docling pipelines and export."""

    enable_ocr: bool = True
    ocr_engine: str = "easyocr"  # easyocr | tesseract
    ocr_languages: list[str] = field(
        default_factory=lambda: ["en", "ch_sim"]
    )
    # 默认关闭：会拉取 CodeFormula VLM，CPU 上极易 OOM，且与大图 PDF 叠加易触发 std::bad_alloc
    enable_formula_enrichment: bool = False
    table_structure_accurate: bool = True
    document_timeout_sec: Optional[float] = 300.0
    # 1.0 为 Docling 推荐默认值；2.0 + generate_page_images 对大页数 PDF 极易内存爆炸
    images_scale: float = 1.0
    generate_page_images: bool = False
    generate_picture_images: bool = True
    # 低内存：进一步关掉嵌入图抽取、表格用 FAST（若库支持）
    low_memory: bool = False
    # 扫描版 PDF（整页位图为主）：强制压低渲染倍率、收紧 threaded 管线队列
    scan_pdf_mode: bool = False
    # 关闭 TableFormer，可明显降低内存（表格结构变差）
    do_table_structure: bool = True
    # StandardPdfPipeline 线程阶段并发：default=Docling 默认；low=batch≤2、队列收紧；minimal=batch=1、队列≤2（最省内存）
    pipeline_concurrency: str = "default"
    max_file_size: Optional[int] = None
    max_num_pages: Optional[int] = None
    accelerator_num_threads: int = 4
    # Docling 内置模型（版面/表格/公式等）推理设备：cuda | cuda:0 | cpu | auto | mps
    accelerator_device: Union[str, None] = "cuda"
    # None 表示随 accelerator_device 推断（cuda* -> True）
    easyocr_use_gpu: Optional[bool] = None
    # OCR/版面识别质量：fast=省内存；balanced=默认；high=更高分辨率 + Table ACCURATE + OCR 阈值/整页 OCR 等（更慢、更吃显存/内存）
    ocr_quality: str = "balanced"
    # 扫描件专用：覆盖档位默认的 images_scale 上限（如 1.25~1.5），None 表示按 ocr_quality 档位
    scan_max_images_scale: Optional[float] = None
    # 覆盖 EasyOCR 置信度阈值（0~1），None 表示按档位自动
    easyocr_confidence: Optional[float] = None
    # 位图占页面积比例阈值，越小越容易对小块图做 OCR（扫描件可略降）
    easyocr_bitmap_area_threshold: Optional[float] = None
    # 将「长宽高/尺寸」类写法中的 * 转义为 \*（全文，不仅表格），避免 2.0*2.0*2 被当成 Markdown 强调
    markdown_escape_dimension_asterisks: bool = True
    # 导出 Markdown 时排除 DocItemLabel.PAGE_HEADER / PAGE_FOOTER（依赖版面模型标注）
    markdown_exclude_page_header_footer: bool = True

    # ---- LLM post-process（DashScope 千问/Qwen-VL）----
    enable_llm_refine: bool = False
    llm_api_key_env: str = "DASHSCOPE_API_KEY"
    llm_model: str = "qwen3.5-35b-a3b"
    llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm_temperature: float = 0.2
    llm_max_tokens: Optional[int] = 8192
    llm_timeout_sec: float = 300.0
    llm_max_retries: int = 3
    llm_retry_backoff_sec: float = 1.5
    llm_max_reasoning_tokens: Optional[int] = 256
    llm_vl_image_mode: Literal["local_abs", "url"] = "local_abs"
    llm_cleanup_max_images: int = 6
    # 表格优先：对每个 table block 单独调用 Qwen-VL 修正单元格错字/漏字
    llm_table_refine: bool = False
    llm_table_cleanup_max_tables: int = 10
    llm_table_cleanup_max_images_per_table: int = 6
    llm_table_context_lines: int = 2
    llm_allow_rerun: bool = False
    llm_rerun_max_attempts: int = 1
    # Qwen3.5：思考模式会占用 completion token；默认开启以保障版式/表格质量（extra_body.enable_thinking）
    llm_enable_thinking: bool = True
    # OpenAI 兼容：content 为空（含仅 reasoning）时最多请求次数，用尽仍空则抛错（见 DashScopeClient）
    llm_empty_content_max_attempts: int = 3
    # True：OpenAI 兼容 /chat/completions 流式拉取并在日志中实时打印模型输出（环境变量 LLM_LOG_STREAM_RESPONSE）
    llm_log_stream_response: bool = False

    llm_table_caption: bool = False
    llm_table_caption_max_tables: int = 20
    llm_table_caption_max_chars: int = 500
    llm_table_caption_context_lines: int = 3

    #图片语义补偿
    llm_image_caption: bool = False
    llm_image_caption_max_images: int = 0
    llm_image_caption_max_chars: int = 0
    llm_image_caption_context_lines: int = 3
    llm_image_caption_min_width: int = 0
    llm_image_caption_min_height: int = 0
    llm_image_caption_min_area: int = 0


    # ---- PDF 方案 A：按页渲染 + Qwen-VL 转写（不经过 Docling）----
    pdf_vl_primary: bool = False
    pdf_vl_dpi: float = 180.0
    pdf_vl_workers: int = 10
    pdf_vl_table_second_pass: bool = True
    pdf_vl_table_second_pass_max_tables: int = 5
    # 是否在 markdown 中嵌入每页渲染图（默认关闭，避免把整页截图当“插图”）
    pdf_vl_embed_page_images: bool = False
    pdf_caption_crop_figures: bool = False
    pdf_caption_crop_max_per_page: int = 4



def _escape_dimension_like_asterisks(text: str) -> str:
    """
    将易被误解析为强调的尺寸链中的 * 转为 \\*。

    覆盖：2.0*2.0*2、(2*3)、20m*30、2.0 * 2.0 * 2 等；不处理 **加粗**、* 列表 等常见语法
    （* 前一般为数字、小数点、括号或 m/M 等单位后缀）。
    """
    if not text:
        return text
    if text.endswith("\r\n"):
        body, nl_suffix = text[:-2], "\r\n"
    elif text.endswith("\n"):
        body, nl_suffix = text[:-1], "\n"
    else:
        body, nl_suffix = text, ""
    s = body
    # 含空格：2.0 * 2.0 * 2.0
    s = re.sub(
        r"((?:\d+\.\d+|\d+))\s+\*(?=\s*\d)",
        r"\1 \\*",
        s,
    )
    # 无空格，前接数字/小数点/括号/方括号：2.0*2.0*2、(2*3)
    s = re.sub(r"(?<=[\d.(\[{])(?<!\\)\*(?=[\d])", r"\\*", s)
    # 无空格，前接 m/M（如 20m*30、100cm*50 中 cm 末尾的 m）
    s = re.sub(r"(?<=[mM])(?<!\\)\*(?=[\d])", r"\\*", s)
    return s + nl_suffix


class IndustrialDocConverter:
    """
    Unified conversion to Markdown with Docling (PDF/DOCX/PPTX/HTML/images)
    plus pandas-based XLSX handling (one Markdown table per sheet).
    """

    def __init__(self, config: Optional[ConverterConfig] = None) -> None:
        self.config = config or ConverterConfig()
        self._converter: Optional[DocumentConverter] = None
        # 仅 pdf-vl-primary 路径在 convert_path_to_markdown 末尾写入
        self._last_pdf_vl_failed_pages: list[int] | None = None

    @property
    def last_pdf_vl_failed_pages(self) -> list[int] | None:
        """最近一次 pdf-vl 转换中异常页的 1-based 页码；非该路径为 None。"""
        return self._last_pdf_vl_failed_pages

    def invalidate_converter_cache(self) -> None:
        """单文件失败后丢弃已缓存的 DocumentConverter，减轻 pypdfium2 异常清理告警。"""
        self._converter = None

    def _effective_ocr_quality(self) -> str:
        if self.config.low_memory:
            return "fast"
        q = (self.config.ocr_quality or "balanced").lower().strip()
        if q not in ("fast", "balanced", "high"):
            return "balanced"
        return q

    def _normalize_figure_title(self, text: str) -> str:
        s = (text or "").strip()
        s = re.sub(r"\s+", " ", s)
        return s.strip("：: ").strip()
    

    def _resolve_image_label(
        self,
        markdown_text: str,
        *,
        line_no: int,
        alt_text: str = "",
    ) -> str:
        lines = markdown_text.splitlines()

        def _extract_id(s: str) -> str:
            s = (s or "").strip()
            s = re.sub(r"\s+", " ", s)
            s = s.strip(">*- ")
            s = re.sub(r"^\*\*(.*?)\*\*$", r"\1", s)

            m = _FIGURE_ID_RE.search(s)
            if m:
                return m.group(1).strip()
            return ""

        # 1) 优先用 alt
        label = _extract_id(alt_text)
        if label:
            return label

        # 2) 再看图片前后更多行
        for k in range(1, 9):
            if line_no - k >= 0:
                label = _extract_id(lines[line_no - k])
                if label:
                    return label

            if line_no + k < len(lines):
                label = _extract_id(lines[line_no + k])
                if label:
                    return label

        return "图像补充"



    def _apply_docling_global_memory_settings(self) -> None:
        """降低 base_pipeline 每批初始化页数与元素批大小（全局 settings）。"""
        pc = self.config.pipeline_concurrency
        if pc == "default" and not (
            self.config.scan_pdf_mode or self.config.low_memory
        ):
            return
        from docling.datamodel.settings import settings

        settings.perf.page_batch_size = 1
        if pc == "minimal" or self.config.scan_pdf_mode or self.config.low_memory:
            settings.perf.elements_batch_size = min(settings.perf.elements_batch_size, 2)
        else:
            settings.perf.elements_batch_size = min(settings.perf.elements_batch_size, 4)

    def _build_pdf_pipeline_options(self) -> PdfPipelineOptions:
        opts = PdfPipelineOptions()
        opts.do_table_structure = self.config.do_table_structure
        opts.table_structure_options = TableStructureOptions(do_cell_matching=True)
        if (
            opts.do_table_structure
            and TableFormerMode is not None
            and self.config.table_structure_accurate
            and hasattr(opts.table_structure_options, "mode")
        ):
            opts.table_structure_options.mode = TableFormerMode.ACCURATE

        opts.do_formula_enrichment = self.config.enable_formula_enrichment

        opts.images_scale = self.config.images_scale
        opts.generate_page_images = self.config.generate_page_images
        opts.generate_picture_images = self.config.generate_picture_images

        if self.config.low_memory:
            opts.do_formula_enrichment = False
            opts.images_scale = min(opts.images_scale, 1.0)
            opts.generate_page_images = False
            opts.generate_picture_images = False
            if TableFormerMode is not None and hasattr(
                opts.table_structure_options, "mode"
            ):
                opts.table_structure_options.mode = TableFormerMode.FAST

        if self.config.scan_pdf_mode:
            # 分辨率越高 OCR/版面越好，但内存越大；由 ocr_quality 分档
            opts.generate_page_images = False
            opts.generate_picture_images = False
            opts.do_formula_enrichment = False
            q = self._effective_ocr_quality()
            if q == "fast":
                tier_cap = 0.25
                tbl_mode = TableFormerMode.FAST if TableFormerMode else None
            elif q == "balanced":
                tier_cap = 0.55
                if TableFormerMode is not None:
                    tbl_mode = (
                        TableFormerMode.ACCURATE
                        if self.config.table_structure_accurate
                        else TableFormerMode.FAST
                    )
                else:
                    tbl_mode = None
            else:
                tier_cap = 1.0
                tbl_mode = TableFormerMode.ACCURATE if TableFormerMode else None
            cap = (
                self.config.scan_max_images_scale
                if self.config.scan_max_images_scale is not None
                else tier_cap
            )
            cap = max(0.2, min(2.0, float(cap)))
            opts.images_scale = min(opts.images_scale, cap)
            if (
                opts.do_table_structure
                and tbl_mode is not None
                and hasattr(opts.table_structure_options, "mode")
            ):
                opts.table_structure_options.mode = tbl_mode

        if (
            not self.config.scan_pdf_mode
            and self._effective_ocr_quality() == "high"
            and not self.config.low_memory
        ):
            opts.images_scale = max(opts.images_scale, 1.0)

        self._apply_pipeline_concurrency(opts)

        opts.do_ocr = self.config.enable_ocr
        if self.config.enable_ocr:
            engine = self.config.ocr_engine.lower().strip()
            if engine == "tesseract":
                to = TesseractOcrOptions(
                    lang=_tesseract_lang_list(
                        self.config.ocr_languages or ["eng", "chi_sim"]
                    )
                )
                q = self._effective_ocr_quality()
                if q == "high":
                    to.psm = 6
                elif q == "fast":
                    to.psm = 3
                opts.ocr_options = to
            else:
                dev_s = (
                    str(self.config.accelerator_device).lower()
                    if self.config.accelerator_device
                    else ""
                )
                use_gpu = self.config.easyocr_use_gpu
                if use_gpu is None:
                    use_gpu = dev_s.startswith("cuda")
                opts.ocr_options = self._make_easyocr_options(use_gpu)

        if self.config.document_timeout_sec is not None:
            opts.document_timeout = self.config.document_timeout_sec

        dev: Union[str, None] = self.config.accelerator_device
        if dev is None or (isinstance(dev, str) and dev.strip() == ""):
            dev = "auto"
        threads = self.config.accelerator_num_threads
        if self.config.pipeline_concurrency == "minimal":
            threads = min(threads, 2)
        elif self.config.pipeline_concurrency == "low":
            threads = min(threads, 3)
        opts.accelerator_options = AcceleratorOptions(
            num_threads=threads,
            device=dev,
        )
        return opts

    def _apply_pipeline_concurrency(self, opts: PdfPipelineOptions) -> None:
        """降低 OCR/layout/table 各阶段 batch 与阶段间队列深度，避免大页 PDF 并发堆积吃爆内存。"""
        pc = self.config.pipeline_concurrency
        if pc == "default":
            return
        if pc == "low":
            opts.ocr_batch_size = min(opts.ocr_batch_size, 2)
            opts.layout_batch_size = min(opts.layout_batch_size, 2)
            opts.table_batch_size = min(opts.table_batch_size, 2)
            opts.queue_max_size = min(opts.queue_max_size, 16)
        elif pc == "minimal":
            opts.ocr_batch_size = 1
            opts.layout_batch_size = 1
            opts.table_batch_size = 1
            opts.queue_max_size = min(opts.queue_max_size, 2)

    def _make_easyocr_options(self, use_gpu: bool) -> EasyOcrOptions:
        q = self._effective_ocr_quality()
        langs = list(self.config.ocr_languages or ["en", "ch_sim"])
        # 中英混排时中文优先，减轻表格/正文里中文被英文模型误切分
        if q == "high" and "ch_sim" in langs and "en" in langs:
            langs = ["ch_sim", "en"]
        kw: dict = {"lang": langs, "use_gpu": use_gpu}
        if self.config.easyocr_confidence is not None:
            kw["confidence_threshold"] = float(self.config.easyocr_confidence)
        elif q == "high":
            # 不使用 recog_network="craft"：EasyOCR 默认不附带 craft.yaml，会触发 FileNotFoundError；
            # 高精度主要依赖 images_scale / TableFormer / 阈值与语言顺序。
            # 含简体中文时略降阈值，减少表格小字漏检（可能略增噪点）。
            kw["confidence_threshold"] = 0.30 if "ch_sim" in langs else 0.32
        elif q == "balanced":
            kw["confidence_threshold"] = 0.38
        else:
            kw["confidence_threshold"] = 0.52
        if q in ("high", "balanced"):
            kw["force_full_page_ocr"] = bool(self.config.scan_pdf_mode)
        if self.config.easyocr_bitmap_area_threshold is not None:
            kw["bitmap_area_threshold"] = float(
                self.config.easyocr_bitmap_area_threshold
            )
        elif q == "high" and (
            self.config.scan_pdf_mode
            or (self.config.do_table_structure and self.config.table_structure_accurate)
        ):
            # 小块位图（含表内截图）也参与 OCR，减轻漏字
            kw["bitmap_area_threshold"] = 0.03
        return EasyOcrOptions(**kw)

    def _get_converter(self) -> DocumentConverter:
        if self._converter is None:
            pdf_opts = self._build_pdf_pipeline_options()
            pdf_fmt = PdfFormatOption(pipeline_options=pdf_opts)
            self._converter = DocumentConverter(
                allowed_formats=[
                    InputFormat.PDF,
                    InputFormat.DOCX,
                    InputFormat.PPTX,
                    InputFormat.HTML,
                    InputFormat.IMAGE,
                ],
                format_options={
                    InputFormat.PDF: pdf_fmt,
                    # DOCX：显式注册 Word 管线（与 Docling 默认一致）
                    InputFormat.DOCX: WordFormatOption(),
                    InputFormat.PPTX: pdf_fmt,
                    InputFormat.IMAGE: pdf_fmt,
                },
            )
        return self._converter

    def _create_llm_refiner(self) -> Optional["DoclingMarkdownRefiner"]:
        """
        创建 Qwen-VL refiner。

        这里用可选导入避免没有 httpx 依赖时直接崩溃：
        - 若用户未开启 `enable_llm_refine`，该方法不会被调用。
        """
        if not self.config.enable_llm_refine:
            return None

        try:
            from src.dashscope_client import DashScopeClient, DashScopeClientConfig
            from src.llm_markdown_refiner import DoclingMarkdownRefiner
        except Exception as e:  # noqa: BLE001
            _log.error("Import llm modules failed: %s", repr(e))
            return None

        api_key = os.getenv(self.config.llm_api_key_env) or ""

        client_cfg = DashScopeClientConfig(
            api_key=api_key,
            base_url=self.config.llm_base_url,
            timeout_sec=self.config.llm_timeout_sec,
            max_retries=self.config.llm_max_retries,
            retry_backoff_sec=self.config.llm_retry_backoff_sec,
            enable_thinking=self.config.llm_enable_thinking,
            max_reasoning_tokens=self.config.llm_max_reasoning_tokens,
            empty_content_max_attempts=self.config.llm_empty_content_max_attempts,
            log_stream_response=self.config.llm_log_stream_response,
        )
        client = DashScopeClient(client_cfg)
        return DoclingMarkdownRefiner(
            client=client,
            model=self.config.llm_model,
            cleanup_max_images=self.config.llm_cleanup_max_images,
            vl_image_mode=self.config.llm_vl_image_mode,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_tokens,
        )

    def _create_dashscope_client(self) -> Optional["DashScopeClient"]:
        """供 pdf-vl-primary 使用；仅需 API Key，不要求 enable_llm_refine。"""
        try:
            from src.dashscope_client import DashScopeClient, DashScopeClientConfig
        except Exception as e:  # noqa: BLE001
            _log.error("Import dashscope_client failed: %s", repr(e))
            return None
        api_key = os.getenv(self.config.llm_api_key_env) or ""
        client_cfg = DashScopeClientConfig(
            api_key=api_key,
            base_url=self.config.llm_base_url,
            timeout_sec=self.config.llm_timeout_sec,
            max_retries=self.config.llm_max_retries,
            retry_backoff_sec=self.config.llm_retry_backoff_sec,
            enable_thinking=self.config.llm_enable_thinking,
            max_reasoning_tokens=self.config.llm_max_reasoning_tokens,
            empty_content_max_attempts=self.config.llm_empty_content_max_attempts,
            log_stream_response=self.config.llm_log_stream_response,
        )
        return DashScopeClient(client_cfg)

    def _apply_llm_refine_once_to_file(
        self,
        markdown_out: Path,
        refiner: "DoclingMarkdownRefiner",
    ) -> None:
        """
        单次 LLM 清洗 + 质检（不触发 Docling rerun）。
        用于 pdf-vl-primary 产出的 Markdown。
        """
        from src.vl_markdown_utils import (
            summarize_markdown_quality,
            validate_image_refs_invariants,
        )

        docling_md_text = markdown_out.read_text(encoding="utf-8")
        try:
            if self.config.llm_table_refine:
                refined_md = refiner.cleanup_tables_per_block(
                    original_markdown=docling_md_text,
                    markdown_out_path=markdown_out,
                    vl_image_mode=self.config.llm_vl_image_mode,
                    cleanup_max_images_per_table=self.config.llm_table_cleanup_max_images_per_table,
                    cleanup_max_tables=self.config.llm_table_cleanup_max_tables,
                    context_lines=self.config.llm_table_context_lines,
                )
            else:
                image_inputs = refiner.prepare_image_inputs(
                    original_md=docling_md_text,
                    markdown_out_path=markdown_out,
                    vl_image_mode=self.config.llm_vl_image_mode,
                    cleanup_max_images=self.config.llm_cleanup_max_images,
                )
                refined_md = refiner.cleanup_markdown(
                    original_markdown=docling_md_text,
                    markdown_out_path=markdown_out,
                    image_inputs=image_inputs,
                )

            refined_md = (refined_md or "").strip()
            if not refined_md:
                _log.warning("LLM cleanup returned empty output; fallback to pdf-vl md")
                refined_md = docling_md_text
            elif len(refined_md) < 50:
                _log.warning(
                    "LLM cleanup output too short (len=%s); fallback to pdf-vl md",
                    len(refined_md),
                )
                refined_md = docling_md_text

            orig_stats = summarize_markdown_quality(docling_md_text)
            refined_stats = summarize_markdown_quality(refined_md)
            if orig_stats.get("table_rows", 0) > 0 and refined_stats.get("table_rows", 0) == 0:
                _log.warning(
                    "LLM cleanup removed table rows; fallback to pdf-vl md (orig_table_rows=%s)",
                    orig_stats.get("table_rows", 0),
                )
                refined_md = docling_md_text

            ok, details = validate_image_refs_invariants(
                original_md=docling_md_text,
                refined_md=refined_md,
            )
            if not ok:
                _log.warning("LLM refined md failed invariants; fallback to pdf-vl: %s", details)
                refined_md = docling_md_text

            markdown_out.write_text(refined_md, encoding="utf-8")

            qc = refiner.quality_check(
                original_markdown=docling_md_text,
                refined_markdown=refined_md,
            )
            _log.info(
                "pdf-vl LLM quality_check score=%s need_rerun=%s（pdf-vl 模式不执行 Docling rerun）",
                qc.score,
                qc.need_rerun,
            )
        except Exception as e:  # noqa: BLE001
            _log.exception("LLM refine failed on pdf-vl output; keep pdf-vl md: %s", repr(e))

    def _apply_llm_rerun_suggestion(
        self,
        *,
        cfg: ConverterConfig,
        suggest: "LlmRerunSuggestion",
    ) -> ConverterConfig:
        """
        将 LLM 给出的 rerun 建议映射成 Docling 配置提升。

        安全策略：
        - 若用户启用 `low_memory` 或 `scan_pdf_mode`，忽略对 generate_page_images 的提升
        - 若用户关闭了 `do_table_structure`，则保持关闭
        """
        # base: only update a subset of fields
        new_cfg = replace(cfg)

        # OCR 精度提升：即便 low_memory= True，后续 _effective_ocr_quality 也会强制 fast。
        new_cfg.ocr_quality = suggest.ocr_quality

        # 表格精度提升
        if cfg.do_table_structure:
            new_cfg.table_structure_accurate = suggest.table_accuracy == "accurate"

        # 生成整页图用于 VL 再校验：仅在允许且内存风险可控时开启
        if (
            suggest.generate_page_images
            and not cfg.low_memory
            and not cfg.scan_pdf_mode
        ):
            new_cfg.generate_page_images = True
            # 整页图生成通常也希望配图可用
            new_cfg.generate_picture_images = True

        # 扫描模式下 opts.images_scale 会被 cap；这里不强制调整以避免 OOM。
        return new_cfg

    @staticmethod
    def xlsx_to_markdown(path: Path) -> str:
        """Each worksheet becomes a level-2 heading plus a GitHub-flavored table."""
        xl = pd.ExcelFile(path, engine="openpyxl")
        parts: list[str] = [f"# {path.stem}\n\n"]
        for sheet in xl.sheet_names:
            df = pd.read_excel(xl, sheet_name=sheet, header=0)
            df = df.fillna("")
            parts.append(f"## Sheet: {sheet}\n\n")
            try:
                table = df.to_markdown(index=False, tablefmt="github")
            except ImportError as e:
                raise RuntimeError(
                    "pandas.to_markdown requires the 'tabulate' package."
                ) from e
            parts.append(table)
            parts.append("\n\n")
        return "".join(parts)

    def convert_path_to_markdown(
        self,
        source: Path,
        markdown_out: Path,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        """
        Convert a single file to Markdown. Images referenced from the .md are
        written beside the output (Docling ``save_as_markdown`` handles layout).
        """
        suffix = source.suffix.lower()
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        self._last_pdf_vl_failed_pages = None

        if suffix in _XLSX_SUFFIXES:
            text = self.xlsx_to_markdown(source)
            if self.config.markdown_escape_dimension_asterisks:
                text = _escape_dimension_like_asterisks(text)
            markdown_out.write_text(text, encoding="utf-8")
            return

        if suffix == ".pdf" and self.config.pdf_vl_primary:
            from src.pdf_vl_transcribe import transcribe_pdf_with_vl

            client = self._create_dashscope_client()
            if client is None:
                raise RuntimeError(
                    "pdf-vl-primary 需要 httpx 可用且 LLM HTTP 客户端可加载（检查依赖与导入错误日志）"
                )
            _log.info(
                "阶段=PDF逐页转写(跳过Docling) dpi=%s workers=%s max_pages=%s",
                self.config.pdf_vl_dpi,
                self.config.pdf_vl_workers,
                self.config.max_num_pages,
            )
            md, pdf_vl_failed = transcribe_pdf_with_vl(
                client=client,
                model=self.config.llm_model,
                pdf_path=source,
                markdown_out_path=markdown_out,
                dpi=self.config.pdf_vl_dpi,
                workers=self.config.pdf_vl_workers,
                embed_page_images=self.config.pdf_vl_embed_page_images,
                caption_crop_figures=self.config.pdf_caption_crop_figures,
                caption_crop_max_per_page=self.config.pdf_caption_crop_max_per_page,
                table_second_pass=self.config.pdf_vl_table_second_pass,
                table_second_pass_max_tables=self.config.pdf_vl_table_second_pass_max_tables,
                max_pages=self.config.max_num_pages,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens,
                progress_callback=progress_callback,
            )
            self._last_pdf_vl_failed_pages = pdf_vl_failed
            if self.config.markdown_escape_dimension_asterisks:
                md = _escape_dimension_like_asterisks(md)
            markdown_out.write_text(md, encoding="utf-8")
            if self.config.enable_llm_refine:
                refiner = self._create_llm_refiner()
                if refiner is not None:
                    self._apply_llm_refine_once_to_file(markdown_out, refiner)

            final_md = markdown_out.read_text(encoding="utf-8")

            if self.config.llm_table_caption:
                final_md = self._append_table_captions_to_markdown(final_md)

            if self.config.llm_image_caption:
                final_md = self._append_image_captions_to_markdown(final_md, markdown_out)

            if self.config.markdown_escape_dimension_asterisks:
                final_md = _escape_dimension_like_asterisks(final_md)

            markdown_out.write_text(final_md, encoding="utf-8")
            return


        if suffix not in _DOCLING_SUFFIXES:
            raise ValueError(f"Unsupported file type: {source}")

        original_cfg = self.config
        refiner = self._create_llm_refiner()
        if self.config.enable_llm_refine:
            if refiner is None:
                _log.warning(
                    "已启用 LLM 清洗，但 refiner 未创建（请检查上文 ERROR，例如 LLM 相关模块导入失败）。"
                    "将仅尝试 Docling 输出。"
                )
            else:
                _log.info(
                    "LLM 已就绪：先完成 Docling 转换并写出 Markdown，再调用大模型；"
                    "若 Docling 在 convert 阶段失败则不会出现 LLM 调用日志。"
                )

        max_attempts = 1
        if refiner is not None and self.config.llm_allow_rerun:
            max_attempts = 1 + max(0, int(self.config.llm_rerun_max_attempts))

        attempt_cfg = self.config
        last_qc_score = None
        for attempt_i in range(max_attempts):
            self.config = attempt_cfg
            if attempt_i > 0:
                # rerun: docling pipeline options 变化，丢弃 converter cache
                self.invalidate_converter_cache()

            self._apply_docling_global_memory_settings()

            conv = self._get_converter()
            kwargs = {}
            if self.config.max_file_size is not None:
                kwargs["max_file_size"] = self.config.max_file_size
            if self.config.max_num_pages is not None:
                kwargs["max_num_pages"] = self.config.max_num_pages

            result = conv.convert(str(source), **kwargs)
            md_kw: dict = {"image_mode": ImageRefMode.REFERENCED}
            if self.config.markdown_exclude_page_header_footer:
                md_kw["labels"] = _MARKDOWN_LABELS_WITHOUT_PAGE_HEADER_FOOTER
            result.document.save_as_markdown(markdown_out, **md_kw)

            if self.config.markdown_escape_dimension_asterisks:
                md = markdown_out.read_text(encoding="utf-8")
                markdown_out.write_text(
                    _escape_dimension_like_asterisks(md),
                    encoding="utf-8",
                )

            if refiner is None:
                break

            docling_md_text = markdown_out.read_text(encoding="utf-8")
            try:
                from src.vl_markdown_utils import (
                    summarize_markdown_quality,
                    validate_image_refs_invariants,
                )
                if self.config.llm_table_refine:
                    refined_md = refiner.cleanup_tables_per_block(
                        original_markdown=docling_md_text,
                        markdown_out_path=markdown_out,
                        vl_image_mode=self.config.llm_vl_image_mode,
                        cleanup_max_images_per_table=self.config.llm_table_cleanup_max_images_per_table,
                        cleanup_max_tables=self.config.llm_table_cleanup_max_tables,
                        context_lines=self.config.llm_table_context_lines,
                    )
                else:
                    image_inputs = refiner.prepare_image_inputs(
                        original_md=docling_md_text,
                        markdown_out_path=markdown_out,
                        vl_image_mode=self.config.llm_vl_image_mode,
                        cleanup_max_images=self.config.llm_cleanup_max_images,
                    )
                    refined_md = refiner.cleanup_markdown(
                        original_markdown=docling_md_text,
                        markdown_out_path=markdown_out,
                        image_inputs=image_inputs,
                    )

                refined_md = (refined_md or "").strip()
                if not refined_md:
                    _log.warning("LLM cleanup returned empty output; fallback to original md")
                    refined_md = docling_md_text
                elif len(refined_md) < 50:
                    _log.warning(
                        "LLM cleanup output too short (len=%s); fallback to original md",
                        len(refined_md),
                    )
                    refined_md = docling_md_text

                orig_stats = summarize_markdown_quality(docling_md_text)
                refined_stats = summarize_markdown_quality(refined_md)
                if orig_stats.get("table_rows", 0) > 0 and refined_stats.get("table_rows", 0) == 0:
                    _log.warning(
                        "LLM cleanup removed table rows; fallback to original md (orig_table_rows=%s)",
                        orig_stats.get("table_rows", 0),
                    )
                    refined_md = docling_md_text

                ok, details = validate_image_refs_invariants(
                    original_md=docling_md_text,
                    refined_md=refined_md,
                )
                if not ok:
                    _log.warning("LLM refined md failed invariants; fallback to original: %s", details)
                    refined_md = docling_md_text

                markdown_out.write_text(refined_md, encoding="utf-8")

                qc = refiner.quality_check(
                    original_markdown=docling_md_text,
                    refined_markdown=refined_md,
                )
                last_qc_score = qc.score
                if (
                    qc.need_rerun
                    and qc.suggest is not None
                    and attempt_i + 1 < max_attempts
                ):
                    new_cfg = self._apply_llm_rerun_suggestion(
                        cfg=self.config, suggest=qc.suggest
                    )
                    if new_cfg == self.config:
                        _log.info(
                            "LLM suggested rerun but config unchanged; stop. score=%s",
                            last_qc_score,
                        )
                        break
                    _log.info(
                        "LLM suggested rerun. attempt=%s->%s score=%s",
                        attempt_i,
                        attempt_i + 1,
                        last_qc_score,
                    )
                    attempt_cfg = new_cfg
                    continue
            except Exception as e:  # noqa: BLE001
                _log.exception("LLM refine failed; fallback to docling md: %s", repr(e))
            break

        # 所有 LLM refine / rerun 完成后，统一补表格语义说明
        if markdown_out.exists():
            try:
                final_md = markdown_out.read_text(encoding="utf-8")

                if self.config.llm_table_caption:
                    final_md = self._append_table_captions_to_markdown(final_md)

                if self.config.llm_image_caption:
                    final_md = self._append_image_captions_to_markdown(final_md, markdown_out)

                if self.config.markdown_escape_dimension_asterisks:
                    final_md = _escape_dimension_like_asterisks(final_md)

                markdown_out.write_text(final_md, encoding="utf-8")
            except Exception as e:  # noqa: BLE001
                _log.exception("Append markdown enrichments failed: %s", repr(e))
        self.config = original_cfg
        if last_qc_score is not None:
            _log.info("LLM quality_check last score=%s", last_qc_score)

    @staticmethod
    def iter_supported_files(input_root: Path) -> Iterable[Path]:
        """Depth-first walk under input_root, yielding convertible files."""
        for path in sorted(input_root.rglob("*")):
            if not path.is_file():
                continue
            if path.name.startswith("."):
                continue
            suf = path.suffix.lower()
            if suf in _DOCLING_SUFFIXES | _XLSX_SUFFIXES:
                yield path
    
    def _should_caption_table(self, table_markdown: str) -> bool:
        """
        过滤掉太短、太小、信息量太低的表格，避免无意义补偿。
        """
        if not table_markdown or not table_markdown.strip():
            return False

        lines = [x for x in table_markdown.splitlines() if x.strip()]
        if len(lines) < 3:  # 至少 header + 分隔行 + 1 行数据
            return False

        first = lines[0].strip()
        if not (first.startswith("|") and first.count("|") >= 2):
            return False

        cols = max(0, first.count("|") - 1)
        if cols < 2:
            return False

        # 太短通常没有补充价值
        if len(table_markdown.strip()) < 40:
            return False

        return True
    

    def _build_table_context_text(
        self,
        markdown_text: str,
        start_line: int,
        end_line: int,
        context_lines: int = 2,
    ) -> str:
        lines = markdown_text.splitlines()
        s = max(0, start_line - context_lines)
        e = min(len(lines), end_line + 1 + context_lines)

        before = "\n".join(lines[s:start_line]).strip()
        after = "\n".join(lines[end_line + 1:e]).strip()

        parts: list[str] = []
        if before:
            parts.append("【前文】\n" + before)
        if after:
            parts.append("【后文】\n" + after)
        return "\n\n".join(parts).strip()
    
    def _append_table_captions_to_markdown(self, markdown_text: str) -> str:
        if not self.config.llm_table_caption:
            return markdown_text

        from src.vl_markdown_utils import extract_markdown_table_blocks

        client = self._create_dashscope_client()
        if client is None:
            _log.warning("llm_table_caption enabled but DashScope client unavailable; skip.")
            return markdown_text

        blocks = extract_markdown_table_blocks(markdown_text)
        if not blocks:
            return markdown_text

        max_tables = max(0, int(self.config.llm_table_caption_max_tables))
        if max_tables <= 0:
            return markdown_text

        lines = markdown_text.splitlines()
        inserts: list[tuple[int, list[str]]] = []

        handled = 0
        for block in blocks:
            if handled >= max_tables:
                break
            if not self._should_caption_table(block.markdown):
                continue

            context_text = self._build_table_context_text(
                markdown_text,
                start_line=block.start_line,
                end_line=block.end_line,
                context_lines=max(0, int(self.config.llm_table_context_lines)),
            )

            caption = self._generate_table_caption_text(
                client=client,
                table_markdown=block.markdown,
                context_text=context_text,
            )
            if not caption:
                continue

            insert_after = block.end_line + 1
            insert_block = [
                "",
                f"> **表格信息：** {caption}",
                "",
            ]
            inserts.append((insert_after, insert_block))
            handled += 1

        if not inserts:
            return markdown_text

        # 倒序插入，避免行号漂移
        for insert_after, insert_block in reversed(inserts):
            lines[insert_after:insert_after] = insert_block

        return "\n".join(lines)
    
    def _extract_markdown_image_blocks(self, markdown_text: str) -> list[dict]:
        items: list[dict] = []
        lines = markdown_text.splitlines()

        for i, line in enumerate(lines):
            for m in _MARKDOWN_IMAGE_RE.finditer(line):
                alt = (m.group("alt") or "").strip()
                src = (m.group("src") or "").strip()
                if not src:
                    continue
                items.append(
                    {
                        "alt": alt,
                        "src": src,
                        "line_no": i,
                        "raw": m.group(0),
                    }
                )
        return items
    
    def _build_image_context_text(
        self,
        markdown_text: str,
        line_no: int,
        context_lines: int = 3,
    ) -> str:
        lines = markdown_text.splitlines()
        s = max(0, line_no - context_lines)
        e = min(len(lines), line_no + 1 + context_lines)

        before = "\n".join(x for x in lines[s:line_no] if x.strip()).strip()
        after = "\n".join(x for x in lines[line_no + 1:e] if x.strip()).strip()

        parts: list[str] = []
        if before:
            parts.append("【前文】\n" + before)
        if after:
            parts.append("【后文】\n" + after)
        return "\n\n".join(parts).strip()

    def _should_caption_image(self, image_path: Path) -> bool:
        if not image_path.is_file():
            return False

        try:
            with Image.open(image_path) as im:
                w, h = im.size
        except Exception:
            return False

        min_w = max(0, int(self.config.llm_image_caption_min_width))
        min_h = max(0, int(self.config.llm_image_caption_min_height))
        min_area = max(0, int(self.config.llm_image_caption_min_area))

        if min_w > 0 and w < min_w:
            return False
        if min_h > 0 and h < min_h:
            return False
        if min_area > 0 and (w * h) < min_area:
            return False

        return True

    
    def _generate_image_caption_text(
        self,
        *,
        client: "DashScopeClient",
        image_path: Path,
        context_text: str,
    ) -> Optional[str]:
        try:
            from src.llm_prompts import build_image_caption_messages

            messages = build_image_caption_messages(
                image_path=image_path,
                context_text=context_text,
                max_chars=self.config.llm_image_caption_max_chars,
                image_mode=self.config.llm_vl_image_mode,
            )

            text = client.generate_multimodal(
                model=self.config.llm_model,
                messages=messages,
                temperature=0.0,
                max_tokens=min(256, int(self.config.llm_max_tokens or 256)),
                biz_stage="图片语义增强",
            )
            text = (text or "").strip()
            if not text:
                return None

            text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            text = re.sub(r"\n+", " ", text).strip()
            text = re.sub(r"^(图像补充[:：]\s*)", "", text).strip()
            text = re.sub(r"^(该图展示了|该图说明了|该图反映了|这张图展示了)", "", text).strip()

            if not text:
                return None

            max_chars = int(self.config.llm_image_caption_max_chars or 0)
            if max_chars > 0 and len(text) > max_chars:
                text = text[:max_chars].rstrip("，,;；。 ") + "。"


            return text
        except Exception as e:  # noqa: BLE001
            _log.exception("Generate image caption failed: %s", repr(e))
            return None
    
    def _append_image_captions_to_markdown(
        self,
        markdown_text: str,
        markdown_out: Path,
    ) -> str:
        if not self.config.llm_image_caption:
            return markdown_text

        client = self._create_dashscope_client()
        if client is None:
            _log.warning("llm_image_caption enabled but DashScope client unavailable; skip.")
            return markdown_text

        images = self._extract_markdown_image_blocks(markdown_text)
        if not images:
            return markdown_text

        max_images = int(self.config.llm_image_caption_max_images or 0)

        lines = markdown_text.splitlines()
        inserts: list[tuple[int, list[str]]] = []

        handled = 0
        for item in images:
            if max_images > 0 and handled >= max_images:
                break

            src = item["src"]
            line_no = int(item["line_no"])

            image_path = (markdown_out.parent / src).resolve()
            if not self._should_caption_image(image_path):
                continue

            context_text = self._build_image_context_text(
                markdown_text,
                line_no=line_no,
                context_lines=max(0, int(self.config.llm_image_caption_context_lines)),
            )

            caption = self._generate_image_caption_text(
                client=client,
                image_path=image_path,
                context_text=context_text,
            )
            if not caption:
                continue

            label = self._resolve_image_label(
                markdown_text,
                line_no=line_no,
                alt_text=str(item.get("alt") or ""),
            )

            insert_after = line_no + 1
            insert_block = [
                "",
                f"> **{label}：** {caption}",
                "",
            ]
            inserts.append((insert_after, insert_block))
            handled += 1


        if not inserts:
            return markdown_text

        for insert_after, insert_block in reversed(inserts):
            lines[insert_after:insert_after] = insert_block

        return "\n".join(lines)





    def _generate_table_caption_text(
        self,
        *,
        client: "DashScopeClient",
        table_markdown: str,
        context_text: str,
    ) -> Optional[str]:
        try:
            from src.llm_prompts import build_table_caption_messages

            max_chars_cfg = max(20, int(self.config.llm_table_caption_max_chars))
            messages = build_table_caption_messages(
                table_markdown=table_markdown,
                context_text=context_text,
                max_chars=max_chars_cfg,
            )
            # 补全长度与说明字符上限对齐，并受全局 llm_max_tokens 约束（不再单独维护 TABLE_CAPTION_MAX_TOKENS）
            cap_tok = min(
                int(self.config.llm_max_tokens or 8192),
                max(32, max_chars_cfg),
            )
            text = client.generate_multimodal(
                model=self.config.llm_model,
                messages=messages,
                temperature=0.0,
                max_tokens=cap_tok,
                biz_stage="表格语义增强",
            )

            text = (text or "").strip()

            if not text:
                return None

            # 清掉模型可能带出来的代码块/多余前缀
            text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            text = re.sub(r"^(该表列出了|该表说明了|该表展示了|该表反映了)", "", text).strip()

            # 单段化，避免太长
            text = re.sub(r"\n+", " ", text).strip()
            if not text:
                return None

            if len(text) > max_chars_cfg:
                text = text[:max_chars_cfg].rstrip("，,;；。 ") + "。"

            return text
        except Exception as e:  # noqa: BLE001
            _log.exception("Generate table caption failed: %s", repr(e))
            return None
    
    



