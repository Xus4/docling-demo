# docling-demo（工业级多格式 → Markdown）

使用 [Docling](https://github.com/docling-project/docling) 将 **PDF / DOCX / PPTX / HTML / 常见图片** 转为 **Markdown**（表格结构识别、可选公式/LaTeX 富集、OCR、图片外链）；**XLSX** 使用 **pandas** 按 **Sheet** 导出为 Markdown 表格。

**默认配置偏内存安全**：`images_scale=1.0`、不生成整页高清 PNG、**不加载公式 VLM**；需要公式或高分辨率配图时请显式传入 `--formula` / `--rich-images`。

## 目录约定

| 路径 | 用途 |
|------|------|
| `data/input/pdf/` | PDF |
| `data/input/word/` | DOCX |
| `data/input/excel/` | XLSX（pandas 按 Sheet 转表） |
| `data/input/ppt/` | PPTX |
| `data/input/html/` | HTML / HTM |
| `data/input/images/` | PNG / JPEG / TIFF / BMP / WEBP |
| `data/output/` | 输出的 `.md` 及 Docling 抽取的图片（与 `save_as_markdown` 行为一致） |

## 环境（Windows / PowerShell）

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

说明：

- **GPU（NVIDIA）**：默认 **`python main.py` 使用 `--device cuda`**。若日志出现 **`torch.cuda.is_available()=False`** 或 **`torch x.x.x+cpu`**，说明当前是 **CPU 版 PyTorch**（从 PyPI 默认装 `docling` 时很容易被 `torch` 依赖拉成 `+cpu`），**不会**用显卡。

先自检：

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

应看到 **`+cu124` / `+cu128` 等**且 **`True`**。若是 **`+cpu`**，请**先卸载再装**（二选一，**RTX 50 系 / Blackwell 优先 cu128**）：

```powershell
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

若仍提示 **sm_120 / Blackwell 与当前 PyTorch 不兼容**，请升级到 **PyTorch 2.7+** 且 **CUDA 12.8（cu128）** 的官方轮子，版本以 [PyTorch 安装页](https://pytorch.org/get-started/locally/) 为准。

无独显或只想用 CPU 时：`python main.py --device cpu` 或 `--cpu`。

- **EasyOCR** 会拉取 **torch** 等依赖，首次运行会下载模型。
- 若使用 **`--ocr-engine tesseract`**，需本机安装 [Tesseract](https://github.com/tesseract-ocr/tesseract) 及语言数据，并执行 `pip install -r requirements-ocr-tesseract.txt`（Windows 上 `tesserocr` 轮子/编译环境需自行准备）。
- `requirements.txt` 请保持 **纯 ASCII**（或 UTF-8 带 BOM），否则在部分 Windows 区域设置下 `pip install -r` 可能因编码报错。

## 一键批量转换（验证）

将待转文件放入 `data/input/` 下对应子目录后：

```powershell
python main.py --input-dir ./data/input --output-dir ./data/output
```

**管线并发（大页/扫描 PDF 防 OOM）**：`StandardPdfPipeline` 在 OCR / 版面 / 表格阶段之间用 **队列** 流水线处理多页；并发过高会让大页同时在内存里排队。可用 **`--pipeline-concurrency`**：`low`（batch≤2、队列≤16）或 **`minimal`**（batch=1、队列≤2，最省内存）。使用 **`--scan`** 或 **`--low-memory`** 且**未传** `--pipeline-concurrency` 时，**自动为 `minimal`**。需要略高吞吐可传 **`--pipeline-concurrency low`**；强行使用 Docling 默认并发可传 **`--pipeline-concurrency default`**（大扫描件易 OOM）。

**推荐预设 `--profile`**（减少 preprocess 报错、提升中文表格 OCR，**显式传入的同名参数优先**）：

- **`stable-chinese`**：在未手写同名开关时启用 **`--scan`**、**`--pipeline-concurrency minimal`**、**`--ocr-quality high`**、**`--ocr-bitmap-threshold 0.03`**，适合扫描件/表内小字截图。
- **`stable-chinese-llm`**：同上，并开启 **`--enable-llm`** 与 **`--llm-table-refine`**（需 **`DASHSCOPE_API_KEY`**），用 Qwen-VL 对表格块做二次纠错。

```powershell
python main.py --profile stable-chinese --input-dir ./data/input --output-dir ./data/output
```

仍 OOM 或内存不足时，在 profile 基础上再加 **`--low-memory`**（会强制 OCR 为 `fast`，表格精度可能下降）或 **`--no-tables`**。

**识别精度（`--ocr-quality`）**：此前为防 OOM 把扫描件 **`images_scale` 压得很低**（`fast` 档），会明显拖垮 OCR/表格。若已能跑通，建议对扫描规范类 PDF 使用 **`--scan --ocr-quality high`**（更高渲染分辨率、TableFormer **ACCURATE**、EasyOCR 阈值与中文优先语言顺序、扫描页整页 OCR）。**不启用** EasyOCR 的 `craft` 识别网络（默认安装常缺 `craft.yaml` 会报错）。更省内存用 **`fast`**。**`--low-memory`** 会强制等价于 **`fast`**。

扫描件优先尝试（精度优先，若 OOM 再改回 `balanced` 或加 `--pipeline-concurrency low`）：

```powershell
python main.py --input-dir ./data/input --output-dir ./data/output --scan --ocr-quality high
```

在 **`high`** 仍不够时，可微调（显存/内存充足时）：**`--scan-max-scale 1.25`** 提高渲染上限；**`--ocr-confidence 0.28`** 略降阈值减少漏字（可能多噪）；**`--ocr-bitmap-threshold 0.025`** 让更小的位图区域参与 OCR。源 PDF 若模糊、倾斜，可先用外部工具做去噪、纠偏、提高 DPI 再转换，效果往往优于单纯调参。

大 PDF / 内存紧张（推荐）：

```powershell
python main.py --low-memory --max-files 1 --no-log-file
```

需要公式识别时再打开（会下载 CodeFormula 模型，CPU 上很慢且占内存）：

```powershell
python main.py --formula
```

需要高分辨率整页图时再打开（**很吃内存**）：

```powershell
python main.py --rich-images
```

仅处理前 1 个文件试跑：

```powershell
python main.py --max-files 1 --no-log-file
```

仅控制台日志、不写 `convert.log`：

```powershell
python main.py --no-log-file
```

## 故障排查

| 现象 | 可能原因与处理 |
|------|----------------|
| `PdfiumError: Data format error` / `Input document ... is not valid` | PDF 损坏、加密、或非标准封装；换阅读器另存为 PDF、或尝试解除密码后再转。 |
| `std::bad_alloc`（预处理/某页失败） | **内存不足**：Docling 预处理会先 **`get_image(scale=1.0)`** 再按 `images_scale` 缩放，超大扫描页在 **管线队列** 里堆积时尤其容易 OOM。请 **`--scan`**（已收紧 `images_scale≤0.25`、各阶段 `batch_size=1`、`queue_max_size≤4`、全局 `page_batch_size=1`），仍失败则 **`--scan --low-memory --no-tables`**，或把 **`--images-scale`** 再降到 **`0.15`**；勿用 **`--rich-images`**，勿开 **`--formula`**。 |
| 扫描件「几乎没字」 | 扫描件无文字层，必须 **开启 OCR**（不要 `--no-ocr`）。 |
| HuggingFace 下载超时 | 网络问题；可重试，或预先 `docling-tools models download`，或设置代理/`HF_ENDPOINT`。 |
| `Cannot close object; pdfium library is destroyed` | 多见于 **上一文件转换失败** 后 pypdfium2 与 Docling 清理顺序；本仓库在失败后会 **丢弃转换器缓存** 以减轻。若仅出现在失败批次末尾，一般可忽略；持续出现可升级 `docling` / `pypdfium2` 或单文件重试。 |

## 核心模块

- `src/converter.py`：`IndustrialDocConverter`（Docling 管道配置 + XLSX/pandas）
- `main.py`：批量扫描、日志、单文件失败不影响整体

## 百炼千问接入（可选）

本项目在 Docling 输出 Markdown 之后，可通过阿里百炼（DashScope）上的千问（Qwen-VL）做一次“清洗/纠错 + 质量检查”，必要时触发最多 1 次 Docling rerun（提升 OCR/表格精度）。

### 前置条件

- 设置环境变量：`DASHSCOPE_API_KEY`（与控制台地域一致：国内控制台密钥走国内端点）
- 机器可访问 DashScope 接口（用于 HTTP 调用）
- 默认使用 **OpenAI 兼容** 地址：`--llm-base-url https://dashscope.aliyuncs.com/compatible-mode/v1`（请求走 `/chat/completions`）。若需原生 DashScope HTTP，可改为 `https://dashscope.aliyuncs.com/api/v1`；**国际版**兼容模式为 `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`。

### 使用示例

启用 LLM 清洗（默认会尽量保留 Docling 的图片引用路径；若图片引用缺失会自动回退到原始 Docling md）：

```powershell
python main.py --input-dir ./data/input --output-dir ./data/output `
  --enable-llm `
  --llm-model qwen3-vl-plus `
  --llm-vl-image-mode local_abs `
  --llm-allow-rerun
```

说明：

- LLM 只处理 Docling 产物的 `.md`（不处理 `XLSX`，XLSX 仍走 pandas 导出）。
- 由于 Qwen-VL 会读取图片上下文，Docling 输出中需要有 `![](<path>)` 图片引用；本项目默认最多把前 `6` 张图片喂给模型。

### 成本与安全

- 成本：每个文件至少 1 次 LLM 调用（cleanup），若触发 rerun 则会多一次 Docling + LLM 流程。
- 安全：程序侧会校验 LLM 输出中的图片引用是否与原始 Markdown 保持一致；不通过会回退到原始 Docling Markdown，避免“路径被改写导致图片丢失”的问题。

## 旧脚本

原根目录下的 `convert.py` / `batch_convert.py` 已移除，请使用 `main.py`。
