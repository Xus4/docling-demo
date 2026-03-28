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

参数写法约定：命令行示例统一使用 **`--opt value`** 形式；仓库内“是否显式传参”的判断也按该形式设计。

## Web 服务（MVP）

本项目支持以轻量 Web 服务方式运行（FastAPI + Uvicorn），用于内网多人上传文件并下载 Markdown。

### 1) 安装依赖

```powershell
pip install -r requirements.txt
```

### 2) 配置管理员参数（环境变量）

推荐使用 `.env` 文件（更方便保存配置）：

```powershell
Copy-Item .env.example .env
```

然后编辑 `.env` 中的参数；程序启动时会自动读取该文件。  
优先级为：**系统环境变量 > `.env` 文件 > 代码默认值**。

```powershell
$env:MAX_FILE_SIZE="20MB"
$env:ALLOWED_TYPES="pdf,docx,pptx,html,png,jpg,jpeg"
$env:DATA_DIR="./data"
$env:DEBUG="false"
$env:AUTO_CLEANUP="true"
$env:CLEANUP_MAX_AGE_HOURS="24"
```

说明：

- `MAX_FILE_SIZE`：最大上传文件大小（支持 `B/KB/MB/GB`，例如 `20MB`）
- `ALLOWED_TYPES`：允许上传的扩展名白名单（逗号分隔）
- `DATA_DIR`：数据根目录（默认 `./data`）
- `AUTO_CLEANUP`：是否自动清理历史任务目录
- `CLEANUP_MAX_AGE_HOURS`：自动清理阈值（小时）

可选转换参数（将你原 CLI 常用参数映射为环境变量）：

- `PDF_VL_PRIMARY`（默认 `true`）
- `PDF_VL_DPI`（默认 `180`）
- `PDF_VL_WORKERS`（默认 `10`）
- `LLM_MODEL`（默认 `qwen3.5-35b-a3b`）
- `PDF_VL_TABLE_SECOND_PASS_MAX_TABLES`（默认 `5`）
- `MAX_NUM_PAGES`（默认不限制）
- `LLM_MAX_TOKENS`（默认 `16384`）
- `LLM_TEMPERATURE`（默认 `0`）
- `LLM_ENABLE_THINKING`（默认 `true`）
- `LLM_TABLE_CAPTION`（默认 `true`）
- `LLM_TABLE_CAPTION_MAX_CHARS`（默认 `500`）
- `PDF_CAPTION_CROP_FIGURES`（默认 `true`）
- `DASHSCOPE_API_KEY`（启用 Qwen/DashScope 时必填）
- `AUTH_DB_PATH`（默认 `./data/auth.db`）
- `SESSION_SECRET`（会话签名密钥）
- `INITIAL_PASSWORD`（初始化新用户统一密码，入库为哈希）
- `AUTH_ADMIN_USERNAME`（默认 `admin`）
- `AUTH_USERS`（普通用户列表，逗号分隔）

示例（接近你原来的 CLI 设定）：

```powershell
$env:PDF_VL_PRIMARY="true"
$env:PDF_VL_DPI="180"
$env:PDF_VL_WORKERS="10"
$env:LLM_MODEL="qwen3.5-35b-a3b"
$env:PDF_VL_TABLE_SECOND_PASS_MAX_TABLES="5"
$env:MAX_NUM_PAGES="5"
$env:LLM_MAX_TOKENS="16384"
$env:LLM_TEMPERATURE="0"
$env:LLM_ENABLE_THINKING="true"
$env:LLM_TABLE_CAPTION="true"
$env:LLM_TABLE_CAPTION_MAX_CHARS="500"
$env:PDF_CAPTION_CROP_FIGURES="true"
```

目录结构（运行后自动创建）：

```text
data/
  input/
  output/
```

每次请求都会生成唯一 `job_id`，并在 `input/<job_id>/` 与 `output/<job_id>/` 下隔离输入输出，避免文件冲突。

### 3) 启动服务

```powershell
python -m uvicorn webapp:app --host 0.0.0.0 --port 8000
```

Linux 上同样使用：

```bash
export MAX_FILE_SIZE=20MB
export ALLOWED_TYPES=pdf,docx,pptx
export DATA_DIR=./data
uvicorn webapp:app --host 0.0.0.0 --port 8000
```

### 4) 使用方式

- 打开浏览器访问 `http://<服务器IP>:8000/`
- 登录后上传文件：任务进入**异步队列**，页面可查看**任务列表**与状态；完成后可下载 Markdown
- 任务记录在 SQLite 中**持久化**：**退出登录或关闭浏览器后再次登录**，仍可看到历史任务；可对排队/已结束任务**删除记录**（并清理对应输入输出目录）；**执行中**的任务需先**取消**再删除
- **管理员**可在任务列表中按**用户**、**状态**筛选；**普通用户**仅能看到自己的任务
- 服务**重启**后，库中仍为 `queued` 的任务会自动重新进入内存队列继续排队执行

接口说明：

- `POST /auth/login`：登录
- `POST /auth/logout`：退出登录
- `GET /auth/me`：查看当前登录态（`username`、`role`）
- `GET /auth/users`：**仅管理员**，返回系统用户列表（用于筛选等）
- `POST /jobs`：上传文件并**创建异步任务**（立即返回 `job_id`、`status`、`created_at`）
- `GET /jobs`：分页任务列表（`page`、`page_size`）；管理员可选 `owner`、`status` 筛选
- `GET /jobs/{job_id}`：任务详情（含 `download_url`，仅成功时有值）
- `POST /jobs/{job_id}/cancel`：取消任务（排队中立即取消；运行中为**软取消**，当前转换步骤结束后不再提供下载）
- `DELETE /jobs/{job_id}`：**删除任务记录**（所有者或管理员；**进行中**的任务返回 409，需先取消）；并尝试删除 `data/input/<job_id>/` 与 `data/output/<job_id>/`
- `GET /jobs/{job_id}/download`：下载该任务的 Markdown（**仅任务所有者或管理员**，且状态为 `succeeded`）
- `POST /convert`：**兼容旧客户端**——内部创建异步任务并**阻塞直到结束**，响应仍为 `job_id`、`filename`、`download_url`（`download_url` 形态仍为 `/download/{job_id}`，与新版校验一致）
- `GET /download/{job_id}`：兼容旧下载链接，权限与 `GET /jobs/{job_id}/download` 相同
- `GET /health`：健康检查

任务状态：`queued`（排队）→ `running`（转换中）→ `succeeded` | `failed` | `cancelled`。任务元数据持久化在 SQLite（默认与认证库同库：`AUTH_DB_PATH`），输入输出仍在 `data/input/<job_id>/`、`data/output/<job_id>/`。会话（登录 Cookie）与任务数据无关，**登出不会删除任务**。

**管线并发（大页/扫描 PDF 防 OOM）**：`StandardPdfPipeline` 在 OCR / 版面 / 表格阶段之间用 **队列** 流水线处理多页；并发过高会让大页同时在内存里排队。可用 **`--pipeline-concurrency`**：`low`（batch≤2、队列≤16）或 **`minimal`**（batch=1、队列≤2，最省内存）。使用 **`--scan`** 或 **`--low-memory`** 且**未传** `--pipeline-concurrency` 时，**自动为 `minimal`**。需要略高吞吐可传 **`--pipeline-concurrency low`**；强行使用 Docling 默认并发可传 **`--pipeline-concurrency default`**（大扫描件易 OOM）。

**扫描件（Docling 管线）**：若希望「更稳 + 中文表格 OCR」，请**自行组合**开关，例如 **`--scan`**、**`--pipeline-concurrency minimal`**、**`--ocr-quality high`**、**`--ocr-bitmap-threshold 0.03`**，或按下文「识别精度」一节微调。仍 OOM 或内存不足时加 **`--low-memory`**（会强制 OCR 为 `fast`，表格精度可能下降）或 **`--no-tables`**。
若需对 Docling 输出的表格块做 **Qwen-VL 二次纠错**，可加 **`--enable-llm`** 与 **`--llm-table-refine`**（需 **`DASHSCOPE_API_KEY`**）。

**PDF 方案 A（按页 Qwen-VL，不经过 Docling）**：仅对 **`.pdf`** 生效。用 **PyMuPDF** 逐页渲染为 PNG，**每页一次**多模态 API 转写为 Markdown；需 **`DASHSCOPE_API_KEY`**（与 `--llm-base-url` 地域一致）。成本与 **页数** 成正比，试跑请配合 **`--max-num-pages`** / **`--max-files`**。

**最简常用命令**（你当前工作流）：

```powershell
python main.py --pdf-vl-primary --output-by-model --pdf-caption-crop-figures --max-files 1 --max-num-pages 5
```

在 **`--pdf-vl-primary`** 下，若命令行**未显式传入**同名参数，程序会采用与仓库常用配置一致的默认值（**显式传参始终优先**）：

| 参数 | 默认 |
|------|------|
| `--llm-model` | `qwen3.5-35b-a3b` |
| `--pdf-vl-dpi` | `180` |
| `--pdf-vl-workers` | `10` |
| `--pdf-vl-table-second-pass-max-tables` | `5` |
| `--llm-temperature` | `0` |
| `--llm-max-tokens` | `16384` |

注：上表中 **`--llm-temperature` / `--llm-max-tokens`** 的自动默认仅在 **`--pdf-vl-primary`** 时生效；不走 pdf-vl 时 CLI 解析默认分别为 **`0.2`** 与 **`8192`**（可自行覆盖）。**`--llm-model`** 的默认值对整条 `main.py` 生效（当前为 `qwen3.5-35b-a3b`）。

你也可以随时显式指定这些参数覆盖默认（例如 `--pdf-vl-dpi 220`、`--llm-max-tokens 24576`、`--llm-temperature 0.1`）。

默认**不嵌入整页截图**到 Markdown（避免把“整页图”误当文档插图）；如需调试可加 `--pdf-vl-embed-page-images`。
默认启用“可疑表格二次 LLM 校对（带验收回退）”，可用 `--no-pdf-vl-table-second-pass` 关闭。

**常用示例**（按模型名分子目录输出、按图题裁局部插图；其余见上表默认值）：

```powershell
$env:DASHSCOPE_API_KEY="..."
python main.py --pdf-vl-primary --output-by-model --pdf-caption-crop-figures `
  --input-dir ./data/input --output-dir ./data/output --max-files 1 --max-num-pages 5
```

若还需对整篇 VL 结果做二次清洗，可加 **`--enable-llm`**（会额外调用 API）。

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

启用 LLM 清洗（默认会尽量保留 Docling 的图片引用路径；若图片引用缺失会自动回退到原始 Docling md）。`main.py` 默认 **`--llm-model`** 为 **`qwen3.5-35b-a3b`**，可按控制台可用模型改写。

```powershell
python main.py --input-dir ./data/input --output-dir ./data/output `
  --enable-llm `
  --llm-vl-image-mode local_abs `
  --llm-allow-rerun
```

说明：

- LLM 只处理 Docling 产物的 `.md`（不处理 `XLSX`，XLSX 仍走 pandas 导出）。
- 由于 Qwen-VL 会读取图片上下文，Docling 输出中需要有 `![](<path>)` 图片引用；本项目默认最多把前 `6` 张图片喂给模型。
- **Qwen3.5 思考模式**：HTTP 请求体顶层发送 **`enable_thinking: false`**（与 OpenAI Python SDK 的 `extra_body={"enable_thinking": false}` 等价展开），避免只出 `reasoning_content` 而 `content` 为空。若需深度思考，可加 **`--llm-enable-thinking`**（会额外消耗 completion token）。
- **输出长度**：`--llm-max-tokens` 控制单次 **completion** 上限（与「最大输入/上下文」不同）；多数 Qwen3.5 模型单路约可到 **64K**，页级转写若仍截断可再调高。

### 成本与安全

- 成本：每个文件至少 1 次 LLM 调用（cleanup），若触发 rerun 则会多一次 Docling + LLM 流程。
- 安全：程序侧会校验 LLM 输出中的图片引用是否与原始 Markdown 保持一致；不通过会回退到原始 Docling Markdown，避免“路径被改写导致图片丢失”的问题。

## 旧脚本

原根目录下的 `convert.py` / `batch_convert.py` 已移除，请使用 `main.py`。
