# 智枢文档（docling-demo）

基于 [Docling](https://github.com/docling-project/docling) 的工业文档转换：**批量 CLI** 与 **FastAPI Web** 共用同一套转换配置（`.env` / `AppConfig`）。支持 PDF、Word、PPT、HTML、常见图片、Excel 等格式输出 **Markdown**，并可选 **PDF 按页多模态（VL）转写**、表格/图片语义说明、表格块纠错等（通过 OpenAI 兼容接口，默认对接阿里云百炼，也可指向本地 Ollama 等）。

---

## 目录

- [能力与技术栈](#能力与技术栈)
- [架构与数据流](#架构与数据流)
- [环境要求](#环境要求)
- [安装](#安装)
- [配置说明（环境变量）](#配置说明环境变量)
- [认证：本地用户与 OA](#认证本地用户与-oa)
- [运行 Web 服务](#运行-web-服务)
- [HTTP API 参考](#http-api-参考)
- [任务状态与权限](#任务状态与权限)
- [批量转换 CLI（main.py）](#批量转换-climainpy)
- [GPU / PyTorch 提示](#gpu--pytorch-提示)
- [日志与排错](#日志与排错)
- [测试](#测试)
- [目录结构](#目录结构)
- [许可证与第三方](#许可证与第三方)

---

## 能力与技术栈

| 模块 | 说明 |
|------|------|
| **Docling** | 版式分析、PDF/Office/HTML/图片等管线；可配 OCR（EasyOCR / Tesseract）、表格结构、可选公式模型等。 |
| **Excel** | `.xlsx` 由 **pandas** 按工作表导出为 Markdown 表格（见 `src/converter.py`）。 |
| **MinerU（magic-pdf）** | 可选；PDF 配图裁剪时优先用其布局检测 bbox，未安装则回退 PyMuPDF（见 `requirements.txt` 注释）。 |
| **VL / LLM** | `src/pdf_vl_transcribe.py`、`src/dashscope_client.py` 等；兼容 OpenAI 风格的 `/chat/completions`。 |
| **Web** | **FastAPI** + `SessionMiddleware` + **JWT**（`access_token.py`）；静态页 `static/index.html`。 |
| **任务与账号** | **SQLAlchemy**；**SQLite** 或 **MySQL**；进程内队列 `JobQueueWorker`（`job_worker.py`）。 |

**默认支持的扩展名**（可通过 `ALLOWED_TYPES` 收窄）：`pdf, docx, pptx, html, htm, png, jpg, jpeg, tif, tiff, bmp, webp, xlsx`。

---

## 架构与数据流

```mermaid
flowchart LR
  subgraph Web
    UI[浏览器 / 静态前端]
    API[FastAPI webapp.py]
    W[JobQueueWorker]
  end
  subgraph Data
    DB[(用户与任务表)]
    IN[input_dir / job_id]
    OUT[output_dir / job_id]
  end
  subgraph Convert
    SVC[ConversionService]
    CV[IndustrialDocConverter]
  end
  UI --> API
  API --> DB
  API --> IN
  W --> DB
  W --> SVC
  SVC --> CV
  CV --> OUT
  API --> OUT
```

- **上传**：鉴权通过后，文件写入 `INPUT_DIR/<job_id>/...`，产出写入 `OUTPUT_DIR/<job_id>/...`。
- **队列**：单进程 worker 从数据库取 `queued` 任务执行；进程重启时会在 lifespan 中把异常中断的 `running` 置回队列并重新调度（见 `webapp.py` 中 `_lifespan`）。
- **下载**：成功任务通过将**整个产出目录**打成 **ZIP** 返回（单任务与批量下载均如此）。

---

## 环境要求

- **Python 3.10+**（建议与当前 Docling 版本说明一致）。
- **内存与磁盘**：Docling + EasyOCR + 可选 VL 较重；大 PDF 建议 CLI 使用 `--low-memory`、`--scan` 或限制 `--max-num-pages`。
- **调用 VL/LLM 时**：在环境中配置 API Key。默认读取 **`DASHSCOPE_API_KEY`**，或通过 **`LLM_API_KEY_ENV`** 指定其他环境变量名。若使用 **Ollama** 等无密钥服务，仍需保证 `LLM_BASE_URL` 与模型名与对端一致。
- **可选 GPU**：Docling 侧可通过 CLI `--device` 使用 CUDA；与 PyTorch 轮子是否带 CUDA 有关（见下文 [GPU / PyTorch](#gpu--pytorch-提示)）。

---

## 安装

```bash
cd docling-demo
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

在项目根目录维护 **`.env`**（`config.py` 与 `main.py` 均会通过 `python-dotenv` 加载，**override=false**：已存在的环境变量不会被 `.env` 覆盖）。

**生产环境务必修改**：`SESSION_SECRET`、`ACCESS_TOKEN_SECRET`（可留空则回退为 `SESSION_SECRET`）、`INITIAL_PASSWORD`，并根据部署方式关闭 `DEBUG`。

---

## 配置说明（环境变量）

下列默认值以 `config.py` 中 `AppConfig.from_env()` 为准；单位大小写不敏感（如 `20MB`、`1GB`）。

### 数据目录与 Web 上传

| 变量 | 默认 | 说明 |
|------|------|------|
| `DATA_DIR` | `项目根/data` | 数据根目录。 |
| `INPUT_DIR` | `DATA_DIR/input` | Web 上传与任务输入根路径。 |
| `OUTPUT_DIR` | `DATA_DIR/output` | 转换产出根路径。 |
| `MAX_FILE_SIZE` | `20MB` | 单文件上传上限（支持 `B`/`KB`/`MB`/`GB`）。 |
| `ALLOWED_TYPES` | 见上文集合 | 逗号分隔扩展名，不带点。 |
| `DEBUG` | `false` | FastAPI `debug`。 |

### PDF 按页 VL 与 Docling 相关（Web 与 `build_converter_config` 一致）

| 变量 | 默认 | 说明 |
|------|------|------|
| `PDF_VL_PRIMARY` | `true` | 是否以「PDF 按页 VL 转写」为主流程（关闭则走 Docling 为主，取决于转换器内部逻辑）。 |
| `PDF_VL_DPI` | `180` | 按页渲染 DPI。 |
| `PDF_VL_WORKERS` | `10` | 页级并发；本地单卡可适当调低避免排队或显存抖动。 |
| `MAX_NUM_PAGES` | 空=不限制 | 每个文档最大处理页数。 |
| `PDF_VL_TABLE_SECOND_PASS` | `true` | 是否对可疑表格做二次 LLM 校对。 |
| `PDF_VL_TABLE_SECOND_PASS_MAX_TABLES` | `0` | 每页二次校对表格数上限；`0` 表示不限制。 |

### 大模型（OpenAI 兼容接口）

| 变量 | 默认 | 说明 |
|------|------|------|
| `LLM_MODEL` | `qwen3.5-35b-a3b` | 模型名（百炼 / Ollama 等各自命名）。 |
| `LLM_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | API 根地址；CLI 可用 `--llm-base-url` 覆盖说明见 `main.py -h`。 |
| `LLM_API_KEY_ENV` | `DASHSCOPE_API_KEY` | 从哪个**环境变量名**读取 API Key。 |
| `LLM_MAX_TOKENS` | `16384` | 单次 completion 上限；未单独配置时，表格/图片说明的字符上限会与此对齐。 |
| `LLM_TEMPERATURE` | `0.0` | 采样温度。 |
| `LLM_MAX_RETRIES` | `3` | 失败重试次数。 |
| `LLM_RETRY_BACKOFF_SEC` | `1.5` | 重试退避（秒）。 |
| `LLM_MAX_REASONING_TOKENS` | `256` | 思考模式相关 token 上限（依提供商语义）。 |
| `LLM_ENABLE_THINKING` | `true` | 是否启用思考链类能力；本地推理卡顿时可关。 |
| `LLM_TIMEOUT_SEC` | `300` | 单次请求超时（秒），实际会取与连接超时相关的下限。 |
| `LLM_EMPTY_CONTENT_MAX_ATTEMPTS` | `3` | `assistant.content` 为空时的重试次数。 |
| `LLM_LOG_STREAM_RESPONSE` | `false` | 是否流式请求并在日志打印片段。 |
| `LLM_VL_IMAGE_MODE` | `local_abs` | 多模态图片引用方式：`local_abs` 或 `url`。 |
| `LLM_CLEANUP_MAX_IMAGES` | `6` | 多模态消息里最多附带几张图（清洗/纠错）。 |

### 表格 / 图片增强

| 变量 | 默认 | 说明 |
|------|------|------|
| `LLM_TABLE_CAPTION` | `true` | 是否为表格生成说明文字。 |
| `LLM_TABLE_CAPTION_MAX_CHARS` | 默认同 `LLM_MAX_TOKENS` | 每个表格说明最大字符数。 |
| `LLM_TABLE_CAPTION_MAX_TABLES` | `0` | 每文档最多处理表格数；`0` 不限制。 |
| `LLM_TABLE_CAPTION_CONTEXT_LINES` | `3` | 表格前后上下文行数。 |
| `LLM_IMAGE_CAPTION` | `true` | 是否对插图做语义补充。 |
| `LLM_IMAGE_CAPTION_MAX_IMAGES` | `0` | 每文档最多处理图片数；`0` 不限制。 |
| `LLM_IMAGE_CAPTION_MAX_CHARS` | 默认同 `LLM_MAX_TOKENS` | 每张图说明长度上限。 |
| `LLM_IMAGE_CAPTION_CONTEXT_LINES` | `3` | 图片前后上下文行数。 |
| `PDF_CAPTION_CROP_FIGURES` | `true` | 是否按图题从页面渲染图中裁切局部插图。 |
| `PDF_CAPTION_CROP_MAX_PER_PAGE` | `4` | 每页最多裁切插图数。 |

### LLM 精修与表格纠错（进阶）

| 变量 | 默认 | 说明 |
|------|------|------|
| `LLM_ENABLE_REFINE` | `false` | Docling 结果后是否走 LLM 精修路径。 |
| `LLM_TABLE_REFINE` | `false` | 是否按 table 块调用模型纠错。 |
| `LLM_TABLE_CLEANUP_MAX_TABLES` | `10` | 最多处理多少个 table 块。 |
| `LLM_TABLE_CLEANUP_MAX_IMAGES_PER_TABLE` | `6` | 每个 table 最多附带图片数。 |
| `LLM_TABLE_CONTEXT_LINES` | `2` | table 块附加上下文行数。 |
| `LLM_ALLOW_RERUN` | `false` | 是否允许质量检查触发 Docling rerun。 |
| `LLM_RERUN_MAX_ATTEMPTS` | `1` | rerun 次数上限。 |

### 数据库与登录（本地账号）

| 变量 | 默认 | 说明 |
|------|------|------|
| `DB_TYPE` | `sqlite` | `sqlite` 或 `mysql`（未设置 `DATABASE_URL` 时用于拼接）。 |
| `DATABASE_URL` | 空 | 若设置则**优先**使用（如 `mysql+pymysql://user:pass@host:3306/db?charset=utf8mb4`）。 |
| `AUTH_DB_PATH` | `DATA_DIR/auth.db` | SQLite 文件路径（仅当使用 sqlite URL 时有效）。 |
| `MYSQL_USER` / `MYSQL_PASSWORD` / `MYSQL_HOST` / `MYSQL_PORT` / `MYSQL_DATABASE` | `root` / 空 / `127.0.0.1` / `3306` / `docling_demo` | 未设 `DATABASE_URL` 且 `DB_TYPE=mysql` 时拼装连接串。 |
| `SESSION_SECRET` | `change-me-in-production` | Cookie 会话签名密钥。 |
| `ACCESS_TOKEN_SECRET` | 空则同 `SESSION_SECRET` | JWT 签名密钥。 |
| `ACCESS_TOKEN_TTL_SEC` | `86400` | JWT 有效期（秒），下限 60。 |
| `INITIAL_PASSWORD` | `ChangeMe123!` | 首次初始化写入用户库的密码（存 PBKDF2 哈希）。 |
| `AUTH_ADMIN_USERNAME` | `admin` | 管理员用户名；OA 模式下同名账号可走「环境管理员」特例（见下节）。 |
| `AUTH_USERS` | `user1,user2,...` | 非 OA 模式下引导创建的用户列表。 |

### OA 登录（可选）

| 变量 | 默认 | 说明 |
|------|------|------|
| `OA_AUTH_ENABLED` | `false` | 为 `true` 时，普通用户登录走 OA 接口，不在本地 `users` 表校验密码。 |
| `OA_AUTH_LOGIN_URL` | 空 | OA 登录 POST 地址（需可访问）。 |
| `OA_AUTH_TENANT_ID` | `1` | 请求头 `tenant-id`。 |
| `OA_AUTH_TENANT_NAME` | 未设置时为四个空格 | JSON 体 `tenantName`；若需真正空字符串可设为 `OA_AUTH_TENANT_NAME=`。 |
| `OA_AUTH_REMEMBER_ME` | `true` | 请求体 `rememberMe`。 |
| `OA_AUTH_VERIFY_SSL` | `false` | HTTPS 是否校验证书。 |
| `OA_AUTH_TIMEOUT_SEC` | `15` | 请求超时（秒），下限 3。 |
| `OA_AUTH_ORIGIN` / `OA_AUTH_REFERER` / `OA_AUTH_USER_AGENT` | 空 | 未设置时由登录 URL 推导 Origin，Referer 默认为 `{origin}/login?redirect=/index`，UA 为内置浏览器串。 |
| `OA_AUTH_COOKIE` | 空 | 需要时可附加 `Cookie` 头。 |
| `OA_AUTH_TRUST_ENV` | `false` | 为 `false` 时**不使用**系统环境代理（避免内网 OA 被 `HTTP_PROXY` 拐走）；与 `oa_auth.py` 中说明一致。 |

### 日志（Web）

| 变量 | 说明 |
|------|------|
| `RUN_LOG_FILE` | 若设置，日志写入该文件；否则在 `LOG_DIR` 下按时间戳与 PID 生成。 |
| `LOG_DIR` | 默认 `DATA_DIR/logs`。 |
| `LOG_MAX_BYTES` | 单文件滚动上限，默认 `52428800`（50MB）。 |
| `LOG_BACKUP_COUNT` | 保留备份数，默认 `10`。 |

---

## 认证：本地用户与 OA

### 鉴权方式

受保护接口接受：

1. **`Authorization: Bearer <access_token>`**（登录接口返回的 JWT）；或  
2. **会话 Cookie**（登录成功后由 `SessionMiddleware` 下发）。

令牌校验失败会返回 `401`（过期、无效等文案见 `webapp._require_auth_user`）。

### 本地模式（`OA_AUTH_ENABLED=false`）

- 用户在 **`AUTH_USERS`** 列表中初始化；密码为 **`INITIAL_PASSWORD`**（哈希存入数据库）。
- **`/auth/users`** 仅 **admin** 可列出全部用户名。

### OA 模式（`OA_AUTH_ENABLED=true`）

- **普通用户**：用户名/密码由 **`authenticate_with_oa`**（`oa_auth.py`）POST 到 `OA_AUTH_LOGIN_URL`；成功则根据返回 JSON 解析用户名与是否 admin（`user.admin`、`roles` 等启发式），**不写本地 users 表**。
- **环境管理员特例**：用户名等于 **`AUTH_ADMIN_USERNAME`**（默认 `admin`）时，**不请求 OA**，仅在本地库校验密码（通过 `ensure_env_admin_user` 与 `INITIAL_PASSWORD` 同步的哈希）。用于运维兜底。

### 登录响应示例字段

`POST /auth/login` 返回：`username`、`role`、`access_token`、`token_type`（`bearer`）、`expires_in`（秒）等。

---

## 运行 Web 服务

```bash
python -m uvicorn webapp:app --host 0.0.0.0 --port 8000
```

- 首页：`GET /`（内嵌 `static/index.html` 内容）。
- 静态资源：`/static/...`。
- 健康检查：`GET /health`。

仓库中 **`start.sh`** / **`restart.sh`** 提供了 `nohup` 后台示例，可按路径与端口调整。

---

## HTTP API 参考

**说明**：除 `/`、`/static/*`、`/health`、`/app/config` 外，下列接口均需登录（Bearer 或 Session）。`job_id` 为 32 位十六进制小写字符串。

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/` | 返回前端 HTML。 |
| `GET` | `/health` | `{"status":"ok"}`。 |
| `GET` | `/app/config` | 公开：允许上传类型、最大字节数等。 |
| `POST` | `/auth/login` | JSON：`username`, `password`。 |
| `POST` | `/auth/logout` | 清除会话。 |
| `GET` | `/auth/me` | 当前用户与角色。 |
| `GET` | `/auth/bootstrap` | 当前用户 + 首页任务列表（分页参数固定实现内为第 1 页、100 条）。 |
| `GET` | `/auth/users` | **admin**：用户名列表。 |
| `POST` | `/jobs` | `multipart/form-data`：支持单文件、多文件、文件夹（`upload_kind=file|folder`，文件夹需 `files` + `relative_paths` 对齐）。 |
| `GET` | `/jobs` | 分页列表；query：`owner`（仅 admin）、`status`、`q`、`page`、`page_size`。 |
| `GET` | `/jobs/{job_id}` | 任务详情（含进度、队列位置、`pdf_vl_failed_pages` 等）。 |
| `POST` | `/jobs/{job_id}/cancel` | 取消排队中或运行中任务。 |
| `POST` | `/jobs/{job_id}/retry` | 失败/取消等终态后可重试；会清空产出目录并入队。 |
| `DELETE` | `/jobs/{job_id}` | 删除记录并清理工作区；**运行中**需先取消。 |
| `GET` | `/jobs/{job_id}/download` | 成功任务：ZIP 下载整个产出目录。 |
| `POST` | `/jobs/batch-download` | JSON：`{"job_ids":["..."]}`，最多 500 个，打包为 `jobs.zip`。 |
| `POST` | `/convert` | **兼容旧客户端**：单文件上传，内部创建任务并**阻塞轮询**直到终态，返回 `job_id`、`filename`、`download_url`（旧式 `/download/{job_id}`）。 |
| `GET` | `/download/{job_id}` | 同单任务下载逻辑（`FileResponse` ZIP）。 |

**任务 JSON 常见字段**（详见 `_job_to_api_dict`）：`job_id`、`owner_username`、`original_filename`、`status`、`created_at`、`started_at`、`finished_at`、`error_message`、`is_directory`、目录批处理的 `total_files` / `processed_files` / `succeeded_files` / `failed_files`、运行中 `progress_*`、成功时 `download_url`，以及可选的 `pdf_vl_failed_pages`、`failed_files_preview`。

---

## 任务状态与权限

- **状态**：`queued` → `running` → `succeeded` | `failed` | `cancelled`。
- **权限**：普通用户只能查看/操作/下载**自己**的任务；**admin** 可看全部，并可用 `owner` 筛选。
- **队列信息**：列表接口对 `queued` 任务返回 `queue_position`、`queue_total`（实现见 `auth_store.get_queue_positions`）。

---

## 批量转换 CLI（main.py）

入口会加载项目根目录 **`.env`**。完整参数以 **`python main.py -h`** 为准。下面按主题归纳。

### 输入输出

- `--input-dir` / `--output-dir`：默认 `data/input`、`data/output`。
- `--output-by-model`：在输出根下按模型名分子目录（纯 Docling 时为 `docling/`）。
- `--max-files`、`--skip-existing`、`--max-file-size`、`--max-num-pages`：批量控制。

### Docling 与资源

- `--no-ocr`、`--ocr-engine`、`--ocr-quality`、`--scan`、`--low-memory`、`--formula`、`--no-tables`、`--rich-images`、`--images-scale`。
- `--device` / `--cpu`、`--pipeline-concurrency`。
- `--keep-page-header-footer`、`--no-escape-dimension-asterisks`。

### LLM / PDF-VL

- `--pdf-vl-primary`：PDF 按页 VL 主路径。
- `--pdf-vl-dpi`、`--pdf-vl-workers`、`--no-pdf-vl-table-second-pass`、`--pdf-vl-table-second-pass-max-tables`。
- `--enable-llm`：对 Docling 产出做清洗/纠错。
- `--llm-model`、`--llm-base-url`、`--llm-temperature`、`--llm-max-tokens`、`--llm-enable-thinking` / `--llm-disable-thinking`。
- `--llm-table-caption` / `--no-llm-table-caption` 及表格说明数量、上下文行数。
- `--pdf-caption-crop-figures`、`--pdf-caption-crop-max-per-page`。
- `--llm-image-caption` 及图片数量/长度/上下文。
- `--llm-table-refine`、`--llm-table-max-tables` 等。
- `--llm-vl-image-mode`、`--llm-allow-rerun` 等。

### 示例

```bash
# 试跑：单文件、少量页、按模型分目录、按页 VL + 图题裁图
python main.py --pdf-vl-primary --output-by-model --pdf-caption-crop-figures --max-files 1 --max-num-pages 5
```

---

## GPU / PyTorch 提示

`main.py` 在请求 CUDA 时会检测 PyTorch 是否为 `+cpu` 构建、`torch.cuda.is_available()` 等，并打印升级建议（如 CUDA 12.8 轮子、RTX 50 系与 PyTorch 2.7+）。**无 NVIDIA 或仅用 CPU 版 PyTorch 时请使用 `--device auto` 或 `--cpu`**，避免 Docling 侧误选 CUDA。

---

## 日志与排错

- **Web**：默认日志在 `data/logs/webapp_*.log`（或 `RUN_LOG_FILE`）；uvicorn access 对任务轮询做了过滤，减少噪音。
- **OA 登录失败**：检查 `OA_AUTH_LOGIN_URL` 是否内网可达、是否需 `OA_AUTH_COOKIE`；若 Postman 正常而程序超时，注意 **`OA_AUTH_TRUST_ENV=false`** 时不走系统代理，反之若需走公司代理可设为 `true`。
- **本地 LLM**：适当增大 `LLM_TIMEOUT_SEC`，`LLM_ENABLE_THINKING=false` 往往可降低延迟；`LLM_MAX_TOKENS` 过大易导致单次生成过久。
- **Markdown 图片链接**：可使用 `scripts/smoke_check_md_images.py` 做本地路径检查（见脚本内说明）。

---

## 测试

```bash
pip install pytest
pytest
```

---

## 目录结构

```
├── main.py                 # 批量转换 CLI
├── webapp.py               # FastAPI 应用与路由
├── config.py               # AppConfig / 环境变量
├── service.py              # ConversionService、上传落盘
├── job_worker.py           # 异步任务 worker
├── auth.py                 # 用户、任务、队列位置（SQLAlchemy）
├── oa_auth.py              # OA 登录客户端
├── access_token.py         # JWT 创建与校验
├── static/                 # 前端资源
├── src/
│   ├── converter.py        # Docling + Excel 等主转换逻辑
│   ├── pdf_vl_transcribe.py
│   ├── dashscope_client.py
│   ├── llm_markdown_refiner.py
│   └── ...
├── tests/
├── scripts/
└── requirements.txt
```

---

## 许可证与第三方

本项目依赖 **Docling**、**MinerU（magic-pdf）**、**DashScope** 或其它兼容 API 提供商，各软件包与 API 受其各自协议与计费条款约束；部署前请确认合规与费用。
