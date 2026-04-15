# 智枢文档（docling-demo）

FastAPI Web：**登录与会话**、**任务队列**、**上传 / 下载 ZIP**。文档解析通过 **MinerU HTTP API**（与官方 `mineru-api` 的 `/tasks`、`/file_parse`、`/health` 等路由兼容，见 `src/mineru_client.py`）。

## 运行

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
python -m uvicorn webapp:app --host 0.0.0.0 --port 8000
```

根目录 `.env` 由 `python-dotenv` 加载（`override=false`）。生产环境请修改 `SESSION_SECRET`、`ACCESS_TOKEN_SECRET`、`INITIAL_PASSWORD`。

## MinerU 对接

各变量在仓库根目录 **`.env`** 中有逐项中文注释（与 `mineru-api` 表单字段一致）；下表为速查。

| 变量 | 默认 | 说明 |
|------|------|------|
| **`MINERU_BASE_URL`** | `http://192.168.2.60:8011` | `mineru-api` 根 URL（无尾斜杠）；可通过环境变量覆盖。 |
| `MINERU_API_KEY` | 空 | 可选。若网关要求鉴权，将设置 `Authorization: Bearer …`。 |
| `MINERU_PARSE_MODE` | `async` | `async`：先 `POST /tasks` 再轮询 `GET /tasks/{id}` 与 `GET /tasks/{id}/result`；`sync`：单次 `POST /file_parse` 阻塞直至完成。 |
| `MINERU_TIMEOUT_SEC` | `300` | 单次 HTTP 超时（秒）。 |
| `MINERU_POLL_INTERVAL_SEC` | `1.5` | 异步模式下轮询状态间隔。 |
| `MINERU_MAX_WAIT_SEC` | `3600` | 异步模式下等待终态的最长时间。 |
| `MINERU_VERIFY_SSL` | `true` | HTTPS 是否校验证书。 |
| `MINERU_BACKEND` | `hybrid-auto-engine` | 与 MinerU 表单 `backend` 一致。 |
| `MINERU_PARSE_METHOD` | `auto` | 与 MinerU 表单 `parse_method` 一致（`auto`/`txt`/`ocr`）。 |
| `MINERU_LANG_LIST` | `ch` | 逗号分隔，对应多个 `lang_list` 表单字段。 |
| `MINERU_FORMULA_ENABLE` | `true` | 表单布尔。 |
| `MINERU_TABLE_ENABLE` | `true` | 表单布尔。 |
| `MINERU_SERVER_URL` | 空 | 仅 `-http-client` 类 backend 时填写。 |
| `MINERU_RETURN_MD` | `true` | 为 `true` 时从 JSON `results[*].md_content` 取正文；若 `MINERU_RESPONSE_FORMAT_ZIP=true` 则解析 ZIP 内 `.md`。 |
| `MINERU_RESPONSE_FORMAT_ZIP` | `false` | 与 MinerU 的 `response_format_zip` 一致。 |
| `MINERU_RETURN_*` | 多为 `false` | 其余 `return_middle_json` 等与官方 OpenAPI 表单一致。 |
| `MINERU_START_PAGE_ID` / `MINERU_END_PAGE_ID` | `0` / `99999` | PDF 页范围（MinerU 约定从 0 起）。 |

默认指向内网 **`http://192.168.2.60:8011`**；其他环境请用 `MINERU_BASE_URL` 覆盖。

## 其他配置

| 主题 | 环境变量示例 |
|------|----------------|
| 数据目录 | `DATA_DIR`、`INPUT_DIR`、`OUTPUT_DIR` |
| 上传 | `MAX_FILE_SIZE`、`ALLOWED_TYPES` |
| 数据库 | `DB_TYPE`、`DATABASE_URL` / `AUTH_DB_PATH` / `MYSQL_*` |
| 认证 | `SESSION_SECRET`、`ACCESS_TOKEN_*`、`AUTH_USERS`、`OA_AUTH_*` |
| Worker | `WORKER_MAX_PARALLEL_JOBS` |
| 日志 | `RUN_LOG_FILE`、`LOG_DIR`、`LOG_*`（见 `webapp.py`） |

## Docker

1. 复制环境模板（仓库根目录）：
   - `cp .env.docker.example .env.docker`
   - `cp .env.docker.mysql.example .env.docker.mysql`（或按需改账号口令）
2. 确认 **`MINERU_BASE_URL`**（示例与代码默认均为 `http://192.168.2.60:8011`；Docker 容器须能路由到该内网地址）。
3. 修改默认的 `SESSION_SECRET`、`INITIAL_PASSWORD`、数据库口令等。

```bash
docker compose up -d
```

说明：`docker-compose.yml` 将 `./data`、`./logs` 挂载到容器；应用连接 MySQL 时使用服务名 **`mysql`**（见 `DATABASE_URL` 示例）。

## 测试

```bash
pip install pytest
pytest
```

## 实现说明

`ConversionService.convert_to_markdown` 调用 `run_mineru_convert`：按配置提交文件、等待完成，将得到的 Markdown 写入任务的 `output_path`。任务取消时会在轮询间隙检测 `cancel_check` 并中断。

对外 HTTP 仅保留「登录 + 任务 + 下载」模型：**不提供**旧版阻塞式 `POST /convert` 与别名 `GET /download/{job_id}`；下载请使用 `GET /jobs/{job_id}/download` 或批量接口。
