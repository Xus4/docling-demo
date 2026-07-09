# 项目长期记忆

## 环境信息
- 运行服务的 Python：`D:\python3.13\python.exe`（安装了 uvicorn 等依赖）
- 启动命令：`D:/python3.13/python.exe -m uvicorn webapp:app --host 0.0.0.0 --port 8000`
- managed Python（`C:\Users\SFJT\.workbuddy\binaries\python\...`）无项目依赖，不能直接运行服务
- index.html 在服务启动时一次性读入内存，修改后需重启服务才能生效
- DB_TYPE=mysql，数据库地址 192.168.2.60:3306

## 依赖版本要求
- fastapi 必须 ≥ 0.116.0：starlette 1.x 移除了 `Router.__init__` 的 `on_startup`/`on_shutdown` 参数，旧版 fastapi（如 0.115.6）内部仍传这俩参数会导致启动报 `TypeError: Router.__init__() got an unexpected keyword argument 'on_startup'`。当前已升级到 fastapi 0.139.0 + starlette 1.3.1，正常。requirements.txt 写的是 `fastapi>=0.110.0`，未设上限，重装环境时注意。

## 已知问题
- `/health` 端点始终报 `db: unreachable`：代码用 `auth_store._engine`（下划线前缀），但 AuthStore 属性名是 `self.engine`（无下划线），不影响实际功能

## 技术栈
- FastAPI + SQLAlchemy + MySQL/SQLite
- 前端：单文件 SPA（index.html 内嵌 JS/CSS），vendor 目录有 jit-viewer 组件
- 文档转换：MinerU API（hybrid-auto-engine），表格语义增强（OpenAI-compatible LLM）
