#!/bin/bash

# 允许用户误用 `sh restart.sh`：自动切回 bash 再执行
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

# 定义变量方便维护
ENV_NAME="doc2md_env"
APP_DIR="/data/docling-demo"
LOG_FILE="app.log"

echo "=== 开始重启服务 ==="

# 1. 进入目录
cd "$APP_DIR" || { echo "错误: 找不到目录 $APP_DIR"; exit 1; }

# 2. 激活虚拟环境 (使用 source 确保在脚本内生效)
# 注意：如果你的 conda 安装路径不同，请检查 ~/anaconda3/etc/profile.d/conda.sh
CONDA_PATH="$(conda info --base 2>/dev/null)"
if [ -z "$CONDA_PATH" ] || [ ! -f "$CONDA_PATH/etc/profile.d/conda.sh" ]; then
  echo "错误: 未找到 conda.sh，请确认 conda 已安装且在 PATH 中"
  exit 1
fi
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate "$ENV_NAME" || { echo "错误: conda 环境 '$ENV_NAME' 激活失败"; exit 1; }

# 3. 杀掉旧进程
# 使用 pgrep 匹配启动命令，排除掉当前脚本进程
PID="$(ps -ef | grep "webapp:app" | grep -v grep | awk '{print $2}')"

if [ -n "$PID" ]; then
    echo "正在关闭旧进程 PID: $PID"
    kill -9 $PID
    sleep 2
else
    echo "未发现正在运行的旧进程"
fi

# 4. 守护启动 (使用 nohup)
echo "正在启动 webapp..."
nohup python -m uvicorn webapp:app --host 0.0.0.0 --port 8000 > "$LOG_FILE" 2>&1 &

if [ $? -eq 0 ]; then
    echo "服务已在后台启动，日志输出至 $LOG_FILE"
else
    echo "启动失败，请检查配置"
fi