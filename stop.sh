#!/bin/bash

# 定义变量方便维护
ENV_NAME="doc2md_env"
APP_DIR="/data/docling-demo"
LOG_FILE="app.log"

echo "=== 开始停止服务 ==="

# 1. 进入目录
cd $APP_DIR || { echo "错误: 找不到目录 $APP_DIR"; exit 1; }

# 2. 激活虚拟环境（这一步其实 stop 不一定必须，但为了和 restart 风格统一可以保留）
CONDA_PATH=$(conda info --base)
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# 3. 杀掉旧进程
PID=$(ps -ef | grep "webapp:app" | grep -v grep | awk '{print $2}')

if [ -n "$PID" ]; then
    echo "正在关闭进程 PID: $PID"
    kill $PID
    sleep 2

    STILL_RUNNING=$(ps -ef | grep "webapp:app" | grep -v grep | awk '{print $2}')
    if [ -n "$STILL_RUNNING" ]; then
        echo "进程未正常退出，强制关闭 PID: $STILL_RUNNING"
        kill -9 $STILL_RUNNING
    fi

    echo "服务已停止"
else
    echo "未发现正在运行的旧进程"
fi
