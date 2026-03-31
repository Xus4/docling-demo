#!/bin/bash

APP_DIR="/data/docling-demo"

echo "=== 开始拉取更新 ==="

# 1. 进入目录执行 git pull
cd $APP_DIR || { echo "错误: 找不到目录 $APP_DIR"; exit 1; }
git pull

if [ $? -eq 0 ]; then
    echo "代码已更新，准备重启..."
    # 2. 调用刚才写好的重启脚本
    bash ./restart.sh
else
    echo "Git pull 失败，停止更新"
    exit 1
fi