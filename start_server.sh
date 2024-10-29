#!/bin/bash

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 激活虚拟环境（假设虚拟环境在 venv 目录下）
if [ -d "$SCRIPT_DIR/venv" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
else
    echo "警告: 虚拟环境不存在，尝试创建..."
    python3 -m venv "$SCRIPT_DIR/venv"
    source "$SCRIPT_DIR/venv/bin/activate"
fi

# 检查并安装依赖
if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
    echo "检查并安装依赖..."
    pip install -r "$SCRIPT_DIR/requirements.txt"
fi

# 设置其他环境变量（如果需要的话）
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

echo "环境已准备就绪" 