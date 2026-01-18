#!/bin/bash
# 自动启动最新训练的 TensorBoard
# 使用方法: ./view_latest_tensorboard.sh

TENSORBOARD_BASE_DIR="results/minimal_qrl/tensorboard"

# 检查目录是否存在
if [ ! -d "$TENSORBOARD_BASE_DIR" ]; then
    echo "错误: TensorBoard 目录不存在: $TENSORBOARD_BASE_DIR"
    echo "请先运行训练脚本生成日志文件"
    exit 1
fi

# 查找最新的子目录
LATEST_DIR=$(find "$TENSORBOARD_BASE_DIR" -mindepth 1 -maxdepth 1 -type d | sort -r | head -n 1)

if [ -z "$LATEST_DIR" ]; then
    echo "错误: 在 $TENSORBOARD_BASE_DIR 中未找到任何训练日志"
    exit 1
fi

echo "找到最新的训练日志目录: $LATEST_DIR"
echo "启动 TensorBoard..."
echo "请在浏览器打开 http://localhost:6006"
echo ""

# 启动 TensorBoard
tensorboard --logdir="$LATEST_DIR" --port=6006
