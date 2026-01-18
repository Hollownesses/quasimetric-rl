#!/bin/bash
# 一键运行最小 QRL 训练脚本
# 支持 Apple Silicon MPS 加速

# 默认配置（自动检测最佳设备：MPS > CUDA > CPU）
python minimal_qrl/train.py \
    --seed 42 \
    --device auto \
    --output-dir ./results/minimal_qrl \
    --grid-size 10 10 \
    --num-episodes 100 \
    --max-steps-per-episode 200 \
    --batch-size 256 \
    --total-steps 5000 \
    --num-critics 2 \
    --log-interval 100 \
    --save-interval 1000 \
    --eval-interval 1000 \
    --eval-n-pairs 500 \
    --visualization-interval 1000

# 如果需要更快的训练（减少评估和可视化频率）:
# python minimal_qrl/train.py \
#     --device auto \
#     --batch-size 512 \
#     --eval-interval 2000 \
#     --eval-n-pairs 300 \
#     --visualization-interval 2000
