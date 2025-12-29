#!/bin/bash
# 一键运行最小 QRL 训练脚本

python minimal_qrl/train.py \
    --seed 42 \
    --device cpu \
    --output-dir ./results/minimal_qrl \
    --grid-size 10 10 \
    --num-episodes 100 \
    --max-steps-per-episode 200 \
    --batch-size 256 \
    --total-steps 10000 \
    --num-critics 2 \
    --log-interval 100 \
    --save-interval 1000

