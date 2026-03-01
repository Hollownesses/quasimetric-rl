#!/bin/bash
# Dubins UAV 初步 QRL 训练：无障碍、小地图、固定 v/dt、大 omega_max、
# 随机 multi-goal 采样，(x,y,cosθ,sinθ) 观测，验证 loss 收敛与 TD 曲线
# 监控：loss 收敛、TD error 震荡、时间代价场热力图（TensorBoard）

set -e
cd "$(dirname "$0")/.."

OUTPUT_DIR="${OUTPUT_DIR:-./results/minimal_qrl_dubins_initial}"
mkdir -p "$OUTPUT_DIR"

python minimal_qrl/train.py \
    --device auto \
    --env-type dubins_uav \
    --output-dir "$OUTPUT_DIR" \
    --bounds 0 0 5 5 \
    --omega-max 3.0 \
    --v 1.0 \
    --dt 0.1 \
    --use-cos-sin-obs \
    --num-episodes 150 \
    --max-steps-per-episode 150 \
    --batch-size 256 \
    --total-steps 16000 \
    --num-critics 2 \
    --log-interval 100 \
    --save-interval 2000 \
    --eval-interval 1000 \
    --eval-n-pairs 400 \
    --visualization-interval 1000

echo "训练结束。查看 TensorBoard: tensorboard --logdir=$OUTPUT_DIR/tensorboard"
echo "  - train/total_loss, train/one_step_dist, train/td_like_error"
echo "  - eval/ 下 mse, spearman_corr 等"
echo "  - eval/distance_heatmap 时间代价场"
