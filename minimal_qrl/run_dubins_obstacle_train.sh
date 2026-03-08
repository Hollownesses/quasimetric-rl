#!/bin/bash
# 在有障碍 Dubins 环境下训练 QRL
# 用法：OBSTACLE=simple|medium|hard ./minimal_qrl/run_dubins_obstacle_train.sh
# 默认障碍预设：simple（单圆）。可改为 medium / hard 或留空用自定义 --obstacles

set -e
cd "$(dirname "$0")/.."

OBSTACLE_CONFIG="${OBSTACLE_CONFIG:-medium}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/minimal_qrl_dubins_obstacle}"
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
    --obstacle-config "$OBSTACLE_CONFIG" \
    --collision-penalty -10.0 \
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

echo "训练结束。checkpoint 与 TensorBoard: $OUTPUT_DIR"
echo "  tensorboard --logdir=$OUTPUT_DIR/tensorboard"
