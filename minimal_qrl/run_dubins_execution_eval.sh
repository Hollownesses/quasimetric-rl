#!/bin/bash
# Dubins UAV 上 QRL 执行策略评估脚本（greedy vs lookahead）
# 使用 minimal_qrl/eval/dubins_execution_mode_eval.py，对比：
#   - greedy：QRLGoalValueAdapter 自带的一步贪心策略
#   - lookahead：基于 QRL value 的多步 shooting / CEM planner
#
# 默认参数与 run_dubins_initial.sh 中的训练设置保持一致。
# 可通过环境变量覆盖部分参数，例如：
#   OUTPUT_DIR=./results/minimal_qrl_dubins_initial \
#   N_TRIALS=300 \
#   ./minimal_qrl/run_dubins_execution_eval.sh

set -e
cd "$(dirname "$0")/.."
echo "评估 Dubins UAV 执行策略（greedy vs lookahead）..."

python minimal_qrl/eval/dubins_execution_mode_eval.py \
  --checkpoint results/minimal_qrl_dubins_initial/checkpoint_final.pth \
  --output-dir results/minimal_qrl_dubins_initial \
  --bounds 0 0 5 5 --omega-max 3.0 --v 1.0 --dt 0.1 \
  --max-episode-steps 200 --epsilon-pos 0.15 --epsilon-theta 0.2 \
  --n-trials 100 --seed 0 \
  --lookahead-horizon 20 \
  --lookahead-num-sequences 512 \
  --lookahead-biased-sequences 96 \
  --lookahead-bias-kp 2.0


echo
echo "评估完成。结果保存在：results/minimal_qrl_dubins_initial/dubins_execution_mode_eval.json"

# 启用 CEM 参数：
# --lookahead-use-cem --lookahead-cem-iters 3 --lookahead-cem-elite-frac 0.1 --lookahead-cem-std-init-frac 0.5