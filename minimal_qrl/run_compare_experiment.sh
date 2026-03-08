#!/usr/bin/env bash
# Dubins UAV：QRL vs TD-based goal-conditioned RL 统一对比实验
#
# - 复用现有 QRL checkpoint（不改动 QRL 训练逻辑）
# - 在同一 Dubins 环境上训练：
#     1) HER + DDPG
#     2) Goal-conditioned SAC
#     3) UVFA-style value learning
# - 将三种对比算法的结果与 QRL 的结果进行对比
#
# 运行后会得到：
#   results/minimal_qrl_dubins_benchmark/all_algorithms_metrics.json
#   results/minimal_qrl_dubins_benchmark/summary_table.csv
# 每个算法子目录：
#   her_ddpg/metrics.json, gc_sac/metrics.json, uvfa/metrics.json；qrl 在综合 json 中。
#
# 障碍环境：可加 --obstacle-config simple|medium|hard 或 --obstacles x1 y1 r1 x2 y2 r2 ...
# 例：--obstacle-config simple  或  --obstacles 2.5 2.5 0.5 1 4 0.4

set -e
cd "$(dirname "$0")/.."

python -m minimal_qrl.run_dubins_gc_experiments \
    --qrl-ckpt results/minimal_qrl_dubins_initial/checkpoint_final.pth \
    --her-ddpg-ckpt results/minimal_qrl_dubins_benchmark/her_ddpg/checkpoint_final.pth \
    --gc-sac-ckpt results/minimal_qrl_dubins_benchmark/gc_sac/checkpoint_final.pth \
    --uvfa-ckpt results/minimal_qrl_dubins_benchmark/uvfa/checkpoint_final.pth \
    --output-dir results/minimal_qrl_dubins_benchmark \
    --total-env-steps 160000 \
    --bounds 0 0 5 5 \
    --omega-max 3.0 \
    --v 1.0 \
    --dt 0.1 \
    --max-episode-steps 200 \
    --epsilon-pos 0.15 \
    --epsilon-theta 0.2 \
    --eval-n-trials 100 \
    --eval-n-pairs 1500 \
    --r-train 1.5 \
    --r-test 2.0 \
    --eval-lookahead --lookahead-horizon 20 --lookahead-num-sequences 512 --lookahead-biased-sequences 96 --lookahead-bias-kp 2.0