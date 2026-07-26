#!/usr/bin/env bash
# 2×2 密度保持工业园区 QRL 扩展实验。
#
# 默认只训练一个 200 m × 200 m 场景：
#   4 个 100 m × 100 m 子园区
#   96 台设备、12 个固定物理尺寸障碍物
#   1 个位于完整园区中心的基站
#   200k 真实转移、200k 梯度更新（24设备基准的4倍）
#   每2k保存模型，在40k/80k/120k/160k/200k做Greedy验证
#   最终每设备5个独立随机测试起点，评估Greedy和MPPI
#
# 直接运行：
#   bash minimal_qrl/run_qrl_tiled_scalability_2x2.sh
#
# 常用替换：
#   # 同时重新训练1×1基准和2×2场景
#   TILE_GRIDS=1,2 OUTPUT_ROOT=results/qrl_tiled_scalability_1x1_2x2 \
#     bash minimal_qrl/run_qrl_tiled_scalability_2x2.sh
#
#   # 小预算冒烟实验；2×2实际预算会自动乘4
#   BASE_TARGET_ENV_TRANSITIONS=1000 BASE_TOTAL_STEPS=1000 \
#     BASE_CHECKPOINTS=500,1000 SAVE_INTERVAL=500 \
#     VALIDATION_PER_DEVICE=1 TEST_PER_DEVICE=1 \
#     OUTPUT_ROOT=results/qrl_tiled_scalability_2x2_smoke \
#     bash minimal_qrl/run_qrl_tiled_scalability_2x2.sh
#
#   # 正式三种子与正式测试量
#   SEEDS=0,1,2 VALIDATION_PER_DEVICE=10 TEST_PER_DEVICE=25 \
#     OUTPUT_ROOT=results/qrl_tiled_scalability_2x2_formal \
#     bash minimal_qrl/run_qrl_tiled_scalability_2x2.sh
#
# 注意：
#   - job串行运行，不启用并行训练。
#   - 200k训练每2k保存一次约产生100个周期checkpoint，预计占用约4.3 GiB。
#   - 96设备的最终MPPI测试可能明显慢于原24设备场景。
#   - 改变硬件、场景或预算时应使用新的OUTPUT_ROOT。

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif [[ -x "./.venv/bin/python" ]]; then
  PYTHON_BIN="./.venv/bin/python"
elif [[ -x "../quasimetric-rl/.venv/bin/python" ]]; then
  PYTHON_BIN="../quasimetric-rl/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

TILE_GRIDS="${TILE_GRIDS:-2}"
SEEDS="${SEEDS:-0}"

# 这些是单个100m/24设备子园区的基准预算；运行器按tile数量自动同比放大。
BASE_TARGET_ENV_TRANSITIONS="${BASE_TARGET_ENV_TRANSITIONS:-50000}"
BASE_TOTAL_STEPS="${BASE_TOTAL_STEPS:-50000}"
BASE_CHECKPOINTS="${BASE_CHECKPOINTS:-10000,20000,30000,40000,50000}"

SAVE_INTERVAL="${SAVE_INTERVAL:-2000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_CRITICS="${NUM_CRITICS:-2}"
TEACHER_RATIO="${TEACHER_RATIO:-1.0}"
VALIDATION_PER_DEVICE="${VALIDATION_PER_DEVICE:-3}"
TEST_PER_DEVICE="${TEST_PER_DEVICE:-5}"

DEVICE="${DEVICE:-cpu}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/qrl_tiled_scalability_2x2_seed0}"

echo "QRL tiled scalability experiment"
echo "  tile grids:                  ${TILE_GRIDS}"
echo "  seeds:                       ${SEEDS}"
echo "  base transitions / 24 dev:  ${BASE_TARGET_ENV_TRANSITIONS}"
echo "  base updates / 24 dev:      ${BASE_TOTAL_STEPS}"
echo "  base checkpoints:            ${BASE_CHECKPOINTS}"
echo "  save interval:               ${SAVE_INTERVAL}"
echo "  validation/test per device:  ${VALIDATION_PER_DEVICE}/${TEST_PER_DEVICE}"
echo "  device:                      ${DEVICE}"
echo "  output:                      ${OUTPUT_ROOT}"

"${PYTHON_BIN}" -m minimal_qrl.industry_exp.tiled_scalability run \
  --output-root "${OUTPUT_ROOT}" \
  --tile-grids "${TILE_GRIDS}" \
  --seeds "${SEEDS}" \
  --base-target-env-transitions "${BASE_TARGET_ENV_TRANSITIONS}" \
  --base-total-steps "${BASE_TOTAL_STEPS}" \
  --base-checkpoints "${BASE_CHECKPOINTS}" \
  --save-interval "${SAVE_INTERVAL}" \
  --validation-per-device "${VALIDATION_PER_DEVICE}" \
  --test-per-device "${TEST_PER_DEVICE}" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --num-critics "${NUM_CRITICS}" \
  --teacher-ratio "${TEACHER_RATIO}"

echo "实验完成：${OUTPUT_ROOT}"
echo "汇总报告：${OUTPUT_ROOT}/report/REPORT.md"
