#!/usr/bin/env bash
# QRL 空间尺度/设备数单种子趋势实验。
#
# 默认场景：
#   面积轴：100 m、300 m、1000 m，均为 K=24
#   设备轴：K=4、12、24，均为 100 m
#   100 m / K=24 只训练一次，所以 seed=0 时共有5个 job。
#
# 直接运行：
#   bash minimal_qrl/run_qrl_scalability_pilot.sh
#
# 常用替换示例：
#   # 正式三种子（每个场景3个模型）
#   SEEDS=0,1,2 bash minimal_qrl/run_qrl_scalability_pilot.sh
#
#   # 只跑面积两个端点，仍保留设备数轴
#   AREA_SIDES=100,1000 bash minimal_qrl/run_qrl_scalability_pilot.sh
#
#   # 20k 快速趋势版（总步数与评估点必须一起改）
#   TARGET_ENV_TRANSITIONS=20000 TOTAL_STEPS=20000 CHECKPOINTS=10000,20000 \
#     OUTPUT_ROOT=results/qrl_scalability_pilot_20k \
#     bash minimal_qrl/run_qrl_scalability_pilot.sh
#
#   # 正式评估量；默认的 3/5 只用于节省初步趋势实验时间
#   VALIDATION_PER_DEVICE=10 TEST_PER_DEVICE=25 \
#     OUTPUT_ROOT=results/qrl_scalability_formal \
#     bash minimal_qrl/run_qrl_scalability_pilot.sh
#
#   # 修改模型保存间隔（默认每2000 step保存）
#   SAVE_INTERVAL=5000 bash minimal_qrl/run_qrl_scalability_pilot.sh
#
# 请勿在同一 OUTPUT_ROOT 中混用 CPU/MPS 或修改场景集合。
# 单种子报告可用于看趋势，但 usable 会保持 false；正式判定仍要求3种子。

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

# 本次初步实验的场景选择。
AREA_SIDES="${AREA_SIDES:-100,300,1000}"
DEVICE_COUNTS="${DEVICE_COUNTS:-4,12,24}"
SEEDS="${SEEDS:-0}"

# 初步训练配置：50k真实转移和50k梯度更新。
TARGET_ENV_TRANSITIONS="${TARGET_ENV_TRANSITIONS:-50000}"
TOTAL_STEPS="${TOTAL_STEPS:-50000}"
# 保存模型与验证调度分离：每2k保存，每10k评估一次。
SAVE_INTERVAL="${SAVE_INTERVAL:-2000}"
CHECKPOINTS="${CHECKPOINTS:-10000,20000,30000,40000,50000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_CRITICS="${NUM_CRITICS:-2}"
TEACHER_RATIO="${TEACHER_RATIO:-1.0}"

# 初步实验降低评估起点数；训练预算未缩减。
VALIDATION_PER_DEVICE="${VALIDATION_PER_DEVICE:-3}"
TEST_PER_DEVICE="${TEST_PER_DEVICE:-5}"

# M1 Pro 默认用 CPU；若你已确认 MPS 更快，可用 DEVICE=mps 覆盖。
DEVICE="${DEVICE:-cpu}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/qrl_scalability_pilot_100_300_1000_k4_12_24_seed0}"

echo "QRL scalability pilot"
echo "  area sides (m): ${AREA_SIDES}"
echo "  device counts:  ${DEVICE_COUNTS}"
echo "  seeds:          ${SEEDS}"
echo "  steps:          ${TOTAL_STEPS}"
echo "  transitions:    ${TARGET_ENV_TRANSITIONS}"
echo "  save interval:  ${SAVE_INTERVAL}"
echo "  device:         ${DEVICE}"
echo "  output:         ${OUTPUT_ROOT}"

"${PYTHON_BIN}" -m minimal_qrl.industry_exp.scalability run \
  --output-root "${OUTPUT_ROOT}" \
  --area-sides "${AREA_SIDES}" \
  --device-counts "${DEVICE_COUNTS}" \
  --seeds "${SEEDS}" \
  --target-env-transitions "${TARGET_ENV_TRANSITIONS}" \
  --total-steps "${TOTAL_STEPS}" \
  --save-interval "${SAVE_INTERVAL}" \
  --checkpoints "${CHECKPOINTS}" \
  --validation-per-device "${VALIDATION_PER_DEVICE}" \
  --test-per-device "${TEST_PER_DEVICE}" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --num-critics "${NUM_CRITICS}" \
  --teacher-ratio "${TEACHER_RATIO}"

echo "实验完成：${OUTPUT_ROOT}"
echo "汇总报告：${OUTPUT_ROOT}/report/REPORT.md"
