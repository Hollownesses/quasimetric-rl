#!/usr/bin/env bash
# 通信感知巡检 Dubins UAV 环境上的 QRL 训练脚本
#
# 直接运行：
#   bash minimal_qrl/run_comm_inspection_train.sh
#
# 常见覆盖方式：
#   OUTPUT_DIR=./results/minimal_qrl_inspection_dubins_exp1 \
#   INSPECTION_TARGET="7.5 6.5" \
#   GROUND_STATION="1.5 2.0" \
#   OBSTACLE_CONFIG=medium \
#   bash minimal_qrl/run_comm_inspection_train.sh
#
# 若希望 reset 时随机采样巡检目标 / 地面站，可设置：
#   RANDOMIZE_INSPECTION_TARGET=1 RANDOMIZE_GROUND_STATION=1 bash minimal_qrl/run_comm_inspection_train.sh
#
# QRL local constraint 单步代价来源：
#   QRL_COST_SOURCE=fixed            # 默认：使用原始固定 step_cost=1.0
#   QRL_COST_SOURCE=negative_reward  # 使用环境 task cost

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -x "./.venv/bin/python" ]]; then
  PYTHON_BIN="./.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

OUTPUT_DIR="${OUTPUT_DIR:-./results/minimal_qrl_inspection_dubins}"
mkdir -p "$OUTPUT_DIR"

BOUNDS="${BOUNDS:-0 0 10 10}"
INSPECTION_TARGET="${INSPECTION_TARGET:-3.0 7.5}"
GROUND_STATION="${GROUND_STATION:-1.5 2.0}"
OBSTACLE_CONFIG="${OBSTACLE_CONFIG:-medium}"

RANDOMIZE_INSPECTION_TARGET_FLAG=""
RANDOMIZE_GROUND_STATION_FLAG=""
REQUIRE_TARGET_LOS_FLAG=""
REQUIRE_GROUND_STATION_LOS_FLAG=""

if [[ "${RANDOMIZE_INSPECTION_TARGET:-0}" == "1" ]]; then
  RANDOMIZE_INSPECTION_TARGET_FLAG="--randomize-inspection-target"
fi
if [[ "${RANDOMIZE_GROUND_STATION:-0}" == "1" ]]; then
  RANDOMIZE_GROUND_STATION_FLAG="--randomize-ground-station"
fi
if [[ "${REQUIRE_TARGET_LOS:-1}" == "1" ]]; then
  REQUIRE_TARGET_LOS_FLAG="--require-target-los"
fi
if [[ "${REQUIRE_GROUND_STATION_LOS:-0}" == "1" ]]; then
  REQUIRE_GROUND_STATION_LOS_FLAG="--require-ground-station-los"
fi

"$PYTHON_BIN" minimal_qrl/train.py \
  --device auto \
  --env-type comm_inspection_dubins_uav \
  --output-dir "$OUTPUT_DIR" \
  --bounds ${BOUNDS} \
  --omega-max "${OMEGA_MAX:-3.0}" \
  --v "${V_FORWARD:-1.0}" \
  --dt "${DT:-0.1}" \
  --observation-mode task_context \
  --obstacle-config "$OBSTACLE_CONFIG" \
  --num-episodes "${NUM_EPISODES:-180}" \
  --max-steps-per-episode "${MAX_STEPS_PER_EPISODE:-180}" \
  --batch-size "${BATCH_SIZE:-256}" \
  --total-steps "${TOTAL_STEPS:-20000}" \
  --num-critics "${NUM_CRITICS:-2}" \
  --qrl-cost-source "${QRL_COST_SOURCE:-fixed}" \
  --log-interval "${LOG_INTERVAL:-100}" \
  --save-interval "${SAVE_INTERVAL:-2000}" \
  --eval-interval "${EVAL_INTERVAL:-1000}" \
  --eval-n-pairs "${EVAL_N_PAIRS:-400}" \
  --visualization-interval "${VIS_INTERVAL:-1000}" \
  --planning-eval-interval 0 \
  --inspection-target ${INSPECTION_TARGET} \
  --ground-station ${GROUND_STATION} \
  ${RANDOMIZE_INSPECTION_TARGET_FLAG} \
  ${RANDOMIZE_GROUND_STATION_FLAG} \
  --observation-radius "${OBS_RADIUS:-1.8}" \
  --fov-angle "${FOV_ANGLE:-1.5707963267948966}" \
  ${REQUIRE_TARGET_LOS_FLAG} \
  --comm-alpha "${COMM_ALPHA:-2.0}" \
  --comm-bias "${COMM_BIAS:-5.0}" \
  --comm-occlusion-penalty "${COMM_OCCLUSION_PENALTY:-6.0}" \
  --comm-threshold "${COMM_THRESHOLD:-0.5}" \
  ${REQUIRE_GROUND_STATION_LOS_FLAG} \
  --goal-sampling-mode "${GOAL_SAMPLING_MODE:-task_feasible}" \
  --goal-position-tolerance "${GOAL_POS_TOL:-0.25}" \
  --goal-heading-tolerance "${GOAL_HEADING_TOL:-0.3}" \
  --collision-cost "${COLLISION_COST:-10.0}" \
  --out-of-bounds-cost "${OUT_OF_BOUNDS_COST:-10.0}" \
  --communication-break-cost "${COMM_BREAK_COST:-1.0}" \
  --observation-violation-cost-weight "${OBS_VIOLATION_COST_WEIGHT:-1.0}" \
  --communication-violation-cost-weight "${COMM_VIOLATION_COST_WEIGHT:-0.5}" \
  --observation-failure-cost "${OBSERVATION_FAILURE_COST:-0.25}"

echo "训练完成。结果目录: $OUTPUT_DIR"
echo "TensorBoard:"
echo "  $PYTHON_BIN -m tensorboard.main --logdir=$OUTPUT_DIR/tensorboard"
echo "检查点:"
echo "  $OUTPUT_DIR/checkpoint_final.pth"
