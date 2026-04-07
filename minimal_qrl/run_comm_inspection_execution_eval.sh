#!/usr/bin/env bash
# 通信感知巡检 Dubins UAV 环境上的 QRL 执行成功率评估脚本
#
# 直接运行：
#   bash minimal_qrl/run_comm_inspection_execution_eval.sh
#
# 常见覆盖方式：
#   OUTPUT_DIR=./results/minimal_qrl_inspection_dubins \
#   N_TRIALS=300 \
#   EXECUTION_MODES=greedy,lookahead \
#   bash minimal_qrl/run_comm_inspection_execution_eval.sh

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -x "./.venv/bin/python" ]]; then
  PYTHON_BIN="./.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

OUTPUT_DIR="${OUTPUT_DIR:-./results/minimal_qrl_inspection_dubins}"
CHECKPOINT="${CHECKPOINT:-$OUTPUT_DIR/checkpoint_final.pth}"

BOUNDS="${BOUNDS:-0 0 10 10}"
INSPECTION_TARGET="${INSPECTION_TARGET:-3.0 7.5}"
GROUND_STATION="${GROUND_STATION:-1.5 2.0}"
OBSTACLE_CONFIG="${OBSTACLE_CONFIG:-medium}"

RANDOMIZE_INSPECTION_TARGET_FLAG=""
RANDOMIZE_GROUND_STATION_FLAG=""
REQUIRE_TARGET_LOS_FLAG="--require-target-los"
REQUIRE_GROUND_STATION_LOS_FLAG=""
APPLY_COMM_BREAK_PENALTY_FLAG="--apply-communication-break-penalty"
SAVE_VISUALIZATIONS_FLAG=""
VIZ_SAVE_FAILURES_FLAG="--viz-save-failures"
VIZ_SAVE_GIF_FLAG=""

if [[ "${RANDOMIZE_INSPECTION_TARGET:-0}" == "1" ]]; then
  RANDOMIZE_INSPECTION_TARGET_FLAG="--randomize-inspection-target"
fi
if [[ "${RANDOMIZE_GROUND_STATION:-0}" == "1" ]]; then
  RANDOMIZE_GROUND_STATION_FLAG="--randomize-ground-station"
fi
if [[ "${REQUIRE_TARGET_LOS:-1}" == "1" ]]; then
  REQUIRE_TARGET_LOS_FLAG="--require-target-los"
else
  REQUIRE_TARGET_LOS_FLAG="--no-require-target-los"
fi
if [[ "${REQUIRE_GROUND_STATION_LOS:-0}" == "1" ]]; then
  REQUIRE_GROUND_STATION_LOS_FLAG="--require-ground-station-los"
fi
if [[ "${APPLY_COMM_BREAK_PENALTY:-1}" == "1" ]]; then
  APPLY_COMM_BREAK_PENALTY_FLAG="--apply-communication-break-penalty"
else
  APPLY_COMM_BREAK_PENALTY_FLAG="--no-apply-communication-break-penalty"
fi
if [[ "${SAVE_VISUALIZATIONS:-0}" == "1" ]]; then
  SAVE_VISUALIZATIONS_FLAG="--save-visualizations"
fi
if [[ "${VIZ_SAVE_FAILURES:-1}" == "1" ]]; then
  VIZ_SAVE_FAILURES_FLAG="--viz-save-failures"
else
  VIZ_SAVE_FAILURES_FLAG="--no-viz-save-failures"
fi
if [[ "${VIZ_SAVE_GIF:-0}" == "1" ]]; then
  VIZ_SAVE_GIF_FLAG="--viz-save-gif"
fi

echo "评估通信巡检 Dubins UAV 执行成功率..."

"$PYTHON_BIN" minimal_qrl/eval/comm_inspection_execution_eval.py \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR/eval_results" \
  --bounds ${BOUNDS} \
  --omega-max "${OMEGA_MAX:-3.0}" \
  --v "${V_FORWARD:-1.0}" \
  --dt "${DT:-0.1}" \
  --max-episode-steps "${MAX_STEPS_PER_EPISODE:-180}" \
  --obstacle-config "$OBSTACLE_CONFIG" \
  --observation-mode "${OBSERVATION_MODE:-task_context}" \
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
  --goal-position-tolerance "${GOAL_POS_TOL:-0.15}" \
  --goal-heading-tolerance "${GOAL_HEADING_TOL:-0.2}" \
  --collision-penalty "${COLLISION_PENALTY:--10.0}" \
  --out-of-bounds-penalty "${OUT_OF_BOUNDS_PENALTY:--10.0}" \
  --communication-break-penalty "${COMM_BREAK_PENALTY:--1.0}" \
  ${APPLY_COMM_BREAK_PENALTY_FLAG} \
  --reward-obs-weight "${REWARD_OBS_WEIGHT:-1.0}" \
  --reward-comm-weight "${REWARD_COMM_WEIGHT:-0.5}" \
  --reward-task-feasible-bonus "${REWARD_TASK_FEASIBLE_BONUS:-1.0}" \
  --reward-goal-success-bonus "${REWARD_GOAL_SUCCESS_BONUS:-1.0}" \
  --num-critics "${NUM_CRITICS:-2}" \
  --n-trials "${N_TRIALS:-100}" \
  --seed "${SEED:-0}" \
  --device "${DEVICE:-auto}" \
  --execution-modes "${EXECUTION_MODES:-greedy,lookahead}" \
  --lookahead-horizon "${LOOKAHEAD_HORIZON:-10}" \
  --lookahead-num-sequences "${LOOKAHEAD_NUM_SEQUENCES:-128}" \
  --lookahead-step-cost-weight "${LOOKAHEAD_STEP_COST_WEIGHT:-0.0}" \
  --lookahead-collision-penalty "${LOOKAHEAD_COLLISION_PENALTY:-0.0}" \
  --lookahead-biased-sequences "${LOOKAHEAD_BIASED_SEQUENCES:-24}" \
  --lookahead-bias-kp "${LOOKAHEAD_BIAS_KP:-2.0}" \
  ${SAVE_VISUALIZATIONS_FLAG} \
  --viz-num-samples "${VIZ_NUM_SAMPLES:-3}" \
  ${VIZ_SAVE_FAILURES_FLAG} \
  --viz-max-failures "${VIZ_MAX_FAILURES:-10}" \
  ${VIZ_SAVE_GIF_FLAG} \
  --viz-gif-fps "${VIZ_GIF_FPS:-8}"

echo
echo "评估完成。结果保存在：$OUTPUT_DIR/eval_results/comm_inspection_execution_eval.json"
