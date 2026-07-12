#!/usr/bin/env bash
# 通信感知巡检 Dubins UAV 环境上的 QRL 执行成功率评估脚本
#
# 直接运行：
#   bash minimal_qrl/run_comm_inspection_execution_eval.sh
#
# 常见覆盖方式：
#   OUTPUT_DIR=./results/minimal_qrl_inspection_dubins \
#   STARTS_PER_DEVICE=100 \
#   EXECUTION_MODES=greedy,lookahead \
#   LOOKAHEAD_HEURISTICS=terminal,dense \
#   PLANNER_QRL_PROGRESS_ALPHA=1.0 \
#   bash minimal_qrl/run_comm_inspection_execution_eval.sh

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -x "./.venv/bin/python" ]]; then
  PYTHON_BIN="./.venv/bin/python"
elif [[ -x "../quasimetric-rl/.venv/bin/python" ]]; then
  PYTHON_BIN="../quasimetric-rl/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/matplotlib}"

OUTPUT_DIR="${OUTPUT_DIR:-./results/goalset_qrl_comm_inspection}"
CHECKPOINT="${CHECKPOINT:-$OUTPUT_DIR/checkpoint_final.pth}"

BOUNDS="${BOUNDS:-0 0 10 10}"
DEVICE_CATALOG="${DEVICE_CATALOG:-./minimal_qrl/configs/industrial_site_devices.json}"
OBSTACLE_CONFIG="${OBSTACLE_CONFIG:-medium}"

REQUIRE_GROUND_STATION_LOS_FLAG=""
SAVE_VISUALIZATIONS_FLAG=""
VIZ_SAVE_GIF_FLAG=""

if [[ "${REQUIRE_GROUND_STATION_LOS:-0}" == "1" ]]; then
  REQUIRE_GROUND_STATION_LOS_FLAG="--require-ground-station-los"
fi
if [[ "${SAVE_VISUALIZATIONS:-0}" == "1" ]]; then
  SAVE_VISUALIZATIONS_FLAG="--save-visualizations"
fi
if [[ "${VIZ_SAVE_GIF:-0}" == "1" ]]; then
  VIZ_SAVE_GIF_FLAG="--viz-save-gif"
fi

echo "评估通信巡检 Dubins UAV 执行成功率..."

if [[ "${CLEAR_OLD_VISUALIZATIONS:-1}" == "1" ]]; then
  rm -rf "$OUTPUT_DIR/eval_results/visualizations"
fi

"$PYTHON_BIN" minimal_qrl/eval/comm_inspection_execution_eval.py \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR/eval_results" \
  --bounds ${BOUNDS} \
  --omega-max "${OMEGA_MAX:-3.0}" \
  --v "${V_FORWARD:-1.0}" \
  --dt "${DT:-0.1}" \
  --max-episode-steps "${MAX_STEPS_PER_EPISODE:-180}" \
  --obstacle-config "$OBSTACLE_CONFIG" \
  --device-catalog "$DEVICE_CATALOG" \
  --comm-alpha "${COMM_ALPHA:-2.0}" \
  --comm-bias "${COMM_BIAS:-5.0}" \
  --comm-occlusion-penalty "${COMM_OCCLUSION_PENALTY:-6.0}" \
  --comm-threshold "${COMM_THRESHOLD:-0.5}" \
  ${REQUIRE_GROUND_STATION_LOS_FLAG} \
  --collision-cost "${COLLISION_COST:-10.0}" \
  --out-of-bounds-cost "${OUT_OF_BOUNDS_COST:-10.0}" \
  --communication-break-cost "${COMM_BREAK_COST:-1.0}" \
  --observation-violation-cost-weight "${OBS_VIOLATION_COST_WEIGHT:-1.0}" \
  --communication-violation-cost-weight "${COMM_VIOLATION_COST_WEIGHT:-0.5}" \
  --observation-failure-cost "${OBSERVATION_FAILURE_COST:-0.25}" \
  --num-critics "${NUM_CRITICS:-2}" \
  --starts-per-device "${STARTS_PER_DEVICE:-50}" \
  --seed "${SEED:-0}" \
  --device "${DEVICE:-auto}" \
  --execution-modes "${EXECUTION_MODES:-greedy,lookahead}" \
  --lookahead-horizon "${LOOKAHEAD_HORIZON:-10}" \
  --lookahead-num-sequences "${LOOKAHEAD_NUM_SEQUENCES:-128}" \
  --lookahead-heuristics "${LOOKAHEAD_HEURISTICS:-dense}" \
  --lookahead-step-cost-weight "${LOOKAHEAD_STEP_COST_WEIGHT:-0.0}" \
  --lookahead-collision-penalty "${LOOKAHEAD_COLLISION_PENALTY:-0.0}" \
  --lookahead-biased-sequences "${LOOKAHEAD_BIASED_SEQUENCES:-24}" \
  --lookahead-bias-kp "${LOOKAHEAD_BIAS_KP:-2.0}" \
  --planner-qrl-progress-alpha "${PLANNER_QRL_PROGRESS_ALPHA:-1.0}" \
  ${SAVE_VISUALIZATIONS_FLAG} \
  --viz-max-successes "${VIZ_MAX_SUCCESSES:-10}" \
  --viz-max-failures "${VIZ_MAX_FAILURES:-10}" \
  ${VIZ_SAVE_GIF_FLAG} \
  --viz-gif-fps "${VIZ_GIF_FPS:-8}"

echo
echo "评估完成。结果保存在：$OUTPUT_DIR/eval_results/comm_inspection_execution_eval.json"
