#!/usr/bin/env bash
# Group 2: ablate task-aware QRL loss on the communication-inspection Dubins task.
#
# This script trains/evaluates two QRL critics under the same environment and
# planner settings:
#   1. Original QRL:     --qrl-cost-source fixed
#   2. Task-aware QRL:   --qrl-cost-source negative_reward

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -x "./.venv/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-./.venv/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

OUTPUT_DIR="${OUTPUT_DIR:-./results/experiments/comm_inspection_task_aware_ablation}"
ORIGINAL_DIR="${ORIGINAL_DIR:-$OUTPUT_DIR/qrl_original_fixed_cost}"
TASK_AWARE_DIR="${TASK_AWARE_DIR:-$OUTPUT_DIR/qrl_task_aware_reward_cost}"
mkdir -p "$OUTPUT_DIR"

BOUNDS="${BOUNDS:-0 0 10 10}"
INSPECTION_TARGET="${INSPECTION_TARGET:-3.0 7.5}"
GROUND_STATION="${GROUND_STATION:-1.5 2.0}"
OBSTACLE_CONFIG="${OBSTACLE_CONFIG:-medium}"

RANDOMIZE_INSPECTION_TARGET_FLAG=""
RANDOMIZE_GROUND_STATION_FLAG=""
REQUIRE_TARGET_LOS_FLAG="--require-target-los"
REQUIRE_GROUND_STATION_LOS_FLAG=""
SAVE_VISUALIZATIONS_FLAG=""
VIZ_SAVE_GIF_FLAG=""

if [[ "${RANDOMIZE_INSPECTION_TARGET:-0}" == "1" ]]; then
  RANDOMIZE_INSPECTION_TARGET_FLAG="--randomize-inspection-target"
fi
if [[ "${RANDOMIZE_GROUND_STATION:-0}" == "1" ]]; then
  RANDOMIZE_GROUND_STATION_FLAG="--randomize-ground-station"
fi
if [[ "${REQUIRE_TARGET_LOS:-1}" != "1" ]]; then
  REQUIRE_TARGET_LOS_FLAG="--no-require-target-los"
fi
if [[ "${REQUIRE_GROUND_STATION_LOS:-0}" == "1" ]]; then
  REQUIRE_GROUND_STATION_LOS_FLAG="--require-ground-station-los"
fi
if [[ "${SAVE_VISUALIZATIONS:-0}" == "1" ]]; then
  SAVE_VISUALIZATIONS_FLAG="--save-visualizations"
fi
if [[ "${VIZ_SAVE_GIF:-0}" == "1" ]]; then
  VIZ_SAVE_GIF_FLAG="--viz-save-gif"
fi

train_qrl() {
  local out_dir="$1"
  local cost_source="$2"

  if [[ "${RUN_TRAIN:-auto}" == "0" ]]; then
    echo "[task_ablation] Skipping training for $cost_source; expecting $out_dir/checkpoint_final.pth"
    return
  fi
  if [[ "${RUN_TRAIN:-auto}" == "auto" && -f "$out_dir/checkpoint_final.pth" ]]; then
    echo "[task_ablation] Reusing checkpoint: $out_dir/checkpoint_final.pth"
    return
  fi

  echo "[task_ablation] Training QRL cost_source=$cost_source -> $out_dir"
  "$PYTHON_BIN" minimal_qrl/train.py \
    --device "${DEVICE:-auto}" \
    --env-type comm_inspection_dubins_uav \
    --output-dir "$out_dir" \
    --bounds ${BOUNDS} \
    --omega-max "${OMEGA_MAX:-3.0}" \
    --v "${V_FORWARD:-1.0}" \
    --dt "${DT:-0.1}" \
    --observation-mode "${OBSERVATION_MODE:-task_context}" \
    --obstacle-config "$OBSTACLE_CONFIG" \
    --num-episodes "${NUM_EPISODES:-180}" \
    --max-steps-per-episode "${MAX_STEPS_PER_EPISODE:-180}" \
    --batch-size "${BATCH_SIZE:-256}" \
    --total-steps "${TOTAL_STEPS:-20000}" \
    --num-critics "${NUM_CRITICS:-2}" \
    --qrl-cost-source "$cost_source" \
    --log-interval "${LOG_INTERVAL:-100}" \
    --save-interval "${SAVE_INTERVAL:-2000}" \
    --eval-interval "${EVAL_INTERVAL:-1000}" \
    --eval-n-pairs "${EVAL_N_PAIRS:-400}" \
    --visualization-interval "${VIS_INTERVAL:-0}" \
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
}

eval_qrl() {
  local label="$1"
  local ckpt="$2"
  local out_dir="$3"

  if [[ ! -f "$ckpt" ]]; then
    echo "[task_ablation] Missing checkpoint for $label: $ckpt" >&2
    exit 1
  fi

  echo "[task_ablation] Evaluating $label: $ckpt"
  "$PYTHON_BIN" minimal_qrl/eval/comm_inspection_execution_eval.py \
    --checkpoint "$ckpt" \
    --output-dir "$out_dir" \
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
    --goal-position-tolerance "${GOAL_POS_TOL:-0.25}" \
    --goal-heading-tolerance "${GOAL_HEADING_TOL:-0.3}" \
    --collision-cost "${COLLISION_COST:-10.0}" \
    --out-of-bounds-cost "${OUT_OF_BOUNDS_COST:-10.0}" \
    --communication-break-cost "${COMM_BREAK_COST:-1.0}" \
    --observation-violation-cost-weight "${OBS_VIOLATION_COST_WEIGHT:-1.0}" \
    --communication-violation-cost-weight "${COMM_VIOLATION_COST_WEIGHT:-0.5}" \
    --observation-failure-cost "${OBSERVATION_FAILURE_COST:-0.25}" \
    --taskscore-beta-obs "${TASKSCORE_BETA_OBS:-1.0}" \
    --taskscore-beta-comm "${TASKSCORE_BETA_COMM:-1.0}" \
    --taskscore-beta-feas "${TASKSCORE_BETA_FEAS:-0.5}" \
    --taskscore-margin-clip "${TASKSCORE_MARGIN_CLIP:-2.0}" \
    --num-critics "${NUM_CRITICS:-2}" \
    --n-trials "${N_TRIALS:-100}" \
    --seed "${SEED:-0}" \
    --device "${DEVICE:-auto}" \
    --execution-modes "${EXECUTION_MODES:-greedy,lookahead}" \
    --lookahead-heuristics terminal \
    --lookahead-horizon "${LOOKAHEAD_HORIZON:-10}" \
    --lookahead-num-sequences "${LOOKAHEAD_NUM_SEQUENCES:-128}" \
    --lookahead-biased-sequences "${LOOKAHEAD_BIASED_SEQUENCES:-24}" \
    --lookahead-bias-kp "${LOOKAHEAD_BIAS_KP:-2.0}" \
    --lookahead-step-cost-weight "${LOOKAHEAD_STEP_COST_WEIGHT:-0.0}" \
    --lookahead-collision-penalty "${LOOKAHEAD_COLLISION_PENALTY:-0.0}" \
    --planner-alpha-final "${PLANNER_ALPHA_FINAL:-0.3}" \
    --planner-alpha-task-terminal "${PLANNER_ALPHA_TASK_TERMINAL:-0.5}" \
    ${SAVE_VISUALIZATIONS_FLAG} \
    --viz-max-successes "${VIZ_MAX_SUCCESSES:-10}" \
    --viz-max-failures "${VIZ_MAX_FAILURES:-10}" \
    ${VIZ_SAVE_GIF_FLAG} \
    --viz-gif-fps "${VIZ_GIF_FPS:-8}"
}

train_qrl "$ORIGINAL_DIR" fixed
train_qrl "$TASK_AWARE_DIR" negative_reward

eval_qrl "qrl_fixed_cost" "$ORIGINAL_DIR/checkpoint_final.pth" "$OUTPUT_DIR/eval_fixed_cost"
eval_qrl "qrl_task_aware" "$TASK_AWARE_DIR/checkpoint_final.pth" "$OUTPUT_DIR/eval_task_aware"

"$PYTHON_BIN" minimal_qrl/summarize_comm_inspection_results.py \
  --input "qrl_fixed_cost=$OUTPUT_DIR/eval_fixed_cost/comm_inspection_execution_eval.json" \
  --input "qrl_task_aware=$OUTPUT_DIR/eval_task_aware/comm_inspection_execution_eval.json" \
  --output-csv "$OUTPUT_DIR/task_aware_ablation_summary.csv"

echo "[task_ablation] Results:"
echo "  $OUTPUT_DIR/eval_fixed_cost/comm_inspection_execution_eval.json"
echo "  $OUTPUT_DIR/eval_task_aware/comm_inspection_execution_eval.json"
echo "  $OUTPUT_DIR/task_aware_ablation_summary.csv"
