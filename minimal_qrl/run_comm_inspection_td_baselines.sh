#!/usr/bin/env bash
# Group 1: communication-inspection Dubins QRL vs TD-based goal-conditioned baselines.

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -x "./.venv/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-./.venv/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

OUTPUT_DIR="${OUTPUT_DIR:-./results/experiments/comm_inspection_td_baselines}"
QRL_DIR="${QRL_DIR:-./results/experiments/comm_inspection_td_baselines/qrl_original}"
QRL_CKPT="${QRL_CKPT:-$QRL_DIR/checkpoint_final.pth}"
mkdir -p "$OUTPUT_DIR"

BOUNDS="${BOUNDS:-0 0 10 10}"
INSPECTION_TARGET="${INSPECTION_TARGET:-3.0 7.5}"
GROUND_STATION="${GROUND_STATION:-1.5 2.0}"
OBSTACLE_CONFIG="${OBSTACLE_CONFIG:-medium}"

RANDOMIZE_INSPECTION_TARGET_FLAG=""
RANDOMIZE_GROUND_STATION_FLAG=""
REQUIRE_TARGET_LOS_FLAG="--require-target-los"
REQUIRE_GROUND_STATION_LOS_FLAG=""
NO_PLANNER_USE_ENV_STAGE_COST_FLAG=""
SAVE_VISUALIZATIONS_FLAG=""
SKIP_TD_TRAINING_FLAG=""

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
if [[ "${PLANNER_USE_ENV_STAGE_COST:-1}" != "1" ]]; then
  NO_PLANNER_USE_ENV_STAGE_COST_FLAG="--no-planner-use-env-stage-cost"
fi
if [[ "${SAVE_VISUALIZATIONS:-0}" == "1" ]]; then
  SAVE_VISUALIZATIONS_FLAG="--save-visualizations"
fi
if [[ "${SKIP_TD_TRAINING:-0}" == "1" ]]; then
  SKIP_TD_TRAINING_FLAG="--skip-td-training"
fi

if [[ "${RUN_QRL_TRAIN:-auto}" == "1" || ( "${RUN_QRL_TRAIN:-auto}" == "auto" && ! -f "$QRL_CKPT" ) ]]; then
  echo "[td_baselines] Training original QRL checkpoint: $QRL_DIR"
  "$PYTHON_BIN" minimal_qrl/train.py \
    --device "${DEVICE:-auto}" \
    --env-type comm_inspection_dubins_uav \
    --output-dir "$QRL_DIR" \
    --bounds ${BOUNDS} \
    --omega-max "${OMEGA_MAX:-3.0}" \
    --v "${V_FORWARD:-1.0}" \
    --dt "${DT:-0.1}" \
    --observation-mode "${OBSERVATION_MODE:-task_context}" \
    --obstacle-config "$OBSTACLE_CONFIG" \
    --num-episodes "${QRL_NUM_EPISODES:-180}" \
    --max-steps-per-episode "${MAX_STEPS_PER_EPISODE:-180}" \
    --batch-size "${QRL_BATCH_SIZE:-256}" \
    --total-steps "${QRL_TOTAL_STEPS:-20000}" \
    --num-critics "${NUM_CRITICS:-2}" \
    --qrl-cost-source "${QRL_COST_SOURCE:-fixed}" \
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
fi

echo "[td_baselines] Training/evaluating TD baselines and QRL..."
"$PYTHON_BIN" minimal_qrl/run_comm_inspection_td_baselines.py \
  --output-dir "$OUTPUT_DIR" \
  --qrl-ckpt "$QRL_CKPT" \
  --qrl-execution-modes "${QRL_EXECUTION_MODES:-${TD_EXECUTION_MODES:-greedy,lookahead}}" \
  --qrl-num-episodes "${QRL_NUM_EPISODES:-180}" \
  --qrl-total-steps "${QRL_TOTAL_STEPS:-20000}" \
  --qrl-batch-size "${QRL_BATCH_SIZE:-256}" \
  --td-algos "${TD_ALGOS:-gc_sac,her_ddpg,uvfa}" \
  --td-execution-modes "${TD_EXECUTION_MODES:-greedy,lookahead}" \
  ${SKIP_TD_TRAINING_FLAG} \
  --total-env-steps "${TD_TOTAL_ENV_STEPS:-200000}" \
  --batch-size "${TD_BATCH_SIZE:-256}" \
  --start-random-steps "${TD_START_RANDOM_STEPS:-1000}" \
  --her-k "${HER_K:-4}" \
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
  --num-critics "${NUM_CRITICS:-2}" \
  --n-trials "${N_TRIALS:-100}" \
  --seed "${SEED:-0}" \
  --device "${DEVICE:-auto}" \
  --lookahead-horizon "${LOOKAHEAD_HORIZON:-10}" \
  --lookahead-num-sequences "${LOOKAHEAD_NUM_SEQUENCES:-128}" \
  --qrl-lookahead-heuristics "${QRL_LOOKAHEAD_HEURISTICS:-terminal}" \
  --lookahead-biased-sequences "${LOOKAHEAD_BIASED_SEQUENCES:-24}" \
  --lookahead-bias-kp "${LOOKAHEAD_BIAS_KP:-2.0}" \
  --planner-alpha-final "${PLANNER_ALPHA_FINAL:-0.4}" \
  --planner-alpha-task-terminal "${PLANNER_ALPHA_TASK_TERMINAL:-0.5}" \
  --planner-qrl-progress-alpha "${PLANNER_QRL_PROGRESS_ALPHA:-0.0}" \
  --viz-max-successes "${VIZ_MAX_SUCCESSES:-3}" \
  --viz-max-failures "${VIZ_MAX_FAILURES:-3}" \
  ${NO_PLANNER_USE_ENV_STAGE_COST_FLAG} \
  ${SAVE_VISUALIZATIONS_FLAG}

echo "[td_baselines] Results:"
echo "  $OUTPUT_DIR/comm_inspection_td_baselines.json"
echo "  $OUTPUT_DIR/comm_inspection_td_baselines.csv"
