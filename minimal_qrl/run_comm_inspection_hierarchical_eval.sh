#!/usr/bin/env bash
# 通信感知巡检 Dubins UAV 上的分层 QRL 评估脚本
#
# 目标：
# - 读取分层训练输出的 critic checkpoint 与 subgoal actor checkpoint
# - 对比 greedy / lookahead / hierarchical 三种执行方式
#
# 直接运行：
#   bash minimal_qrl/run_comm_inspection_hierarchical_eval.sh
#
# 常见覆盖方式：
#   OUTPUT_DIR=./results/minimal_qrl_inspection_dubins_hier \
#   N_TRIALS=300 \
#   SAVE_VISUALIZATIONS=1 \
#   bash minimal_qrl/run_comm_inspection_hierarchical_eval.sh

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -x "./.venv/bin/python" ]]; then
  PYTHON_BIN="./.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

OUTPUT_DIR="${OUTPUT_DIR:-./results/minimal_qrl_inspection_dubins_hier}"
CHECKPOINT="${CHECKPOINT:-$OUTPUT_DIR/checkpoint_final.pth}"
SUBGOAL_ACTOR_CHECKPOINT="${SUBGOAL_ACTOR_CHECKPOINT:-$OUTPUT_DIR/subgoal_actor_checkpoint_final.pth}"

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
NO_PLANNER_USE_ENV_STAGE_COST_FLAG=""

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
if [[ "${SAVE_VISUALIZATIONS:-0}" == "1" ]]; then
  SAVE_VISUALIZATIONS_FLAG="--save-visualizations"
fi
if [[ "${VIZ_SAVE_GIF:-0}" == "1" ]]; then
  VIZ_SAVE_GIF_FLAG="--viz-save-gif"
fi
if [[ "${PLANNER_USE_ENV_STAGE_COST:-1}" != "1" ]]; then
  NO_PLANNER_USE_ENV_STAGE_COST_FLAG="--no-planner-use-env-stage-cost"
fi

echo "评估通信巡检 Dubins UAV 分层 QRL..."

"$PYTHON_BIN" minimal_qrl/eval/comm_inspection_execution_eval.py \
  --checkpoint "$CHECKPOINT" \
  --subgoal-actor-checkpoint "$SUBGOAL_ACTOR_CHECKPOINT" \
  --output-dir "$OUTPUT_DIR/hier_eval_results" \
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
  --execution-modes "${EXECUTION_MODES:-greedy,lookahead,hierarchical}" \
  --high-level-period "${HIGH_LEVEL_PERIOD:-5}" \
  --subgoal-candidates "${SUBGOAL_CANDIDATES:-64}" \
  --subgoal-lambda-final "${SUBGOAL_LAMBDA_FINAL:-0.3}" \
  --subgoal-lambda-task "${SUBGOAL_LAMBDA_TASK:-1.0}" \
  --lookahead-horizon "${LOOKAHEAD_HORIZON:-10}" \
  --lookahead-num-sequences "${LOOKAHEAD_NUM_SEQUENCES:-128}" \
  --lookahead-step-cost-weight "${LOOKAHEAD_STEP_COST_WEIGHT:-0.0}" \
  --lookahead-collision-penalty "${LOOKAHEAD_COLLISION_PENALTY:-0.0}" \
  --lookahead-biased-sequences "${LOOKAHEAD_BIASED_SEQUENCES:-24}" \
  --lookahead-bias-kp "${LOOKAHEAD_BIAS_KP:-2.0}" \
  --planner-alpha-subgoal "${PLANNER_ALPHA_SUBGOAL:-1.0}" \
  --planner-alpha-final "${PLANNER_ALPHA_FINAL:-0.3}" \
  --planner-alpha-task-terminal "${PLANNER_ALPHA_TASK_TERMINAL:-0.5}" \
  ${NO_PLANNER_USE_ENV_STAGE_COST_FLAG} \
  ${SAVE_VISUALIZATIONS_FLAG} \
  --viz-max-successes "${VIZ_MAX_SUCCESSES:-10}" \
  --viz-max-failures "${VIZ_MAX_FAILURES:-10}" \
  ${VIZ_SAVE_GIF_FLAG} \
  --viz-gif-fps "${VIZ_GIF_FPS:-8}"

echo
echo "分层评估完成。结果保存在：$OUTPUT_DIR/hier_eval_results/comm_inspection_execution_eval.json"
