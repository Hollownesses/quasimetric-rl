#!/usr/bin/env bash
# 通信感知巡检 Dubins UAV 上的分层 QRL 训练脚本
#
# 功能：
# 1. 先训练 QRL critic
# 2. 冻结 critic，再训练高层 CostAwareSubgoalPolicy (goal-conditioned SAC)
#
# 直接运行：
#   bash minimal_qrl/run_comm_inspection_hierarchical_train.sh
#
# 常见覆盖方式：
#   OUTPUT_DIR=./results/minimal_qrl_inspection_dubins_hier \
#   TOTAL_STEPS=30000 \
#   HIGH_LEVEL_TRAIN_STEPS=8000 \
#   bash minimal_qrl/run_comm_inspection_hierarchical_train.sh

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -x "./.venv/bin/python" ]]; then
  PYTHON_BIN="./.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

OUTPUT_DIR="${OUTPUT_DIR:-./results/minimal_qrl_inspection_dubins_hier}"
mkdir -p "$OUTPUT_DIR"
DEFAULT_CRITIC_CHECKPOINT="$OUTPUT_DIR/checkpoint_final.pth"

BOUNDS="${BOUNDS:-0 0 10 10}"
INSPECTION_TARGET="${INSPECTION_TARGET:-3.0 7.5}"
GROUND_STATION="${GROUND_STATION:-1.5 2.0}"
OBSTACLE_CONFIG="${OBSTACLE_CONFIG:-medium}"

RANDOMIZE_INSPECTION_TARGET_FLAG=""
RANDOMIZE_GROUND_STATION_FLAG=""
REQUIRE_TARGET_LOS_FLAG=""
REQUIRE_GROUND_STATION_LOS_FLAG=""
INIT_CHECKPOINT_FLAG=""
SKIP_CRITIC_TRAINING_FLAG=""

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
if [[ "${ONLY_HIGH_LEVEL_POLICY:-${ONLY_SUBGOAL_ACTOR:-0}}" == "1" ]]; then
  SKIP_CRITIC_TRAINING_FLAG="--skip-critic-training"
  CRITIC_CHECKPOINT="${CRITIC_CHECKPOINT:-$DEFAULT_CRITIC_CHECKPOINT}"
fi
if [[ -n "${CRITIC_CHECKPOINT:-}" ]]; then
  INIT_CHECKPOINT_FLAG="--init-checkpoint ${CRITIC_CHECKPOINT}"
fi

"$PYTHON_BIN" minimal_qrl/train.py \
  --device auto \
  --env-type comm_inspection_dubins_uav \
  --output-dir "$OUTPUT_DIR" \
  ${INIT_CHECKPOINT_FLAG} \
  ${SKIP_CRITIC_TRAINING_FLAG} \
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
  --goal-position-tolerance "${GOAL_POS_TOL:-0.15}" \
  --goal-heading-tolerance "${GOAL_HEADING_TOL:-0.2}" \
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
  --hierarchical-mode "${HIERARCHICAL_MODE:-sac_subgoal}" \
  --high-level-period "${HIGH_LEVEL_PERIOD:-5}" \
  --lookahead-horizon "${LOOKAHEAD_HORIZON:-10}" \
  --lookahead-num-sequences "${LOOKAHEAD_NUM_SEQUENCES:-128}" \
  --lookahead-step-cost-weight "${LOOKAHEAD_STEP_COST_WEIGHT:-0.0}" \
  --lookahead-collision-penalty "${LOOKAHEAD_COLLISION_PENALTY:-0.0}" \
  --lookahead-biased-sequences "${LOOKAHEAD_BIASED_SEQUENCES:-24}" \
  --lookahead-bias-kp "${LOOKAHEAD_BIAS_KP:-2.0}" \
  --planner-alpha-subgoal "${PLANNER_ALPHA_SUBGOAL:-1.0}" \
  --planner-alpha-final "${PLANNER_ALPHA_FINAL:-0.0}" \
  --planner-alpha-task-terminal "${PLANNER_ALPHA_TASK_TERMINAL:-0.0}" \
  --high-level-train-steps "${HIGH_LEVEL_TRAIN_STEPS:-5000}" \
  --high-level-batch-size "${HIGH_LEVEL_BATCH_SIZE:-128}" \
  --high-level-actor-lr "${HIGH_LEVEL_ACTOR_LR:-3e-4}" \
  --high-level-critic-lr "${HIGH_LEVEL_CRITIC_LR:-3e-4}" \
  --high-level-gamma "${HIGH_LEVEL_GAMMA:-0.99}" \
  --high-level-tau "${HIGH_LEVEL_TAU:-0.005}" \
  --high-level-init-alpha "${HIGH_LEVEL_INIT_ALPHA:-0.2}" \
  --high-level-replay-size "${HIGH_LEVEL_REPLAY_SIZE:-200000}" \
  --high-level-start-random-steps "${HIGH_LEVEL_START_RANDOM_STEPS:-1000}" \
  --high-level-updates-per-step "${HIGH_LEVEL_UPDATES_PER_STEP:-1}" \
  --high-level-hidden-dim "${HIGH_LEVEL_HIDDEN_DIM:-256}" \
  --high-level-save-interval "${HIGH_LEVEL_SAVE_INTERVAL:-1000}" \
  --subgoal-max-radius "${SUBGOAL_MAX_RADIUS:-1.5}" \
  --subgoal-relative-param "${SUBGOAL_RELATIVE_PARAM:-polar_local}"

echo "分层训练完成。结果目录: $OUTPUT_DIR"
echo "Critic checkpoint:"
echo "  $OUTPUT_DIR/checkpoint_final.pth"
echo "High-level policy checkpoint:"
echo "  $OUTPUT_DIR/high_level_policy_checkpoint_final.pth"
echo "TensorBoard:"
echo "  $PYTHON_BIN -m tensorboard.main --logdir=$OUTPUT_DIR/tensorboard"
