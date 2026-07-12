#!/usr/bin/env bash
# 通信感知巡检 Dubins UAV 上的分层 QRL 训练脚本
#
# 功能：
# 1. 先训练 QRL critic
# 2. 冻结 critic，再训练 SubgoalActor
#
# 直接运行：
#   bash minimal_qrl/run_comm_inspection_hierarchical_train.sh
#
# 常见覆盖方式：
#   OUTPUT_DIR=./results/minimal_qrl_inspection_dubins_hier \
#   TOTAL_STEPS=30000 \
#   SUBGOAL_TRAIN_STEPS=8000 \
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
DEVICE_CATALOG="${DEVICE_CATALOG:-./minimal_qrl/configs/industrial_site_devices.json}"
OBSTACLE_CONFIG="${OBSTACLE_CONFIG:-medium}"

REQUIRE_GROUND_STATION_LOS_FLAG=""
INIT_CHECKPOINT_FLAG=""
SKIP_CRITIC_TRAINING_FLAG=""

if [[ "${REQUIRE_GROUND_STATION_LOS:-0}" == "1" ]]; then
  REQUIRE_GROUND_STATION_LOS_FLAG="--require-ground-station-los"
fi
if [[ "${ONLY_SUBGOAL_ACTOR:-0}" == "1" ]]; then
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
  --device-catalog "$DEVICE_CATALOG" \
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
  --taskscore-beta-obs "${TASKSCORE_BETA_OBS:-1.0}" \
  --taskscore-beta-comm "${TASKSCORE_BETA_COMM:-1.0}" \
  --taskscore-beta-feas "${TASKSCORE_BETA_FEAS:-0.5}" \
  --taskscore-margin-clip "${TASKSCORE_MARGIN_CLIP:-2.0}" \
  --hierarchical-mode subgoal_actor \
  --subgoal-train-steps "${SUBGOAL_TRAIN_STEPS:-5000}" \
  --subgoal-batch-size "${SUBGOAL_BATCH_SIZE:-32}" \
  --subgoal-actor-lr "${SUBGOAL_ACTOR_LR:-3e-4}" \
  --subgoal-actor-hidden-dim "${SUBGOAL_ACTOR_HIDDEN_DIM:-256}" \
  --subgoal-save-interval "${SUBGOAL_SAVE_INTERVAL:-1000}" \
  --subgoal-candidates "${SUBGOAL_CANDIDATES:-64}" \
  --high-level-period "${HIGH_LEVEL_PERIOD:-5}" \
  --subgoal-lambda-final "${SUBGOAL_LAMBDA_FINAL:-0.3}" \
  --subgoal-lambda-task "${SUBGOAL_LAMBDA_TASK:-1.0}"

echo "分层训练完成。结果目录: $OUTPUT_DIR"
echo "Critic checkpoint:"
echo "  $OUTPUT_DIR/checkpoint_final.pth"
echo "Subgoal actor checkpoint:"
echo "  $OUTPUT_DIR/subgoal_actor_checkpoint_final.pth"
echo "TensorBoard:"
echo "  $PYTHON_BIN -m tensorboard.main --logdir=$OUTPUT_DIR/tensorboard"
