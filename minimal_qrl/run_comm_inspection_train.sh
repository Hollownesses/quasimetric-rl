#!/usr/bin/env bash
# 通信感知巡检 Dubins UAV 环境上的 QRL 训练脚本
#
# 直接运行：
#   bash minimal_qrl/run_comm_inspection_train.sh
#
# 常见覆盖方式：
#   OUTPUT_DIR=./results/minimal_qrl_inspection_dubins_exp1 \
#   DEVICE_CATALOG=./minimal_qrl/configs/industrial_site_devices.json \
#   OBSTACLE_CONFIG=medium \
#   bash minimal_qrl/run_comm_inspection_train.sh
#
# reset 时从设备目录均匀采样异常设备，私有5G基站始终固定。
#
# QRL local constraint 单步代价来源：
#   QRL_COST_SOURCE=fixed            # 默认：使用原始固定 step_cost=1.0
#   QRL_COST_SOURCE=negative_reward  # 使用环境 task cost

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -x "./.venv/bin/python" ]]; then
  PYTHON_BIN="./.venv/bin/python"
elif [[ -x "../quasimetric-rl/.venv/bin/python" ]]; then
  PYTHON_BIN="../quasimetric-rl/.venv/bin/python"
else
  echo "未找到可用的 .venv Python：请先创建 ./.venv 或 ../quasimetric-rl/.venv" >&2
  exit 1
fi

OUTPUT_DIR="${OUTPUT_DIR:-./results/industrial_inspection_env}"
mkdir -p "$OUTPUT_DIR"

BOUNDS="${BOUNDS:-0 0 10 10}"
DEVICE_CATALOG="${DEVICE_CATALOG:-./minimal_qrl/configs/industrial_site_devices.json}"
OBSTACLE_CONFIG="${OBSTACLE_CONFIG:-medium}"

REQUIRE_GROUND_STATION_LOS_FLAG=""

if [[ "${REQUIRE_GROUND_STATION_LOS:-0}" == "1" ]]; then
  REQUIRE_GROUND_STATION_LOS_FLAG="--require-ground-station-los"
fi

"$PYTHON_BIN" minimal_qrl/train.py \
  --device "${DEVICE:-cpu}" \
  --env-type comm_inspection_dubins_uav \
  --output-dir "$OUTPUT_DIR" \
  --bounds ${BOUNDS} \
  --omega-max "${OMEGA_MAX:-3.0}" \
  --v "${V_FORWARD:-1.0}" \
  --dt "${DT:-0.1}" \
  --device-catalog "$DEVICE_CATALOG" \
  --obstacle-config "$OBSTACLE_CONFIG" \
  --num-episodes "${NUM_EPISODES:-500}" \
  --max-steps-per-episode "${MAX_STEPS_PER_EPISODE:-180}" \
  --batch-size "${BATCH_SIZE:-256}" \
  --total-steps "${TOTAL_STEPS:-30000}" \
  --num-critics "${NUM_CRITICS:-2}" \
  --qrl-cost-source "${QRL_COST_SOURCE:-negative_reward}" \
  --global-push-abstract-goal-ratio "${GLOBAL_PUSH_ABSTRACT_GOAL_RATIO:-0.6}" \
  --global-push-state-goal-ratio "${GLOBAL_PUSH_STATE_GOAL_RATIO:-0.4}" \
  --abstract-goal-edge-loss-weight "${ABSTRACT_GOAL_EDGE_LOSS_WEIGHT:-1.0}" \
  --task-aware-teacher-ratio "${TASK_AWARE_TEACHER_RATIO:-1.0}" \
  --log-interval "${LOG_INTERVAL:-100}" \
  --save-interval "${SAVE_INTERVAL:-2000}" \
  --eval-interval "${EVAL_INTERVAL:-1000}" \
  --oracle-bank-eval \
  --oracle-bank-dir "${ORACLE_BANK_DIR:-$OUTPUT_DIR/oracle_banks}" \
  --oracle-bank-size "${ORACLE_BANK_SIZE:-192}" \
  --oracle-bank-seed "${ORACLE_BANK_SEED:-20260729}" \
  --oracle-astar-timeout-sec "${ORACLE_ASTAR_TIMEOUT_SEC:-60}" \
  --oracle-final-bootstrap-samples "${ORACLE_FINAL_BOOTSTRAP_SAMPLES:-2000}" \
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
  --observation-failure-cost "${OBSERVATION_FAILURE_COST:-0.25}"

echo "训练完成。结果目录: $OUTPUT_DIR"
echo "TensorBoard:"
echo "  $PYTHON_BIN -m tensorboard.main --logdir=$OUTPUT_DIR/tensorboard"
echo "检查点:"
echo "  $OUTPUT_DIR/checkpoint_final.pth"
