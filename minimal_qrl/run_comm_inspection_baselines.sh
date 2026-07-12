#!/usr/bin/env bash
# Unified Hybrid A*, no-terminal/model/QRL MPPI, goal-set SAC, and QRL greedy benchmark.
#
# STAGE: smoke, pilot, final
# STAGE=smoke：SAC 训练 200 环境步,评估 3 个 episode
# STAGE=pilot：训练一个 SAC seed，共 300000 步
# STAGE=final：训练 5 个 SAC seeds
# STARTS_PER_DEVICE：每台设备的随机起点数，默认 25，可通过环境变量覆盖

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -x "./.venv/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-./.venv/bin/python}"
elif [[ -x "../quasimetric-rl/.venv/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-../quasimetric-rl/.venv/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

STAGE="${STAGE:-pilot}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/experiments/comm_inspection_baselines_${STAGE}}"
QRL_CHECKPOINTS="${QRL_CHECKPOINTS:-./results/goalset_qrl_comm_inspection/checkpoint_final.pth}"
METHODS="${METHODS:-hybrid_astar,mppi_no_terminal,model_mppi,goal_set_sac,qrl_greedy,qrl_mppi}"
STARTS_PER_DEVICE="${STARTS_PER_DEVICE:-25}"

TRAIN_SAC_FLAG=""
SAVE_VISUALIZATIONS_FLAG=""
EXTRA_ARGS=()
if [[ "${TRAIN_SAC:-1}" == "1" ]]; then
  TRAIN_SAC_FLAG="--train-sac"
fi
if [[ "${SAVE_VISUALIZATIONS:-0}" == "1" ]]; then
  SAVE_VISUALIZATIONS_FLAG="--save-visualizations"
fi
EXTRA_ARGS+=(--starts-per-device "$STARTS_PER_DEVICE")
if [[ -n "${SAC_TOTAL_ENV_STEPS:-}" ]]; then
  EXTRA_ARGS+=(--sac-total-env-steps "$SAC_TOTAL_ENV_STEPS")
fi
if [[ -n "${SAC_SEEDS:-}" ]]; then
  EXTRA_ARGS+=(--sac-seeds "$SAC_SEEDS")
fi
if [[ -n "${SAC_BATCH_SIZE:-}" ]]; then
  EXTRA_ARGS+=(--sac-batch-size "$SAC_BATCH_SIZE")
fi

read -r -a QRL_CKPT_ARRAY <<< "$QRL_CHECKPOINTS"

SAC_CKPT_ARRAY=()
if [[ "${TRAIN_SAC:-1}" != "1" ]]; then
  if [[ -n "${SAC_CHECKPOINTS:-}" ]]; then
    read -r -a SAC_CKPT_ARRAY <<< "$SAC_CHECKPOINTS"
  else
    shopt -s nullglob
    for ckpt in "$OUTPUT_DIR"/goal_set_sac/seed_*/checkpoint_final.pth; do
      SAC_CKPT_ARRAY+=("$ckpt")
    done
    shopt -u nullglob
  fi
  if (( ${#SAC_CKPT_ARRAY[@]} > 0 )); then
    EXTRA_ARGS+=(--sac-checkpoints "${SAC_CKPT_ARRAY[@]}")
  fi
fi

echo "通信巡检 baseline 评估配置："
echo "  stage=$STAGE"
echo "  methods=$METHODS"
echo "  starts_per_device=$STARTS_PER_DEVICE"
echo "  device_catalog=${DEVICE_CATALOG:-./minimal_qrl/configs/industrial_site_devices.json}"
echo "  output_dir=$OUTPUT_DIR"
echo "  incremental_csv=$OUTPUT_DIR/baseline_results.partial.csv"
echo "  incremental_jsonl=$OUTPUT_DIR/baseline_results.partial.jsonl"

"$PYTHON_BIN" minimal_qrl/eval/comm_inspection_baseline_eval.py \
  --stage "$STAGE" \
  --methods "$METHODS" \
  --output-dir "$OUTPUT_DIR" \
  --qrl-checkpoints "${QRL_CKPT_ARRAY[@]}" \
  ${TRAIN_SAC_FLAG} \
  --seed "${SEED:-0}" \
  --device "${DEVICE:-auto}" \
  --bounds ${BOUNDS:-0 0 10 10} \
  --omega-max "${OMEGA_MAX:-3.0}" \
  --v "${V_FORWARD:-1.0}" \
  --dt "${DT:-0.1}" \
  --max-episode-steps "${MAX_STEPS_PER_EPISODE:-180}" \
  --obstacle-config "${OBSTACLE_CONFIG:-medium}" \
  --device-catalog "${DEVICE_CATALOG:-./minimal_qrl/configs/industrial_site_devices.json}" \
  --comm-threshold "${COMM_THRESHOLD:-0.5}" \
  --mppi-horizon "${MPPI_HORIZON:-10}" \
  --mppi-num-samples "${MPPI_NUM_SAMPLES:-128}" \
  --astar-timeout-sec "${ASTAR_TIMEOUT_SEC:-30}" \
  --astar-terminal-samples "${ASTAR_TERMINAL_SAMPLES:-128}" \
  ${SAVE_VISUALIZATIONS_FLAG} \
  --viz-max-successes 10 \
  --viz-max-failures 10 \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo "Baseline results: $OUTPUT_DIR/baseline_results.json"
