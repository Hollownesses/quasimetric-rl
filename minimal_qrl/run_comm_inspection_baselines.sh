#!/usr/bin/env bash
# Unified Hybrid A*, no-terminal/model/QRL MPPI, goal-set SAC, and QRL greedy benchmark.
#
# STAGE: smoke, pilot, final
# STAGE=smoke：SAC 训练 200 环境步,评估 3 个 episode
# STAGE=pilot：训练一个 SAC seed，共 300000 步，并默认评估 50 个共同任务
# STAGE=final：训练 5 个 SAC seeds；可用 N_TRIALS 覆盖评估任务数

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -x "./.venv/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-./.venv/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

STAGE="${STAGE:-pilot}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/experiments/comm_inspection_baselines_${STAGE}}"
QRL_CHECKPOINTS="${QRL_CHECKPOINTS:-./results/goalset_qrl_comm_inspection/checkpoint_final.pth}"
METHODS="${METHODS:-hybrid_astar,mppi_no_terminal,model_mppi,goal_set_sac,qrl_greedy,qrl_mppi}"
N_TRIALS="${N_TRIALS:-}"

TRAIN_SAC_FLAG=""
SAVE_VISUALIZATIONS_FLAG=""
RANDOMIZE_TARGET_FLAG=""
RANDOMIZE_STATION_FLAG=""
EXTRA_ARGS=()
if [[ "${TRAIN_SAC:-1}" == "1" ]]; then
  TRAIN_SAC_FLAG="--train-sac"
fi
if [[ "${SAVE_VISUALIZATIONS:-0}" == "1" ]]; then
  SAVE_VISUALIZATIONS_FLAG="--save-visualizations"
fi
if [[ "${RANDOMIZE_INSPECTION_TARGET:-1}" == "1" ]]; then
  RANDOMIZE_TARGET_FLAG="--randomize-inspection-target"
fi
if [[ "${RANDOMIZE_GROUND_STATION:-1}" == "1" ]]; then
  RANDOMIZE_STATION_FLAG="--randomize-ground-station"
fi
if [[ -n "$N_TRIALS" ]]; then
  EXTRA_ARGS+=(--n-trials "$N_TRIALS")
fi
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
  --inspection-target ${INSPECTION_TARGET:-3.0 7.5} \
  --ground-station ${GROUND_STATION:-1.5 2.0} \
  ${RANDOMIZE_TARGET_FLAG} \
  ${RANDOMIZE_STATION_FLAG} \
  --observation-radius "${OBS_RADIUS:-1.8}" \
  --fov-angle "${FOV_ANGLE:-1.5707963267948966}" \
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
