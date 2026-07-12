#!/usr/bin/env bash
# Experiment 2: one fixed-site QRL model reused across paired starts and device targets.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
elif [[ -x "$ROOT_DIR/../quasimetric-rl/.venv/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/../quasimetric-rl/.venv/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/results/experiments/comm_inspection_multitask_reuse}"
DEVICE_CATALOG="${DEVICE_CATALOG:-$ROOT_DIR/minimal_qrl/configs/chemical_process_plant_devices.json}"
DEFAULT_CHECKPOINT="$ROOT_DIR/results/industrial_inspection_env_che/checkpoint_final.pth"
QRL_CHECKPOINTS="${QRL_CHECKPOINTS:-$DEFAULT_CHECKPOINT}"
METHODS="${METHODS:-qrl_mppi,mppi_no_terminal,hybrid_astar}"
BOUNDS_RAW="${BOUNDS:-0 0 10 10}"
read -r -a BOUNDS_ARGS <<< "$BOUNDS_RAW"
read -r -a QRL_CKPT_ARGS <<< "$QRL_CHECKPOINTS"

if [[ "$METHODS" == *qrl* ]]; then
  for checkpoint in "${QRL_CKPT_ARGS[@]}"; do
    if [[ ! -f "$checkpoint" ]]; then
      echo "Missing QRL checkpoint: $checkpoint" >&2
      echo "Set QRL_CHECKPOINTS='/absolute/path/checkpoint_final.pth' and rerun." >&2
      exit 2
    fi
  done
fi

EXTRA_ARGS=()
if [[ "${REQUIRE_GROUND_STATION_LOS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--require-ground-station-los)
fi
if [[ "${SAVE_VISUALIZATIONS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--save-visualizations)
fi

"$PYTHON_BIN" minimal_qrl/eval/comm_inspection_baseline_eval.py \
  --stage pilot \
  --methods "$METHODS" \
  --output-dir "$OUTPUT_DIR" \
  --qrl-checkpoints "${QRL_CKPT_ARGS[@]}" \
  --starts-per-device "${STARTS_PER_DEVICE:-25}" \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES:-2000}" \
  --device "${DEVICE:-auto}" \
  --device-catalog "$DEVICE_CATALOG" \
  --bounds "${BOUNDS_ARGS[@]}" \
  --obstacle-config "${OBSTACLE_CONFIG:-medium}" \
  --omega-max "${OMEGA_MAX:-3.0}" \
  --v "${V_FORWARD:-1.0}" \
  --dt "${DT:-0.1}" \
  --max-episode-steps "${MAX_EPISODE_STEPS:-180}" \
  --comm-threshold "${COMM_THRESHOLD:-0.5}" \
  --mppi-horizon "${MPPI_HORIZON:-10}" \
  --mppi-num-samples "${MPPI_NUM_SAMPLES:-128}" \
  --astar-timeout-sec "${ASTAR_TIMEOUT_SEC:-30}" \
  --astar-terminal-samples "${ASTAR_TERMINAL_SAMPLES:-128}" \
  --seed "${SEED:-20260712}" \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

"$PYTHON_BIN" minimal_qrl/industry_exp/comm_inspection_multitask_report.py \
  --input-json "$OUTPUT_DIR/baseline_results.json" \
  --output-dir "$OUTPUT_DIR/report"

echo "Experiment 2 results: $OUTPUT_DIR"
