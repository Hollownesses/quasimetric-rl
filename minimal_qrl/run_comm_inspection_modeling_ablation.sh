#!/usr/bin/env bash
# Experiment 4: 2x2 goal representation x communication-planning ablation.
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

OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/results/experiments/comm_inspection_modeling_ablation}"
DEVICE_CATALOG="${DEVICE_CATALOG:-$ROOT_DIR/minimal_qrl/configs/chemical_process_plant_devices.json}"
BOUNDS_RAW="${BOUNDS:-0 0 10 10}"
read -r -a BOUNDS_ARGS <<< "$BOUNDS_RAW"

EXTRA_ARGS=()
if [[ "${REQUIRE_GROUND_STATION_LOS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--require-ground-station-los)
fi

"$PYTHON_BIN" minimal_qrl/industry_exp/comm_inspection_modeling_ablation.py \
  --output-dir "$OUTPUT_DIR" \
  --device-catalog "$DEVICE_CATALOG" \
  --bounds "${BOUNDS_ARGS[@]}" \
  --obstacle-config "${OBSTACLE_CONFIG:-medium}" \
  --max-episode-steps "${MAX_EPISODE_STEPS:-180}" \
  --comm-threshold "${COMM_THRESHOLD:-0.5}" \
  --starts-per-device "${STARTS_PER_DEVICE:-10}" \
  --mppi-horizon "${MPPI_HORIZON:-10}" \
  --mppi-num-samples "${MPPI_NUM_SAMPLES:-128}" \
  --mppi-terminal-samples "${MPPI_TERMINAL_SAMPLES:-128}" \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES:-2000}" \
  --seed "${SEED:-20260712}" \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo "Experiment 4 results: $OUTPUT_DIR"
