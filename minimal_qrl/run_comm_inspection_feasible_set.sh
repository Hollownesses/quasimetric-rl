#!/usr/bin/env bash
# Experiment 1: task-feasible-set visualization and Monte Carlo quantification.
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

OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/results/experiments/comm_inspection_feasible_set}"
DEVICE_CATALOG="${DEVICE_CATALOG:-$ROOT_DIR/minimal_qrl/configs/chemical_process_plant_devices.json}"
BOUNDS_RAW="${BOUNDS:-0 0 10 10}"
read -r -a BOUNDS_ARGS <<< "$BOUNDS_RAW"

EXTRA_ARGS=()
if [[ -n "${DEVICE_IDS:-}" ]]; then
  EXTRA_ARGS+=(--device-ids "$DEVICE_IDS")
fi
if [[ "${REQUIRE_GROUND_STATION_LOS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--require-ground-station-los)
fi

"$PYTHON_BIN" minimal_qrl/industry_exp/comm_inspection_feasible_set.py \
  --output-dir "$OUTPUT_DIR" \
  --device-catalog "$DEVICE_CATALOG" \
  --bounds "${BOUNDS_ARGS[@]}" \
  --obstacle-config "${OBSTACLE_CONFIG:-medium}" \
  --num-samples "${NUM_SAMPLES:-50000}" \
  --grid-resolution "${GRID_RESOLUTION:-140}" \
  "--thresholds=${COMM_THRESHOLDS:--0.5,0.0,0.5,1.0,1.5}" \
  --comm-threshold "${COMM_THRESHOLD:-0.5}" \
  --comm-alpha "${COMM_ALPHA:-2.0}" \
  --comm-bias "${COMM_BIAS:-5.0}" \
  --comm-occlusion-penalty "${COMM_OCCLUSION_PENALTY:-6.0}" \
  --seed "${SEED:-20260712}" \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo "Experiment 1 results: $OUTPUT_DIR"
