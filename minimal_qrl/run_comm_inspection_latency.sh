#!/usr/bin/env bash
# Experiment 3: repeated QRL queries and end-to-end first-decision latency.
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

OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/results/experiments/comm_inspection_latency}"
DEVICE_CATALOG="${DEVICE_CATALOG:-$ROOT_DIR/minimal_qrl/configs/chemical_process_plant_devices.json}"
CHECKPOINT="${CHECKPOINT:-$ROOT_DIR/results/industrial_inspection_env_che/checkpoint_final.pth}"
BOUNDS_RAW="${BOUNDS:-0 0 10 10}"
read -r -a BOUNDS_ARGS <<< "$BOUNDS_RAW"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Missing QRL checkpoint: $CHECKPOINT" >&2
  echo "Set CHECKPOINT=/absolute/path/checkpoint_final.pth and rerun." >&2
  exit 2
fi

EXTRA_ARGS=()
if [[ "${REQUIRE_GROUND_STATION_LOS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--require-ground-station-los)
fi

"$PYTHON_BIN" minimal_qrl/industry_exp/comm_inspection_latency.py \
  --output-dir "$OUTPUT_DIR" \
  --checkpoint "$CHECKPOINT" \
  --device "${DEVICE:-auto}" \
  --device-catalog "$DEVICE_CATALOG" \
  --bounds "${BOUNDS_ARGS[@]}" \
  --obstacle-config "${OBSTACLE_CONFIG:-medium}" \
  --max-episode-steps "${MAX_EPISODE_STEPS:-180}" \
  --comm-threshold "${COMM_THRESHOLD:-0.5}" \
  --batch-sizes "${BATCH_SIZES:-1,8,24,128,600}" \
  --query-repeats "${QUERY_REPEATS:-1000}" \
  --warmup "${WARMUP:-20}" \
  --controller-methods "${CONTROLLER_METHODS:-qrl_mppi,mppi_no_terminal,hybrid_astar}" \
  --controller-trials "${CONTROLLER_TRIALS:-10}" \
  --mppi-horizon "${MPPI_HORIZON:-10}" \
  --mppi-num-samples "${MPPI_NUM_SAMPLES:-128}" \
  --astar-timeout-sec "${ASTAR_TIMEOUT_SEC:-5}" \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES:-2000}" \
  --seed "${SEED:-20260712}" \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo "Experiment 3 results: $OUTPUT_DIR"
