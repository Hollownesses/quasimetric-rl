#!/usr/bin/env bash
# Evaluate a QRL final checkpoint with MPPI terminal_weight=0.1 and temperature=1.0.

set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -n "${PYTHON_BIN:-}" ]]; then
  QRL_EVAL_PYTHON="$PYTHON_BIN"
elif [[ -x "./.venv/bin/python" ]]; then
  QRL_EVAL_PYTHON="./.venv/bin/python"
elif [[ -x "../quasimetric-rl/.venv/bin/python" ]]; then
  QRL_EVAL_PYTHON="../quasimetric-rl/.venv/bin/python"
else
  QRL_EVAL_PYTHON="python3"
fi

GP="${GP:-300}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-./results/industrial_inspection_gp${GP}}"
QRL_CHECKPOINT="${QRL_CHECKPOINT:-${EXPERIMENT_DIR}/checkpoint_final.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-${EXPERIMENT_DIR}/eval_res_qrl_final_tw01_t1}"
DEVICE_CATALOG="${DEVICE_CATALOG:-./minimal_qrl/configs/chemical_process_plant_devices.json}"

if [[ ! -f "$QRL_CHECKPOINT" ]]; then
  echo "Checkpoint not found: $QRL_CHECKPOINT" >&2
  exit 1
fi

if [[ ! -f "$DEVICE_CATALOG" ]]; then
  echo "Device catalog not found: $DEVICE_CATALOG" >&2
  exit 1
fi

RESUME_ARGS=()
if [[ "${RESUME:-1}" == "1" ]]; then
  RESUME_ARGS+=(--resume)
fi

echo "QRL final checkpoint evaluation (TW=0.1, temperature=1.0)"
echo "  gp=$GP"
echo "  experiment_dir=$EXPERIMENT_DIR"
echo "  checkpoint=$QRL_CHECKPOINT"
echo "  output_dir=$OUTPUT_DIR"
echo "  methods=qrl_mppi,qrl_greedy"
echo "  starts_per_device=${STARTS_PER_DEVICE:-30}"
echo "  seed=${SEED:-20260716}"
echo "  terminal_weight=0.1"
echo "  temperature=1.0"
echo "  resume=${RESUME:-1}"

"$QRL_EVAL_PYTHON" minimal_qrl/eval/comm_inspection_baseline_eval.py \
  --stage pilot \
  --methods qrl_mppi,qrl_greedy \
  --output-dir "$OUTPUT_DIR" \
  --qrl-checkpoints "$QRL_CHECKPOINT" \
  --starts-per-device "${STARTS_PER_DEVICE:-30}" \
  --seed "${SEED:-20260716}" \
  --device "${DEVICE:-mps}" \
  --bounds 0 0 10 10 \
  --omega-max 3.0 \
  --v 1.0 \
  --dt 0.1 \
  --max-episode-steps 180 \
  --obstacle-config medium \
  --device-catalog "$DEVICE_CATALOG" \
  --comm-alpha 2.0 \
  --comm-bias 5.0 \
  --comm-occlusion-penalty 6.0 \
  --comm-threshold 0.5 \
  --collision-cost 10.0 \
  --out-of-bounds-cost 10.0 \
  --communication-break-cost 1.0 \
  --observation-violation-cost-weight 1.0 \
  --communication-violation-cost-weight 0.5 \
  --observation-failure-cost 0.25 \
  --num-critics 2 \
  --mppi-horizon 10 \
  --mppi-num-samples 128 \
  --mppi-noise-sigma 0.8 \
  --mppi-terminal-weight 0.1 \
  --mppi-temperature 1.0 \
  --bootstrap-samples 2000 \
  --save-visualizations \
  --viz-max-successes 10 \
  --viz-max-failures 10 \
  "${RESUME_ARGS[@]}"

echo "Evaluation complete: $OUTPUT_DIR/baseline_results.json"
