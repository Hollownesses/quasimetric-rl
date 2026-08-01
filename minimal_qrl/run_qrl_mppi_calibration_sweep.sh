#!/usr/bin/env bash
# Evaluate the GP500 QRL checkpoint with four MPPI calibration settings.

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -x "./.venv/bin/python" ]]; then
  QRL_EVAL_PYTHON="${PYTHON_BIN:-./.venv/bin/python}"
elif [[ -x "../quasimetric-rl/.venv/bin/python" ]]; then
  QRL_EVAL_PYTHON="${PYTHON_BIN:-../quasimetric-rl/.venv/bin/python}"
else
  QRL_EVAL_PYTHON="${PYTHON_BIN:-python3}"
fi

QRL_CHECKPOINT="${QRL_CHECKPOINT:-./results/industrial_inspection_gp500/checkpoint_final.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./results/industrial_inspection_gp500}"

run_qrl_mppi_eval() {
  local run_name="$1"
  local terminal_weight="$2"
  local temperature="$3"
  local output_dir="${OUTPUT_ROOT}/eval_res_qrl_${run_name}"

  echo "Running QRL-MPPI: terminal_weight=${terminal_weight}, temperature=${temperature}"
  echo "Output: ${output_dir}"

  "$QRL_EVAL_PYTHON" minimal_qrl/eval/comm_inspection_baseline_eval.py \
    --stage pilot \
    --methods qrl_mppi,qrl_greedy \
    --output-dir "$output_dir" \
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
    --device-catalog "./minimal_qrl/configs/chemical_process_plant_devices.json" \
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
    --mppi-terminal-weight "$terminal_weight" \
    --mppi-temperature "$temperature" \
    --bootstrap-samples 2000 \
    --save-visualizations \
    --viz-max-successes 10 \
    --viz-max-failures 10
}

run_qrl_mppi_eval "tw015_t1" 0.15 1.0
run_qrl_mppi_eval "tw005_t1" 0.05 1.0

echo "All QRL-MPPI calibration evaluations completed."
