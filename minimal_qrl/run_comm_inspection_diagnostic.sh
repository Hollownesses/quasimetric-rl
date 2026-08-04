#!/usr/bin/env bash
# Prepare, train, or benchmark the fixed long-horizon diagnostic scenario.
#
# Examples:
#   PHASE=prepare bash minimal_qrl/run_comm_inspection_diagnostic.sh
#   PHASE=visualize bash minimal_qrl/run_comm_inspection_diagnostic.sh
#   PHASE=train_qrl DEVICE=mps bash minimal_qrl/run_comm_inspection_diagnostic.sh
#   PHASE=eval_qrl QRL_CHECKPOINT=... bash minimal_qrl/run_comm_inspection_diagnostic.sh
#   PHASE=benchmark QRL_CHECKPOINTS="..." CONTEXT_CHECKPOINTS="..." \
#     bash minimal_qrl/run_comm_inspection_diagnostic.sh

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -x "./.venv/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-./.venv/bin/python}"
elif [[ -x "../quasimetric-rl/.venv/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-../quasimetric-rl/.venv/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

PHASE="${PHASE:-prepare}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./results/diagnostic_u_shadow_corridors}"
CONFIG_DIR="$OUTPUT_ROOT/config"
SCENARIO_CONFIG="$CONFIG_DIR/diagnostic_scenario.json"
TASK_BANK="$CONFIG_DIR/diagnostic_task_bank.json"
TRAIN_DIR="${TRAIN_DIR:-$OUTPUT_ROOT/qrl_training}"

"$PYTHON_BIN" -m minimal_qrl.industry_exp.diagnostic_scenario \
  --output-dir "$CONFIG_DIR"

train_qrl() {
  "$PYTHON_BIN" minimal_qrl/train.py \
    --scenario-config "$SCENARIO_CONFIG" \
    --output-dir "$TRAIN_DIR" \
    --seed "${SEED:-0}" \
    --device "${DEVICE:-auto}" \
    --target-env-transitions "${TARGET_ENV_TRANSITIONS:-120000}" \
    --total-steps "${TOTAL_STEPS:-120000}" \
    --batch-size "${BATCH_SIZE:-256}" \
    --num-critics "${NUM_CRITICS:-2}" \
    --task-aware-teacher-ratio "${TASK_AWARE_TEACHER_RATIO:-1.0}" \
    --save-interval "${SAVE_INTERVAL:-5000}" \
    --eval-interval "${EVAL_INTERVAL:-0}" \
    --visualization-interval "${VISUALIZATION_INTERVAL:-0}" \
    --planning-eval-interval 0
}

eval_qrl() {
  local checkpoint="${QRL_CHECKPOINT:-$TRAIN_DIR/checkpoint_final.pth}"
  "$PYTHON_BIN" minimal_qrl/eval/comm_inspection_execution_eval.py \
    --checkpoint "$checkpoint" \
    --output-dir "${QRL_EVAL_DIR:-$OUTPUT_ROOT/qrl_eval}" \
    --scenario-config "$SCENARIO_CONFIG" \
    --task-bank "$TASK_BANK" \
    --task-split "${TASK_SPLIT:-test}" \
    --execution-modes "${EXECUTION_MODES:-greedy,lookahead}" \
    --lookahead-horizon "${LOOKAHEAD_HORIZON:-20}" \
    --lookahead-num-sequences "${LOOKAHEAD_NUM_SEQUENCES:-256}" \
    --lookahead-heuristics "${LOOKAHEAD_HEURISTICS:-terminal}" \
    --seed "${SEED:-0}" \
    --device "${DEVICE:-auto}"
}

visualize() {
  "$PYTHON_BIN" -m minimal_qrl.visualize_diagnostic_scenarios \
    --scenario-config "$SCENARIO_CONFIG" \
    --task-bank "$TASK_BANK" \
    --split "${TASK_SPLIT:-validation}" \
    --sample-index "${SAMPLE_INDEX:-0}" \
    --communication-resolution "${COMMUNICATION_RESOLUTION:-180}" \
    --dpi "${VIZ_DPI:-180}" \
    --output-dir "${VISUALIZATION_DIR:-$OUTPUT_ROOT/visualizations}"
}

benchmark() {
  local qrl_checkpoints="${QRL_CHECKPOINTS:-${QRL_CHECKPOINT:-$TRAIN_DIR/checkpoint_final.pth}}"
  local extra_args=()
  local qrl_array=()
  local context_array=()
  read -r -a qrl_array <<< "$qrl_checkpoints"
  if [[ -n "${CONTEXT_CHECKPOINTS:-}" ]]; then
    read -r -a context_array <<< "$CONTEXT_CHECKPOINTS"
    extra_args+=(--context-checkpoints "${context_array[@]}")
  fi
  if [[ "${TRAIN_CONTEXT_AGENTS:-0}" == "1" ]]; then
    extra_args+=(--train-context-agents)
  fi
  "$PYTHON_BIN" minimal_qrl/eval/comm_inspection_baseline_eval.py \
    --stage "${STAGE:-pilot}" \
    --methods "${METHODS:-hybrid_astar,mppi_no_terminal,model_mppi,goal_set_sac,qrl_greedy,qrl_mppi,context_her_ddpg,context_her_ddpg_mppi,context_contrastive_rl,context_contrastive_rl_mppi,mrn_context_her_ddpg,mrn_context_her_ddpg_mppi}" \
    --output-dir "${BENCHMARK_DIR:-$OUTPUT_ROOT/benchmark}" \
    --qrl-checkpoints "${qrl_array[@]}" \
    --scenario-config "$SCENARIO_CONFIG" \
    --task-bank "$TASK_BANK" \
    --task-split "${TASK_SPLIT:-test}" \
    --seed "${SEED:-0}" \
    --device "${DEVICE:-auto}" \
    --mppi-horizon "${MPPI_HORIZON:-20}" \
    --mppi-num-samples "${MPPI_NUM_SAMPLES:-256}" \
    --astar-position-resolution "${ASTAR_POSITION_RESOLUTION:-0.3}" \
    --astar-heading-bins "${ASTAR_HEADING_BINS:-20}" \
    --astar-heuristic-weight "${ASTAR_HEURISTIC_WEIGHT:-25.0}" \
    --astar-timeout-sec "${ASTAR_TIMEOUT_SEC:-60.0}" \
    --astar-terminal-samples "${ASTAR_TERMINAL_SAMPLES:-64}" \
    "${extra_args[@]}"
}

case "$PHASE" in
  prepare)
    ;;
  visualize)
    visualize
    ;;
  train_qrl)
    train_qrl
    ;;
  eval_qrl)
    eval_qrl
    ;;
  benchmark)
    benchmark
    ;;
  all)
    visualize
    train_qrl
    eval_qrl
    benchmark
    ;;
  *)
    echo "Unknown PHASE=$PHASE (expected prepare, visualize, train_qrl, eval_qrl, benchmark, or all)" >&2
    exit 2
    ;;
esac

echo "Diagnostic scenario: $SCENARIO_CONFIG"
echo "Fixed task bank:    $TASK_BANK"
