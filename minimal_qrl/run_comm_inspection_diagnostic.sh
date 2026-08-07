#!/usr/bin/env bash
# Prepare, train, evaluate, or benchmark the fixed long-horizon diagnostic scenario.
#
# The optimization/training defaults intentionally mirror:
#   - run_comm_inspection_train.sh
#   - run_comm_inspection_execution_eval.sh
#   - run_comm_inspection_baselines.sh
# Map geometry, communication regions, the task bank, and the episode horizon come
# from diagnostic_scenario.json and are the only intentional environment changes.
#
# Examples:
#   PHASE=prepare bash minimal_qrl/run_comm_inspection_diagnostic.sh
#   PHASE=visualize bash minimal_qrl/run_comm_inspection_diagnostic.sh
#   PHASE=train_qrl DEVICE=mps bash minimal_qrl/run_comm_inspection_diagnostic.sh
#   PHASE=eval_qrl QRL_CHECKPOINT=... bash minimal_qrl/run_comm_inspection_diagnostic.sh
#   PHASE=benchmark QRL_CHECKPOINTS="..." \
#     TRAIN_SAC=1 TRAIN_CONTEXT_AGENTS=1 \
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
  local oracle_bank_eval_flag=()

  if [[ "${ORACLE_BANK_EVAL:-1}" == "1" ]]; then
    oracle_bank_eval_flag+=(--oracle-bank-eval)
  fi

  "$PYTHON_BIN" minimal_qrl/train.py \
    --scenario-config "$SCENARIO_CONFIG" \
    --output-dir "$TRAIN_DIR" \
    --seed "${SEED:-42}" \
    --device "${DEVICE:-cpu}" \
    --num-episodes "${NUM_EPISODES:-500}" \
    --target-env-transitions "${TARGET_ENV_TRANSITIONS:-120000}" \
    --batch-size "${BATCH_SIZE:-256}" \
    --total-steps "${TOTAL_STEPS:-120000}" \
    --num-critics "${NUM_CRITICS:-2}" \
    --qrl-cost-source "${QRL_COST_SOURCE:-negative_reward}" \
    --global-push-softplus-offset "${GLOBAL_PUSH_SOFTPLUS_OFFSET:-15.0}" \
    --global-push-softplus-beta "${GLOBAL_PUSH_SOFTPLUS_BETA:-0.1}" \
    --global-push-abstract-goal-ratio "${GLOBAL_PUSH_ABSTRACT_GOAL_RATIO:-0.6}" \
    --global-push-state-goal-ratio "${GLOBAL_PUSH_STATE_GOAL_RATIO:-0.4}" \
    --abstract-goal-edge-loss-weight "${ABSTRACT_GOAL_EDGE_LOSS_WEIGHT:-1.0}" \
    --task-aware-teacher-ratio "${TASK_AWARE_TEACHER_RATIO:-1.0}" \
    --log-interval "${LOG_INTERVAL:-100}" \
    --save-interval "${SAVE_INTERVAL:-2000}" \
    --eval-interval "${EVAL_INTERVAL:-1000}" \
    ${oracle_bank_eval_flag[@]+"${oracle_bank_eval_flag[@]}"} \
    --oracle-bank-dir "${ORACLE_BANK_DIR:-$TRAIN_DIR/oracle_banks}" \
    --oracle-bank-size "${ORACLE_BANK_SIZE:-192}" \
    --oracle-bank-seed "${ORACLE_BANK_SEED:-20260729}" \
    --oracle-astar-timeout-sec "${ORACLE_ASTAR_TIMEOUT_SEC:-60}" \
    --oracle-final-bootstrap-samples "${ORACLE_FINAL_BOOTSTRAP_SAMPLES:-2000}" \
    --visualization-interval "${VIS_INTERVAL:-1000}" \
    --planning-eval-interval 0
}

eval_qrl() {
  local checkpoint="${QRL_CHECKPOINT:-$TRAIN_DIR/checkpoint_final.pth}"
  local eval_dir="${QRL_EVAL_DIR:-$OUTPUT_ROOT/qrl_eval}"
  local save_visualizations_flag=()
  local viz_save_gif_flag=()

  if [[ "${SAVE_VISUALIZATIONS:-0}" == "1" ]]; then
    save_visualizations_flag+=(--save-visualizations)
  fi
  if [[ "${VIZ_SAVE_GIF:-0}" == "1" ]]; then
    viz_save_gif_flag+=(--viz-save-gif)
  fi

  "$PYTHON_BIN" minimal_qrl/eval/comm_inspection_execution_eval.py \
    --checkpoint "$checkpoint" \
    --output-dir "$eval_dir" \
    --scenario-config "$SCENARIO_CONFIG" \
    --task-bank "$TASK_BANK" \
    --task-split "${TASK_SPLIT:-test}" \
    --num-critics "${NUM_CRITICS:-2}" \
    --starts-per-device "${STARTS_PER_DEVICE:-50}" \
    --seed "${SEED:-0}" \
    --device "${DEVICE:-auto}" \
    --execution-modes "${EXECUTION_MODES:-greedy,lookahead}" \
    --lookahead-horizon "${LOOKAHEAD_HORIZON:-10}" \
    --lookahead-num-sequences "${LOOKAHEAD_NUM_SEQUENCES:-128}" \
    --lookahead-heuristics "${LOOKAHEAD_HEURISTICS:-dense}" \
    --lookahead-step-cost-weight "${LOOKAHEAD_STEP_COST_WEIGHT:-0.0}" \
    --lookahead-collision-penalty "${LOOKAHEAD_COLLISION_PENALTY:-0.0}" \
    --lookahead-biased-sequences "${LOOKAHEAD_BIASED_SEQUENCES:-24}" \
    --lookahead-bias-kp "${LOOKAHEAD_BIAS_KP:-2.0}" \
    --planner-qrl-progress-alpha "${PLANNER_QRL_PROGRESS_ALPHA:-1.0}" \
    ${save_visualizations_flag[@]+"${save_visualizations_flag[@]}"} \
    --viz-max-successes "${VIZ_MAX_SUCCESSES:-10}" \
    --viz-max-failures "${VIZ_MAX_FAILURES:-10}" \
    ${viz_save_gif_flag[@]+"${viz_save_gif_flag[@]}"} \
    --viz-gif-fps "${VIZ_GIF_FPS:-8}"
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
  local stage="${STAGE:-pilot}"
  local benchmark_dir="${BENCHMARK_DIR:-$OUTPUT_ROOT/benchmark}"
  local methods="${METHODS:-hybrid_astar,mppi_no_terminal,model_mppi,goal_set_sac,qrl_greedy,qrl_mppi,context_her_ddpg,context_her_ddpg_mppi,context_contrastive_rl,context_contrastive_rl_mppi,mrn_context_her_ddpg,mrn_context_her_ddpg_mppi}"
  local starts_per_device="${STARTS_PER_DEVICE:-25}"
  local qrl_checkpoints="${QRL_CHECKPOINTS:-${QRL_CHECKPOINT:-$TRAIN_DIR/checkpoint_final.pth}}"
  local extra_args=()
  local qrl_array=()
  local context_array=()
  local sac_array=()
  local train_sac_flag=()
  local train_context_flag=()
  local resume_flag=()
  local save_visualizations_flag=()

  read -r -a qrl_array <<< "$qrl_checkpoints"

  if [[ "${TRAIN_SAC:-1}" == "1" ]]; then
    train_sac_flag+=(--train-sac)
  fi
  if [[ "${TRAIN_CONTEXT_AGENTS:-1}" == "1" ]]; then
    train_context_flag+=(--train-context-agents)
  fi
  if [[ "${RESUME:-0}" == "1" ]]; then
    resume_flag+=(--resume)
  fi
  if [[ "${SAVE_VISUALIZATIONS:-0}" == "1" ]]; then
    save_visualizations_flag+=(--save-visualizations)
  fi

  extra_args+=(--starts-per-device "$starts_per_device")
  if [[ -n "${SAC_TOTAL_ENV_STEPS:-}" ]]; then
    extra_args+=(--sac-total-env-steps "$SAC_TOTAL_ENV_STEPS")
  fi
  if [[ -n "${SAC_SEEDS:-}" ]]; then
    extra_args+=(--sac-seeds "$SAC_SEEDS")
  fi
  if [[ -n "${SAC_BATCH_SIZE:-}" ]]; then
    extra_args+=(--sac-batch-size "$SAC_BATCH_SIZE")
  fi
  if [[ -n "${CONTEXT_TOTAL_ENV_STEPS:-}" ]]; then
    extra_args+=(--context-total-env-steps "$CONTEXT_TOTAL_ENV_STEPS")
  fi
  if [[ -n "${CONTEXT_SEEDS:-}" ]]; then
    extra_args+=(--context-seeds "$CONTEXT_SEEDS")
  fi
  if [[ -n "${CONTEXT_CHECKPOINTS:-}" ]]; then
    read -r -a context_array <<< "$CONTEXT_CHECKPOINTS"
    extra_args+=(--context-checkpoints "${context_array[@]}")
  fi
  extra_args+=(--context-batch-size "${CONTEXT_BATCH_SIZE:-256}")
  extra_args+=(--context-her-k "${CONTEXT_HER_K:-4}")
  extra_args+=(--context-teacher-ratio "${CONTEXT_TEACHER_RATIO:-1.0}")

  if [[ "${TRAIN_SAC:-1}" != "1" ]]; then
    if [[ -n "${SAC_CHECKPOINTS:-}" ]]; then
      read -r -a sac_array <<< "$SAC_CHECKPOINTS"
    else
      shopt -s nullglob
      for checkpoint in "$benchmark_dir"/goal_set_sac/seed_*/checkpoint_final.pth; do
        sac_array+=("$checkpoint")
      done
      shopt -u nullglob
    fi
    if (( ${#sac_array[@]} > 0 )); then
      extra_args+=(--sac-checkpoints "${sac_array[@]}")
    fi
  fi

  echo "Diagnostic baseline configuration:"
  echo "  stage=$stage"
  echo "  methods=$methods"
  echo "  starts_per_device=$starts_per_device (task bank fixes the actual tasks)"
  echo "  output_dir=$benchmark_dir"
  echo "  resume=${RESUME:-0}"

  # As in the formal baseline script, A* resolution, heading bins, primitive
  # steps, heuristic weight, and expansion cap use the evaluator defaults.
  "$PYTHON_BIN" minimal_qrl/eval/comm_inspection_baseline_eval.py \
    --stage "$stage" \
    --methods "$methods" \
    --output-dir "$benchmark_dir" \
    --qrl-checkpoints "${qrl_array[@]}" \
    ${train_sac_flag[@]+"${train_sac_flag[@]}"} \
    ${train_context_flag[@]+"${train_context_flag[@]}"} \
    ${resume_flag[@]+"${resume_flag[@]}"} \
    --scenario-config "$SCENARIO_CONFIG" \
    --task-bank "$TASK_BANK" \
    --task-split "${TASK_SPLIT:-test}" \
    --seed "${SEED:-0}" \
    --device "${DEVICE:-auto}" \
    --mppi-horizon "${MPPI_HORIZON:-10}" \
    --mppi-num-samples "${MPPI_NUM_SAMPLES:-128}" \
    --astar-timeout-sec "${ASTAR_TIMEOUT_SEC:-30}" \
    --astar-terminal-samples "${ASTAR_TERMINAL_SAMPLES:-128}" \
    ${save_visualizations_flag[@]+"${save_visualizations_flag[@]}"} \
    --viz-max-successes "${VIZ_MAX_SUCCESSES:-10}" \
    --viz-max-failures "${VIZ_MAX_FAILURES:-10}" \
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
