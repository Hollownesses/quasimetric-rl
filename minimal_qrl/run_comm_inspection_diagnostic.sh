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
#   PHASE=train_qrl QRL_DATASET_MODE=qrl_explore \
#     OUTPUT_ROOT=./results/diagnostic_u_shadow_corridors_explore \
#     bash minimal_qrl/run_comm_inspection_diagnostic.sh
#   PHASE=eval_qrl QRL_CHECKPOINT=... bash minimal_qrl/run_comm_inspection_diagnostic.sh
#   PHASE=local_nav_eval QRL_CHECKPOINTS="checkpoint_a.pth checkpoint_b.pth" \
#     bash minimal_qrl/run_comm_inspection_diagnostic.sh
#   PHASE=oracle_mppi bash minimal_qrl/run_comm_inspection_diagnostic.sh
#   PHASE=supervised_iqe bash minimal_qrl/run_comm_inspection_diagnostic.sh
#   PHASE=targeted_supervised_iqe bash minimal_qrl/run_comm_inspection_diagnostic.sh
#   PHASE=dense_transition_qrl DEVICE=mps bash minimal_qrl/run_comm_inspection_diagnostic.sh
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
SHARED_ORACLE_DIR="${SHARED_ORACLE_DIR:-./results/shared_oracle_banks/chemical_process}"
ORACLE_VALIDATION_BANK="${ORACLE_VALIDATION_BANK:-$SHARED_ORACLE_DIR/hybrid_astar_validation_192.json}"
ORACLE_FINAL_TEST_BANK="${ORACLE_FINAL_TEST_BANK:-$SHARED_ORACLE_DIR/hybrid_astar_final_test_192.json}"

"$PYTHON_BIN" -m minimal_qrl.industry_exp.diagnostic_scenario \
  --output-dir "$CONFIG_DIR"

train_qrl() {
  local oracle_bank_eval_flag=()
  local dataset_budget_args=()
  local qrl_dataset_mode="${QRL_DATASET_MODE:-standard}"
  local teacher_ratio="${TASK_AWARE_TEACHER_RATIO:-1.0}"

  if [[ "${ORACLE_BANK_EVAL:-1}" == "1" ]]; then
    oracle_bank_eval_flag+=(--oracle-bank-eval)
    if [[ ! -f "$ORACLE_VALIDATION_BANK" ]]; then
      echo "Missing fixed validation oracle bank: $ORACLE_VALIDATION_BANK" >&2
      exit 1
    fi
    if [[ ! -f "$ORACLE_FINAL_TEST_BANK" ]]; then
      echo "Missing fixed final-test oracle bank: $ORACLE_FINAL_TEST_BANK" >&2
      exit 1
    fi
  fi

  if [[ "$qrl_dataset_mode" == "qrl_explore" ]]; then
    # QRL-explore is deliberately expert-free.  Keep this invariant even if a
    # caller has TASK_AWARE_TEACHER_RATIO set in the surrounding shell.
    teacher_ratio="0.0"
    dataset_budget_args+=(
      --comm-dataset-mode "$qrl_dataset_mode"
      --explore-attempted-env-steps "${EXPLORE_ATTEMPTED_ENV_STEPS:-200000}"
      --explore-start-position-resolution "${EXPLORE_START_POSITION_RESOLUTION:-1.0}"
      --explore-start-heading-bins "${EXPLORE_START_HEADING_BINS:-12}"
      --explore-action-hold-min-steps "${EXPLORE_ACTION_HOLD_MIN_STEPS:-3}"
      --explore-action-hold-max-steps "${EXPLORE_ACTION_HOLD_MAX_STEPS:-10}"
      --explore-straight-action-probability "${EXPLORE_STRAIGHT_ACTION_PROBABILITY:-0.5}"
      --explore-start-boundary-margin "${EXPLORE_START_BOUNDARY_MARGIN:-0.5}"
      --explore-local-safety-lookahead-steps "${EXPLORE_LOCAL_SAFETY_LOOKAHEAD_STEPS:-10}"
      --explore-exclusion-task-bank "${EXPLORE_EXCLUSION_TASK_BANK:-$TASK_BANK}"
      --explore-exclusion-radius "${EXPLORE_EXCLUSION_RADIUS:-0.25}"
    )
  elif [[ "$qrl_dataset_mode" == "standard" ]]; then
    dataset_budget_args+=(--target-env-transitions "${TARGET_ENV_TRANSITIONS:-120000}")
  else
    echo "Unknown QRL_DATASET_MODE=$qrl_dataset_mode (expected standard or qrl_explore)" >&2
    exit 2
  fi

  "$PYTHON_BIN" minimal_qrl/train.py \
    --scenario-config "$SCENARIO_CONFIG" \
    --output-dir "$TRAIN_DIR" \
    --seed "${SEED:-42}" \
    --device "${DEVICE:-cpu}" \
    --num-episodes "${NUM_EPISODES:-500}" \
    ${dataset_budget_args[@]+"${dataset_budget_args[@]}"} \
    --batch-size "${BATCH_SIZE:-256}" \
    --total-steps "${TOTAL_STEPS:-120000}" \
    --num-critics "${NUM_CRITICS:-2}" \
    --qrl-cost-source "${QRL_COST_SOURCE:-negative_reward}" \
    --global-push-softplus-offset "${GLOBAL_PUSH_SOFTPLUS_OFFSET:-15.0}" \
    --global-push-softplus-beta "${GLOBAL_PUSH_SOFTPLUS_BETA:-0.1}" \
    --global-push-abstract-goal-ratio "${GLOBAL_PUSH_ABSTRACT_GOAL_RATIO:-0.6}" \
    --global-push-state-goal-ratio "${GLOBAL_PUSH_STATE_GOAL_RATIO:-0.4}" \
    --abstract-goal-edge-loss-weight "${ABSTRACT_GOAL_EDGE_LOSS_WEIGHT:-1.0}" \
    --qrl-temporal-constraint-weight "${QRL_TEMPORAL_CONSTRAINT_WEIGHT:-1.0}" \
    --qrl-temporal-min-future-steps "${QRL_TEMPORAL_MIN_FUTURE_STEPS:-2}" \
    --qrl-goal-return-constraint-weight "${QRL_GOAL_RETURN_CONSTRAINT_WEIGHT:-1.0}" \
    --qrl-nstep-goal-constraint-weight "${QRL_NSTEP_GOAL_CONSTRAINT_WEIGHT:-0.0}" \
    --qrl-nstep-target-tau "${QRL_NSTEP_TARGET_TAU:-0.005}" \
    --qrl-success-transition-weight "${QRL_SUCCESS_TRANSITION_WEIGHT:-4.0}" \
    --task-aware-teacher-ratio "$teacher_ratio" \
    --log-interval "${LOG_INTERVAL:-100}" \
    --save-interval "${SAVE_INTERVAL:-2000}" \
    --eval-interval "${EVAL_INTERVAL:-1000}" \
    ${oracle_bank_eval_flag[@]+"${oracle_bank_eval_flag[@]}"} \
    --oracle-bank-dir "${ORACLE_BANK_DIR:-$TRAIN_DIR/oracle_banks}" \
    --oracle-validation-bank "$ORACLE_VALIDATION_BANK" \
    --oracle-final-test-bank "$ORACLE_FINAL_TEST_BANK" \
    --oracle-bank-size "${ORACLE_BANK_SIZE:-192}" \
    --oracle-bank-seed "${ORACLE_BANK_SEED:-20260729}" \
    --oracle-astar-timeout-sec "${ORACLE_ASTAR_TIMEOUT_SEC:-60}" \
    --oracle-final-bootstrap-samples "${ORACLE_FINAL_BOOTSTRAP_SAMPLES:-2000}" \
    --visualization-interval "${VIS_INTERVAL:-1000}" \
    --planning-eval-interval 0
}

local_nav_eval() {
  local qrl_checkpoints="${QRL_CHECKPOINTS:-${QRL_CHECKPOINT:-$TRAIN_DIR/checkpoint_final.pth}}"
  local checkpoint_array=()
  local reuse_oracle_args=()
  read -r -a checkpoint_array <<< "$qrl_checkpoints"
  if (( ${#checkpoint_array[@]} == 0 )); then
    echo "QRL_CHECKPOINTS must contain at least one checkpoint" >&2
    exit 2
  fi
  if [[ -n "${LOCAL_NAV_REUSE_ORACLE_JSON:-}" ]]; then
    reuse_oracle_args+=(--reuse-oracle-json "$LOCAL_NAV_REUSE_ORACLE_JSON")
  fi

  "$PYTHON_BIN" -m minimal_qrl.eval.u_trap_local_navigability \
    --scenario-config "$SCENARIO_CONFIG" \
    --checkpoints "${checkpoint_array[@]}" \
    --output-dir "${LOCAL_NAV_EVAL_DIR:-$OUTPUT_ROOT/u_trap_local_navigability}" \
    --device "${DEVICE:-auto}" \
    --num-critics "${NUM_CRITICS:-2}" \
    --seed "${LOCAL_NAV_SEED:-20260802}" \
    --astar-position-resolution "${LOCAL_NAV_ASTAR_POSITION_RESOLUTION:-0.25}" \
    --astar-heading-bins "${LOCAL_NAV_ASTAR_HEADING_BINS:-24}" \
    --astar-primitive-steps "${LOCAL_NAV_ASTAR_PRIMITIVE_STEPS:-5}" \
    --astar-heuristic-weight "${LOCAL_NAV_ASTAR_HEURISTIC_WEIGHT:-1.0}" \
    --astar-max-expansions "${LOCAL_NAV_ASTAR_MAX_EXPANSIONS:-200000}" \
    --astar-timeout-sec "${LOCAL_NAV_ASTAR_TIMEOUT_SEC:-120}" \
    --astar-terminal-samples "${LOCAL_NAV_ASTAR_TERMINAL_SAMPLES:-128}" \
    ${reuse_oracle_args[@]+"${reuse_oracle_args[@]}"}
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
    --mppi-terminal-weight "${MPPI_TERMINAL_WEIGHT:-1.0}" \
    --astar-timeout-sec "${ASTAR_TIMEOUT_SEC:-30}" \
    --astar-terminal-samples "${ASTAR_TERMINAL_SAMPLES:-128}" \
    ${save_visualizations_flag[@]+"${save_visualizations_flag[@]}"} \
    --viz-max-successes "${VIZ_MAX_SUCCESSES:-10}" \
    --viz-max-failures "${VIZ_MAX_FAILURES:-10}" \
    "${extra_args[@]}"
}

oracle_mppi() {
  local oracle_dir="${ORACLE_MPPI_DIR:-$OUTPUT_ROOT/oracle_mppi_test_u_trap}"
  local resume_flag=()
  local save_visualizations_flag=()
  if [[ "${RESUME:-0}" == "1" ]]; then
    resume_flag+=(--resume)
  fi
  if [[ "${SAVE_VISUALIZATIONS:-0}" == "1" ]]; then
    save_visualizations_flag+=(--save-visualizations)
  fi

  echo "Oracle-MPPI U-trap configuration:"
  echo "  split=${TASK_SPLIT:-test}, stratum=u_trap (12 test episodes)"
  echo "  horizon=${MPPI_HORIZON:-10}, samples=${MPPI_NUM_SAMPLES:-128}"
  echo "  output_dir=$oracle_dir"

  "$PYTHON_BIN" minimal_qrl/eval/comm_inspection_baseline_eval.py \
    --stage pilot \
    --methods oracle_mppi \
    --output-dir "$oracle_dir" \
    --scenario-config "$SCENARIO_CONFIG" \
    --task-bank "$TASK_BANK" \
    --task-split "${TASK_SPLIT:-test}" \
    --task-strata u_trap \
    --seed "${SEED:-0}" \
    --device cpu \
    --mppi-horizon "${MPPI_HORIZON:-10}" \
    --mppi-num-samples "${MPPI_NUM_SAMPLES:-128}" \
    --mppi-noise-sigma "${MPPI_NOISE_SIGMA:-0.8}" \
    --mppi-temperature "${MPPI_TEMPERATURE:-1.0}" \
    --mppi-terminal-weight "${MPPI_TERMINAL_WEIGHT:-1.0}" \
    --astar-position-resolution "${ASTAR_POSITION_RESOLUTION:-0.25}" \
    --astar-heading-bins "${ASTAR_HEADING_BINS:-24}" \
    --astar-primitive-steps "${ASTAR_PRIMITIVE_STEPS:-5}" \
    --astar-heuristic-weight "${ASTAR_HEURISTIC_WEIGHT:-1.0}" \
    --astar-max-expansions "${ASTAR_MAX_EXPANSIONS:-50000}" \
    --astar-timeout-sec "${ASTAR_TIMEOUT_SEC:-30}" \
    --astar-terminal-samples "${ASTAR_TERMINAL_SAMPLES:-128}" \
    --oracle-value-cache-dir "${ORACLE_VALUE_CACHE_DIR:-$OUTPUT_ROOT/oracle_value_cache}" \
    ${resume_flag[@]+"${resume_flag[@]}"} \
    ${save_visualizations_flag[@]+"${save_visualizations_flag[@]}"} \
    --viz-max-successes "${VIZ_MAX_SUCCESSES:-12}" \
    --viz-max-failures "${VIZ_MAX_FAILURES:-12}"
}

supervised_iqe() {
  local experiment_dir="${SUPERVISED_IQE_DIR:-$OUTPUT_ROOT/supervised_iqe_oracle}"
  local checkpoint="$experiment_dir/checkpoint_final.pth"
  local mppi_dir="${SUPERVISED_IQE_MPPI_DIR:-$experiment_dir/mppi_test_u_trap}"
  local save_visualizations_flag=()
  if [[ "${SAVE_VISUALIZATIONS:-0}" == "1" ]]; then
    save_visualizations_flag+=(--save-visualizations)
  fi

  "$PYTHON_BIN" -m minimal_qrl.industry_exp.supervised_iqe_oracle \
    --scenario-config "$SCENARIO_CONFIG" \
    --output-dir "$experiment_dir" \
    --device "${DEVICE:-auto}" \
    --seed "${SUPERVISED_IQE_SEED:-20260823}" \
    --num-critics "${NUM_CRITICS:-2}" \
    --train-samples "${SUPERVISED_IQE_TRAIN_SAMPLES:-200000}" \
    --eval-samples "${SUPERVISED_IQE_EVAL_SAMPLES:-20000}" \
    --low-cost-fraction "${SUPERVISED_IQE_LOW_COST_FRACTION:-0.25}" \
    --train-steps "${SUPERVISED_IQE_TRAIN_STEPS:-10000}" \
    --batch-size "${SUPERVISED_IQE_BATCH_SIZE:-512}" \
    --learning-rate "${SUPERVISED_IQE_LR:-0.0001}" \
    --loss "${SUPERVISED_IQE_LOSS:-huber}" \
    --huber-delta "${SUPERVISED_IQE_HUBER_DELTA:-10}" \
    --oracle-value-cache-dir "${ORACLE_VALUE_CACHE_DIR:-$OUTPUT_ROOT/oracle_value_cache}" \
    --astar-position-resolution "${ASTAR_POSITION_RESOLUTION:-0.25}" \
    --astar-heading-bins "${ASTAR_HEADING_BINS:-24}" \
    --astar-primitive-steps "${ASTAR_PRIMITIVE_STEPS:-5}"

  "$PYTHON_BIN" minimal_qrl/eval/comm_inspection_baseline_eval.py \
    --stage pilot \
    --methods supervised_iqe_mppi \
    --output-dir "$mppi_dir" \
    --qrl-checkpoints "$checkpoint" \
    --scenario-config "$SCENARIO_CONFIG" \
    --task-bank "$TASK_BANK" \
    --task-split test \
    --task-strata u_trap \
    --seed "${SEED:-0}" \
    --device "${DEVICE:-auto}" \
    --num-critics "${NUM_CRITICS:-2}" \
    --mppi-horizon "${MPPI_HORIZON:-10}" \
    --mppi-num-samples "${MPPI_NUM_SAMPLES:-128}" \
    --mppi-noise-sigma "${MPPI_NOISE_SIGMA:-0.8}" \
    --mppi-temperature "${MPPI_TEMPERATURE:-1.0}" \
    --mppi-terminal-weight "${MPPI_TERMINAL_WEIGHT:-1.0}" \
    ${save_visualizations_flag[@]+"${save_visualizations_flag[@]}"} \
    --viz-max-successes "${VIZ_MAX_SUCCESSES:-12}" \
    --viz-max-failures "${VIZ_MAX_FAILURES:-12}"
}

targeted_supervised_iqe() {
  local experiment_dir="${TARGETED_SUPERVISED_IQE_DIR:-$OUTPUT_ROOT/targeted_supervised_iqe_oracle}"
  local checkpoint="$experiment_dir/checkpoint_final.pth"
  local mppi_dir="${TARGETED_SUPERVISED_IQE_MPPI_DIR:-$experiment_dir/mppi_test_u_trap}"
  local failure_results="${TARGETED_FAILURE_RESULTS:-$OUTPUT_ROOT/supervised_iqe_oracle/mppi_test_u_trap/baseline_results.json}"
  local save_visualizations_flag=()
  if [[ ! -f "$failure_results" ]]; then
    echo "Missing prior Supervised-IQE results used to locate failed starts: $failure_results" >&2
    exit 1
  fi
  if [[ "${SAVE_VISUALIZATIONS:-0}" == "1" ]]; then
    save_visualizations_flag+=(--save-visualizations)
  fi

  "$PYTHON_BIN" -m minimal_qrl.industry_exp.supervised_iqe_oracle \
    --scenario-config "$SCENARIO_CONFIG" \
    --output-dir "$experiment_dir" \
    --device "${DEVICE:-auto}" \
    --seed "${SUPERVISED_IQE_SEED:-20260823}" \
    --num-critics "${NUM_CRITICS:-2}" \
    --train-samples "${SUPERVISED_IQE_TRAIN_SAMPLES:-200000}" \
    --eval-samples "${SUPERVISED_IQE_EVAL_SAMPLES:-20000}" \
    --low-cost-fraction "${SUPERVISED_IQE_LOW_COST_FRACTION:-0.25}" \
    --sampling-mode targeted_u_trap \
    --targeted-local-fraction "${TARGETED_LOCAL_FRACTION:-0.5}" \
    --targeted-failure-results "$failure_results" \
    --targeted-failure-position-radius "${TARGETED_FAILURE_POSITION_RADIUS:-0.75}" \
    --targeted-failure-heading-radius "${TARGETED_FAILURE_HEADING_RADIUS:-0.65}" \
    --train-steps "${SUPERVISED_IQE_TRAIN_STEPS:-10000}" \
    --batch-size "${SUPERVISED_IQE_BATCH_SIZE:-512}" \
    --learning-rate "${SUPERVISED_IQE_LR:-0.0001}" \
    --loss "${SUPERVISED_IQE_LOSS:-huber}" \
    --huber-delta "${SUPERVISED_IQE_HUBER_DELTA:-10}" \
    --oracle-value-cache-dir "${ORACLE_VALUE_CACHE_DIR:-$OUTPUT_ROOT/oracle_value_cache}" \
    --astar-position-resolution "${ASTAR_POSITION_RESOLUTION:-0.25}" \
    --astar-heading-bins "${ASTAR_HEADING_BINS:-24}" \
    --astar-primitive-steps "${ASTAR_PRIMITIVE_STEPS:-5}"

  "$PYTHON_BIN" minimal_qrl/eval/comm_inspection_baseline_eval.py \
    --stage pilot \
    --methods targeted_supervised_iqe_mppi \
    --output-dir "$mppi_dir" \
    --qrl-checkpoints "$checkpoint" \
    --scenario-config "$SCENARIO_CONFIG" \
    --task-bank "$TASK_BANK" \
    --task-split test \
    --task-strata u_trap \
    --seed "${SEED:-0}" \
    --device "${DEVICE:-auto}" \
    --num-critics "${NUM_CRITICS:-2}" \
    --mppi-horizon "${MPPI_HORIZON:-10}" \
    --mppi-num-samples "${MPPI_NUM_SAMPLES:-128}" \
    --mppi-noise-sigma "${MPPI_NOISE_SIGMA:-0.8}" \
    --mppi-temperature "${MPPI_TEMPERATURE:-1.0}" \
    --mppi-terminal-weight "${MPPI_TERMINAL_WEIGHT:-1.0}" \
    ${save_visualizations_flag[@]+"${save_visualizations_flag[@]}"} \
    --viz-max-successes "${VIZ_MAX_SUCCESSES:-12}" \
    --viz-max-failures "${VIZ_MAX_FAILURES:-12}"
}

dense_transition_qrl() {
  local experiment_dir="${DENSE_TRANSITION_QRL_DIR:-$OUTPUT_ROOT/dense_transition_original_qrl}"
  local checkpoint="$experiment_dir/checkpoint_final.pth"
  local local_eval_dir="${DENSE_TRANSITION_LOCAL_EVAL_DIR:-$experiment_dir/u_trap_local_eval}"
  local mppi_dir="${DENSE_TRANSITION_MPPI_DIR:-$experiment_dir/mppi_test_u_trap}"
  local failure_results="${DENSE_TRANSITION_FAILURE_RESULTS:-$OUTPUT_ROOT/supervised_iqe_oracle/mppi_test_u_trap/baseline_results.json}"
  local reuse_oracle_args=()
  if [[ ! -f "$failure_results" ]]; then
    echo "Missing prior Supervised-IQE failure results: $failure_results" >&2
    exit 1
  fi
  if [[ -n "${DENSE_TRANSITION_REUSE_ORACLE_JSON:-}" ]]; then
    reuse_oracle_args+=(--reuse-oracle-json "$DENSE_TRANSITION_REUSE_ORACLE_JSON")
  fi

  "$PYTHON_BIN" minimal_qrl/train.py \
    --scenario-config "$SCENARIO_CONFIG" \
    --output-dir "$experiment_dir" \
    --seed "${DENSE_TRANSITION_SEED:-42}" \
    --device "${DEVICE:-cpu}" \
    --num-episodes "${NUM_EPISODES:-500}" \
    --target-env-transitions "${TARGET_ENV_TRANSITIONS:-120000}" \
    --comm-dataset-mode dense_transition_original \
    --dense-transition-device-id u_trap_target \
    --dense-transition-position-resolution "${DENSE_TRANSITION_POSITION_RESOLUTION:-0.25}" \
    --dense-transition-heading-bins "${DENSE_TRANSITION_HEADING_BINS:-24}" \
    --dense-transition-primitive-steps "${DENSE_TRANSITION_PRIMITIVE_STEPS:-5}" \
    --dense-transition-primitive-scales -1.0 -0.5 0.0 0.5 1.0 \
    --dense-transition-local-fraction "${DENSE_TRANSITION_LOCAL_FRACTION:-0.5}" \
    --dense-transition-failure-results "$failure_results" \
    --dense-transition-failure-position-radius "${DENSE_TRANSITION_FAILURE_POSITION_RADIUS:-0.75}" \
    --dense-transition-failure-heading-radius "${DENSE_TRANSITION_FAILURE_HEADING_RADIUS:-0.65}" \
    --task-aware-teacher-ratio "${DENSE_GLOBAL_TEACHER_RATIO:-1.0}" \
    --batch-size "${BATCH_SIZE:-256}" \
    --total-steps "${TOTAL_STEPS:-120000}" \
    --num-critics "${NUM_CRITICS:-2}" \
    --qrl-cost-source negative_reward \
    --global-push-softplus-offset "${GLOBAL_PUSH_SOFTPLUS_OFFSET:-15.0}" \
    --global-push-softplus-beta "${GLOBAL_PUSH_SOFTPLUS_BETA:-0.1}" \
    --global-push-abstract-goal-ratio "${GLOBAL_PUSH_ABSTRACT_GOAL_RATIO:-0.6}" \
    --global-push-state-goal-ratio "${GLOBAL_PUSH_STATE_GOAL_RATIO:-0.4}" \
    --abstract-goal-edge-loss-weight "${ABSTRACT_GOAL_EDGE_LOSS_WEIGHT:-1.0}" \
    --qrl-temporal-constraint-weight 0.0 \
    --qrl-goal-return-constraint-weight 0.0 \
    --qrl-nstep-goal-constraint-weight 0.0 \
    --qrl-success-transition-weight 1.0 \
    --log-interval "${LOG_INTERVAL:-100}" \
    --save-interval "${SAVE_INTERVAL:-2000}" \
    --eval-interval 0 \
    --visualization-interval 0 \
    --planning-eval-interval 0

  "$PYTHON_BIN" -m minimal_qrl.industry_exp.qrl_oracle_diagnostics \
    --scenario-config "$SCENARIO_CONFIG" \
    --checkpoint "$checkpoint" \
    --output-dir "$experiment_dir/oracle_diagnostics" \
    --device "${DEVICE:-auto}" \
    --num-critics "${NUM_CRITICS:-2}" \
    --seed "${DENSE_ORACLE_EVAL_SEED:-20260823}" \
    --global-eval-samples "${DENSE_ORACLE_EVAL_SAMPLES:-20000}" \
    --astar-position-resolution "${ASTAR_POSITION_RESOLUTION:-0.25}" \
    --astar-heading-bins "${ASTAR_HEADING_BINS:-24}" \
    --astar-primitive-steps "${ASTAR_PRIMITIVE_STEPS:-5}" \
    --oracle-value-cache-dir "${ORACLE_VALUE_CACHE_DIR:-$OUTPUT_ROOT/oracle_value_cache}"

  "$PYTHON_BIN" -m minimal_qrl.eval.u_trap_local_navigability \
    --scenario-config "$SCENARIO_CONFIG" \
    --checkpoints "$checkpoint" \
    --output-dir "$local_eval_dir" \
    --device "${DEVICE:-auto}" \
    --num-critics "${NUM_CRITICS:-2}" \
    --seed "${LOCAL_NAV_SEED:-20260802}" \
    --astar-position-resolution "${ASTAR_POSITION_RESOLUTION:-0.25}" \
    --astar-heading-bins "${ASTAR_HEADING_BINS:-24}" \
    --astar-primitive-steps "${ASTAR_PRIMITIVE_STEPS:-5}" \
    --astar-max-expansions "${ASTAR_MAX_EXPANSIONS:-200000}" \
    --astar-timeout-sec "${ASTAR_TIMEOUT_SEC:-120}" \
    ${reuse_oracle_args[@]+"${reuse_oracle_args[@]}"}

  "$PYTHON_BIN" minimal_qrl/eval/comm_inspection_baseline_eval.py \
    --stage pilot \
    --methods dense_transition_qrl_mppi \
    --output-dir "$mppi_dir" \
    --qrl-checkpoints "$checkpoint" \
    --scenario-config "$SCENARIO_CONFIG" \
    --task-bank "$TASK_BANK" \
    --task-split test \
    --task-strata u_trap \
    --seed "${SEED:-0}" \
    --device "${DEVICE:-auto}" \
    --num-critics "${NUM_CRITICS:-2}" \
    --mppi-horizon "${MPPI_HORIZON:-10}" \
    --mppi-num-samples "${MPPI_NUM_SAMPLES:-128}" \
    --mppi-noise-sigma "${MPPI_NOISE_SIGMA:-0.8}" \
    --mppi-temperature "${MPPI_TEMPERATURE:-1.0}" \
    --mppi-terminal-weight "${MPPI_TERMINAL_WEIGHT:-1.0}"
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
  local_nav_eval)
    local_nav_eval
    ;;
  oracle_mppi)
    oracle_mppi
    ;;
  supervised_iqe)
    supervised_iqe
    ;;
  targeted_supervised_iqe)
    targeted_supervised_iqe
    ;;
  dense_transition_qrl)
    dense_transition_qrl
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
    echo "Unknown PHASE=$PHASE (expected prepare, visualize, train_qrl, eval_qrl, local_nav_eval, oracle_mppi, supervised_iqe, targeted_supervised_iqe, dense_transition_qrl, benchmark, or all)" >&2
    exit 2
    ;;
esac

echo "Diagnostic scenario: $SCENARIO_CONFIG"
echo "Fixed task bank:    $TASK_BANK"
