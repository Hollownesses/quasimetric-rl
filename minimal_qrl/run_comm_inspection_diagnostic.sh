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
#   PHASE=train_qrl_nstep_upper_bound DEVICE=mps \
#     OUTPUT_ROOT=./results/diagnostic_u_shadow_corridors_topology_v2 \
#     bash minimal_qrl/run_comm_inspection_diagnostic.sh
#   PHASE=train_qrl_mqe_waypoint_consistency DEVICE=mps \
#     OUTPUT_ROOT=./results/diagnostic_u_shadow_corridors_topology_v2 \
#     bash minimal_qrl/run_comm_inspection_diagnostic.sh
#   PHASE=train_qrl_mqe_separate_anchor_stop_loss DEVICE=mps \
#     OUTPUT_ROOT=./results/diagnostic_u_shadow_corridors_topology_v2 \
#     bash minimal_qrl/run_comm_inspection_diagnostic.sh
#   PHASE=train_qrl_kkt_functional DEVICE=mps \
#     OUTPUT_ROOT=./results/diagnostic_u_shadow_corridors_topology_v2 \
#     bash minimal_qrl/run_comm_inspection_diagnostic.sh
#   PHASE=eval_qrl QRL_CHECKPOINT=... bash minimal_qrl/run_comm_inspection_diagnostic.sh
#   PHASE=local_nav_eval QRL_CHECKPOINTS="checkpoint_a.pth checkpoint_b.pth" \
#     bash minimal_qrl/run_comm_inspection_diagnostic.sh
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
if [[ -d "./results/shared_oracle_banks/chemical_process" ]]; then
  DEFAULT_SHARED_ORACLE_DIR="./results/shared_oracle_banks/chemical_process"
elif [[ -d "../quasimetric-rl-industrial-inspection/results/shared_oracle_banks/chemical_process" ]]; then
  DEFAULT_SHARED_ORACLE_DIR="../quasimetric-rl-industrial-inspection/results/shared_oracle_banks/chemical_process"
else
  DEFAULT_SHARED_ORACLE_DIR="./results/shared_oracle_banks/chemical_process"
fi
SHARED_ORACLE_DIR="${SHARED_ORACLE_DIR:-$DEFAULT_SHARED_ORACLE_DIR}"
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
    --qrl-local-constraint-mode "${QRL_LOCAL_CONSTRAINT_MODE:-legacy_squared_hinge}" \
    --qrl-kkt-augmented-lagrangian-rho "${QRL_KKT_AUGMENTED_LAGRANGIAN_RHO:-1.0}" \
    --qrl-kkt-dual-hidden-sizes ${QRL_KKT_DUAL_HIDDEN_SIZES:-128 128} \
    --qrl-kkt-dual-max "${QRL_KKT_DUAL_MAX:-100000.0}" \
    --qrl-kkt-dual-steps "${QRL_KKT_DUAL_STEPS:-1}" \
    --qrl-kkt-dual-start-violation-fraction "${QRL_KKT_DUAL_START_VIOLATION_FRACTION:-0.05}" \
    --qrl-kkt-dual-projected-step-size "${QRL_KKT_DUAL_PROJECTED_STEP_SIZE:-0.1}" \
    --qrl-kkt-dual-feature-scale "${QRL_KKT_DUAL_FEATURE_SCALE:-5.0}" \
    --qrl-kkt-dual-raw-min "${QRL_KKT_DUAL_RAW_MIN:--10.0}" \
    --qrl-kkt-init-lagrange-multiplier "${QRL_KKT_INIT_LAGRANGE_MULTIPLIER:-0.01}" \
    --qrl-kkt-dual-lr "${QRL_KKT_DUAL_LR:-0.0001}" \
    --qrl-kkt-fail-violation-fraction "${QRL_KKT_FAIL_VIOLATION_FRACTION:-0.05}" \
    --qrl-kkt-fail-lagrange-max "${QRL_KKT_FAIL_LAGRANGE_MAX:-1e-8}" \
    --global-push-objective "${GLOBAL_PUSH_OBJECTIVE:-softplus}" \
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
    --qrl-mqe-waypoint-consistency-weight "${QRL_MQE_WAYPOINT_CONSISTENCY_WEIGHT:-0.0}" \
    --qrl-mqe-goal-discount "${QRL_MQE_GOAL_DISCOUNT:-0.995}" \
    --qrl-mqe-waypoint-lambda "${QRL_MQE_WAYPOINT_LAMBDA:-0.95}" \
    --qrl-mqe-next-state-probability "${QRL_MQE_NEXT_STATE_PROBABILITY:-0.2}" \
    --qrl-mqe-terminal-anchor-fraction "${QRL_MQE_TERMINAL_ANCHOR_FRACTION:-0.1}" \
    --qrl-mqe-target-tau "${QRL_MQE_TARGET_TAU:-0.005}" \
    --qrl-mqe-huber-delta "${QRL_MQE_HUBER_DELTA:-1.0}" \
    --qrl-mqe-family-normalization "${QRL_MQE_FAMILY_NORMALIZATION:-mixed}" \
    --qrl-mqe-terminal-anchor-loss-weight "${QRL_MQE_TERMINAL_ANCHOR_LOSS_WEIGHT:-1.0}" \
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

# One-sided n-step sanity check.  It reuses train_qrl and changes only the
# optional task-goal upper-bound weight and output directory.
train_qrl_nstep_upper_bound() {
  local nstep_train_dir="${NSTEP_TRAIN_DIR:-$OUTPUT_ROOT/qrl_training_nstep_upper_bound}"

  echo "QRL topology-improvement ablation:"
  echo "  variant=one_sided_nstep_upper_bound"
  echo "  global_push_objective=softplus"
  echo "  dataset_mode=${QRL_DATASET_MODE:-qrl_explore}"
  echo "  nstep_weight=${QRL_NSTEP_GOAL_CONSTRAINT_WEIGHT:-1.0}"
  echo "  target_tau=${QRL_NSTEP_TARGET_TAU:-0.005}"
  echo "  output_dir=$nstep_train_dir"

  QRL_DATASET_MODE="${QRL_DATASET_MODE:-qrl_explore}" \
  QRL_NSTEP_GOAL_CONSTRAINT_WEIGHT="${QRL_NSTEP_GOAL_CONSTRAINT_WEIGHT:-1.0}" \
  QRL_MQE_WAYPOINT_CONSISTENCY_WEIGHT=0.0 \
  TRAIN_DIR="$nstep_train_dir" \
    train_qrl
}

# KKT-aligned functional dual with violation-triggered wake-up, lambda-space
# projected-target fitting, straight-through gradients, and a raw lower bound.
# Optional temporal/MQE additions stay disabled to isolate the optimizer change.
train_qrl_kkt_functional() {
  local kkt_train_dir="${KKT_TRAIN_DIR:-$OUTPUT_ROOT/qrl_training_kkt_functional}"

  echo "QRL topology-improvement ablation:"
  echo "  variant=kkt_functional_lambda_space_projected_dual_v4"
  echo "  global_push_objective=linear"
  echo "  dataset_mode=${QRL_DATASET_MODE:-qrl_explore}"
  echo "  augmented_rho=${QRL_KKT_AUGMENTED_LAGRANGIAN_RHO:-1.0}"
  echo "  dual_steps=${QRL_KKT_DUAL_STEPS:-1}"
  echo "  dual_lr=${QRL_KKT_DUAL_LR:-0.0001}"
  echo "  dual_start_violation_fraction=${QRL_KKT_DUAL_START_VIOLATION_FRACTION:-0.05}"
  echo "  dual_projected_step_size=${QRL_KKT_DUAL_PROJECTED_STEP_SIZE:-0.1}"
  echo "  dual_fit=lambda_space_mse_straight_through"
  echo "  output_dir=$kkt_train_dir"

  QRL_DATASET_MODE="${QRL_DATASET_MODE:-qrl_explore}" \
  QRL_LOCAL_CONSTRAINT_MODE=kkt_functional \
  GLOBAL_PUSH_OBJECTIVE=linear \
  QRL_TEMPORAL_CONSTRAINT_WEIGHT=0.0 \
  QRL_GOAL_RETURN_CONSTRAINT_WEIGHT=0.0 \
  QRL_NSTEP_GOAL_CONSTRAINT_WEIGHT=0.0 \
  QRL_MQE_WAYPOINT_CONSISTENCY_WEIGHT=0.0 \
  TRAIN_DIR="$kkt_train_dir" \
    train_qrl
}

# MQE-inspired two-sided consistency.  A fixed fraction of MQE-only slots is
# drawn from complete natural successes, so abstract-goal anchors do not depend
# on their prevalence in the main replay batch.
train_qrl_mqe_waypoint_consistency() {
  local mqe_train_dir="${MQE_WAYPOINT_TRAIN_DIR:-$OUTPUT_ROOT/qrl_training_mqe_waypoint_consistency}"

  echo "QRL topology-improvement ablation:"
  echo "  variant=mqe_inspired_two_sided_waypoint_consistency"
  echo "  global_push_objective=softplus"
  echo "  dataset_mode=${QRL_DATASET_MODE:-qrl_explore}"
  echo "  waypoint_weight=${QRL_MQE_WAYPOINT_CONSISTENCY_WEIGHT:-1.0}"
  echo "  goal_discount=${QRL_MQE_GOAL_DISCOUNT:-0.995}"
  echo "  waypoint_lambda=${QRL_MQE_WAYPOINT_LAMBDA:-0.95}"
  echo "  next_state_probability=${QRL_MQE_NEXT_STATE_PROBABILITY:-0.2}"
  echo "  terminal_anchor_fraction=${QRL_MQE_TERMINAL_ANCHOR_FRACTION:-0.1}"
  echo "  target_tau=${QRL_MQE_TARGET_TAU:-0.005}"
  echo "  huber_delta=${QRL_MQE_HUBER_DELTA:-1.0}"
  echo "  output_dir=$mqe_train_dir"

  QRL_DATASET_MODE="${QRL_DATASET_MODE:-qrl_explore}" \
  QRL_NSTEP_GOAL_CONSTRAINT_WEIGHT=0.0 \
  QRL_MQE_WAYPOINT_CONSISTENCY_WEIGHT="${QRL_MQE_WAYPOINT_CONSISTENCY_WEIGHT:-1.0}" \
  QRL_MQE_FAMILY_NORMALIZATION=mixed \
  TRAIN_DIR="$mqe_train_dir" \
    train_qrl
}

# MQE-v2.1 stop-loss experiment.  Sampling and every other QRL loss remain
# unchanged; only physical and terminal-anchor Huber families are normalized
# independently, while preserving the previous physical-family coefficient.
train_qrl_mqe_separate_anchor_stop_loss() {
  local stop_loss_train_dir="${MQE_STOP_LOSS_TRAIN_DIR:-$OUTPUT_ROOT/qrl_training_mqe_separate_anchor_stop_loss}"

  echo "QRL topology-improvement stop-loss ablation:"
  echo "  variant=mqe_separately_normalized_terminal_anchor_stop_loss"
  echo "  global_push_objective=softplus"
  echo "  dataset_mode=${QRL_DATASET_MODE:-qrl_explore}"
  echo "  waypoint_weight=${QRL_MQE_WAYPOINT_CONSISTENCY_WEIGHT:-1.0}"
  echo "  family_normalization=separate"
  echo "  terminal_anchor_loss_weight=${QRL_MQE_TERMINAL_ANCHOR_LOSS_WEIGHT:-1.0}"
  echo "  terminal_anchor_fraction=${QRL_MQE_TERMINAL_ANCHOR_FRACTION:-0.1}"
  echo "  output_dir=$stop_loss_train_dir"

  QRL_DATASET_MODE="${QRL_DATASET_MODE:-qrl_explore}" \
  QRL_NSTEP_GOAL_CONSTRAINT_WEIGHT=0.0 \
  QRL_MQE_WAYPOINT_CONSISTENCY_WEIGHT="${QRL_MQE_WAYPOINT_CONSISTENCY_WEIGHT:-1.0}" \
  QRL_MQE_FAMILY_NORMALIZATION=separate \
  QRL_MQE_TERMINAL_ANCHOR_LOSS_WEIGHT="${QRL_MQE_TERMINAL_ANCHOR_LOSS_WEIGHT:-1.0}" \
  TRAIN_DIR="$stop_loss_train_dir" \
    train_qrl
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

case "$PHASE" in
  prepare)
    ;;
  visualize)
    visualize
    ;;
  train_qrl)
    train_qrl
    ;;
  train_qrl_nstep_upper_bound)
    train_qrl_nstep_upper_bound
    ;;
  train_qrl_kkt_functional)
    train_qrl_kkt_functional
    ;;
  train_qrl_mqe_waypoint_consistency)
    train_qrl_mqe_waypoint_consistency
    ;;
  train_qrl_mqe_separate_anchor_stop_loss)
    train_qrl_mqe_separate_anchor_stop_loss
    ;;
  eval_qrl)
    eval_qrl
    ;;
  local_nav_eval)
    local_nav_eval
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
    echo "Unknown PHASE=$PHASE (expected prepare, visualize, train_qrl, train_qrl_nstep_upper_bound, train_qrl_kkt_functional, train_qrl_mqe_waypoint_consistency, train_qrl_mqe_separate_anchor_stop_loss, eval_qrl, local_nav_eval, benchmark, or all)" >&2
    exit 2
    ;;
esac

echo "Diagnostic scenario: $SCENARIO_CONFIG"
echo "Fixed task bank:    $TASK_BANK"
