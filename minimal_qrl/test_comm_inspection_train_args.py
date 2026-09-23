from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import minimal_qrl.train as train_module


def test_comm_global_push_cli_values_reach_loss_config(monkeypatch):
    captured = {}
    monkeypatch.setattr(train_module, "train", lambda args: captured.setdefault("args", args))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "minimal_qrl/train.py",
            "--env-type",
            "comm_inspection_dubins_uav",
            "--global-push-softplus-offset",
            "300",
            "--global-push-softplus-beta",
            "0.01",
            "--global-push-abstract-goal-ratio",
            "0.5",
            "--global-push-state-goal-ratio",
            "0.5",
            "--global-push-objective",
            "linear",
        ],
    )

    train_module.main()
    args = captured["args"]
    config = train_module._comm_inspection_global_push_conf(args)

    assert np.isclose(config.softplus_offset, 300.0)
    assert np.isclose(config.softplus_beta, 0.01)
    assert np.isclose(config.abstract_goal_ratio, 0.5)
    assert np.isclose(config.state_goal_ratio, 0.5)
    assert config.objective == "linear"


def test_comm_training_shell_forwards_global_push_environment_variables():
    script = (
        Path(__file__).with_name("run_comm_inspection_train.sh")
        .read_text(encoding="utf-8")
    )

    assert (
        '--global-push-softplus-offset '
        '"${GLOBAL_PUSH_SOFTPLUS_OFFSET:-15.0}"'
    ) in script
    assert (
        '--global-push-softplus-beta '
        '"${GLOBAL_PUSH_SOFTPLUS_BETA:-0.1}"'
    ) in script
    assert '--global-push-objective "${GLOBAL_PUSH_OBJECTIVE:-softplus}"' in script


def test_qrl_explore_cli_defaults_to_fixed_200k_attempted_steps(monkeypatch):
    captured = {}
    monkeypatch.setattr(train_module, "train", lambda args: captured.setdefault("args", args))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "minimal_qrl/train.py",
            "--env-type",
            "comm_inspection_dubins_uav",
            "--comm-dataset-mode",
            "qrl_explore",
        ],
    )

    train_module.main()
    args = captured["args"]
    assert args.comm_dataset_mode == "qrl_explore"
    assert args.explore_attempted_env_steps == 200_000
    assert args.explore_start_heading_bins == 12
    assert args.explore_action_hold_min_steps == 3
    assert args.explore_action_hold_max_steps == 10
    assert np.isclose(args.explore_straight_action_probability, 0.5)
    assert np.isclose(args.explore_start_boundary_margin, 0.5)
    assert args.explore_local_safety_lookahead_steps == 10
    assert np.isclose(args.qrl_temporal_constraint_weight, 1.0)
    assert args.qrl_temporal_min_future_steps == 2
    assert np.isclose(args.qrl_goal_return_constraint_weight, 1.0)
    assert np.isclose(args.qrl_nstep_goal_constraint_weight, 0.0)
    assert np.isclose(args.qrl_mqe_waypoint_consistency_weight, 0.0)
    assert np.isclose(args.qrl_mqe_terminal_anchor_fraction, 0.1)
    assert args.qrl_mqe_family_normalization == "mixed"
    assert np.isclose(args.qrl_mqe_terminal_anchor_loss_weight, 1.0)
    assert np.isclose(args.qrl_success_transition_weight, 4.0)
    assert args.qrl_local_constraint_mode == "legacy_squared_hinge"
    assert args.global_push_objective == "softplus"
    assert np.isclose(args.qrl_kkt_augmented_lagrangian_rho, 1.0)
    assert args.qrl_kkt_dual_hidden_sizes == [128, 128]
    assert np.isclose(args.qrl_kkt_dual_max, 100000.0)
    assert args.qrl_kkt_dual_steps == 1
    assert np.isclose(args.qrl_kkt_dual_active_margin, 1.0)
    assert np.isclose(args.qrl_kkt_dual_start_violation_fraction, 0.05)
    assert np.isclose(args.qrl_kkt_dual_slack_weight, 0.1)
    assert np.isclose(args.qrl_kkt_dual_feature_scale, 5.0)
    assert np.isclose(args.qrl_kkt_dual_raw_min, -10.0)
    assert np.isclose(args.qrl_kkt_init_lagrange_multiplier, 0.01)
    assert np.isclose(args.qrl_kkt_dual_lr, 1e-4)
    assert np.isclose(args.qrl_kkt_fail_violation_fraction, 0.05)
    assert np.isclose(args.qrl_kkt_fail_lagrange_max, 1e-8)


def test_diagnostic_shell_exposes_qrl_explore_without_changing_standard_budget():
    script = (
        Path(__file__).with_name("run_comm_inspection_diagnostic.sh")
        .read_text(encoding="utf-8")
    )

    assert 'qrl_dataset_mode="${QRL_DATASET_MODE:-standard}"' in script
    assert '--explore-attempted-env-steps "${EXPLORE_ATTEMPTED_ENV_STEPS:-200000}"' in script
    assert '--target-env-transitions "${TARGET_ENV_TRANSITIONS:-120000}"' in script
    assert 'qrl_explore_v2' not in script
    assert '--explore-action-hold-min-steps "${EXPLORE_ACTION_HOLD_MIN_STEPS:-3}"' in script
    assert '--explore-start-boundary-margin "${EXPLORE_START_BOUNDARY_MARGIN:-0.5}"' in script
    assert '--qrl-temporal-constraint-weight "${QRL_TEMPORAL_CONSTRAINT_WEIGHT:-1.0}"' in script
    assert '--qrl-goal-return-constraint-weight "${QRL_GOAL_RETURN_CONSTRAINT_WEIGHT:-1.0}"' in script
    assert '--qrl-nstep-goal-constraint-weight "${QRL_NSTEP_GOAL_CONSTRAINT_WEIGHT:-0.0}"' in script
    assert '--qrl-mqe-waypoint-consistency-weight "${QRL_MQE_WAYPOINT_CONSISTENCY_WEIGHT:-0.0}"' in script
    assert '--qrl-mqe-terminal-anchor-fraction "${QRL_MQE_TERMINAL_ANCHOR_FRACTION:-0.1}"' in script
    assert '--qrl-mqe-family-normalization "${QRL_MQE_FAMILY_NORMALIZATION:-mixed}"' in script
    assert '--qrl-mqe-terminal-anchor-loss-weight "${QRL_MQE_TERMINAL_ANCHOR_LOSS_WEIGHT:-1.0}"' in script
    assert '--qrl-success-transition-weight "${QRL_SUCCESS_TRANSITION_WEIGHT:-4.0}"' in script
    assert '--qrl-local-constraint-mode "${QRL_LOCAL_CONSTRAINT_MODE:-legacy_squared_hinge}"' in script
    assert '--qrl-kkt-dual-steps "${QRL_KKT_DUAL_STEPS:-1}"' in script
    assert '--qrl-kkt-dual-active-margin "${QRL_KKT_DUAL_ACTIVE_MARGIN:-1.0}"' in script
    assert (
        '--qrl-kkt-dual-start-violation-fraction '
        '"${QRL_KKT_DUAL_START_VIOLATION_FRACTION:-0.05}"'
    ) in script
    assert '--qrl-kkt-dual-lr "${QRL_KKT_DUAL_LR:-0.0001}"' in script
    assert '../quasimetric-rl-industrial-inspection/results/shared_oracle_banks/chemical_process' in script
    assert 'teacher_ratio="0.0"' in script
    assert '--task-aware-teacher-ratio "$teacher_ratio"' in script
    assert 'local_nav_eval()' in script
    assert 'LOCAL_NAV_REUSE_ORACLE_JSON' in script
    assert '--reuse-oracle-json "$LOCAL_NAV_REUSE_ORACLE_JSON"' in script


def test_diagnostic_shell_has_isolated_kkt_functional_ablation():
    script = (
        Path(__file__).with_name("run_comm_inspection_diagnostic.sh")
        .read_text(encoding="utf-8")
    )

    assert "train_qrl_kkt_functional()" in script
    assert "variant=kkt_functional_primal_dual_v2" in script
    assert "GLOBAL_PUSH_OBJECTIVE=linear" in script
    assert "QRL_LOCAL_CONSTRAINT_MODE=kkt_functional" in script
    assert "QRL_TEMPORAL_CONSTRAINT_WEIGHT=0.0" in script
    assert "QRL_GOAL_RETURN_CONSTRAINT_WEIGHT=0.0" in script
    assert "QRL_MQE_WAYPOINT_CONSISTENCY_WEIGHT=0.0" in script
    assert "train_qrl_kkt_functional)" in script
    assert '--astar-heuristic-weight "${LOCAL_NAV_ASTAR_HEURISTIC_WEIGHT:-1.0}"' in script


def test_diagnostic_shell_has_isolated_nstep_upper_bound_ablation():
    script = (
        Path(__file__).with_name("run_comm_inspection_diagnostic.sh")
        .read_text(encoding="utf-8")
    )

    assert "train_qrl_nstep_upper_bound()" in script
    assert (
        'local nstep_train_dir="${NSTEP_TRAIN_DIR:-$OUTPUT_ROOT/'
        'qrl_training_nstep_upper_bound}"'
    ) in script
    assert 'QRL_DATASET_MODE="${QRL_DATASET_MODE:-qrl_explore}"' in script
    assert (
        'QRL_NSTEP_GOAL_CONSTRAINT_WEIGHT="${QRL_NSTEP_GOAL_CONSTRAINT_WEIGHT:-1.0}"'
        in script
    )
    assert "variant=one_sided_nstep_upper_bound" in script
    assert "train_qrl_nstep_upper_bound)" in script


def test_diagnostic_shell_has_isolated_mqe_waypoint_ablation():
    script = (
        Path(__file__).with_name("run_comm_inspection_diagnostic.sh")
        .read_text(encoding="utf-8")
    )

    assert "train_qrl_mqe_waypoint_consistency()" in script
    assert "variant=mqe_inspired_two_sided_waypoint_consistency" in script
    assert "global_push_objective=softplus" in script
    assert "terminal_anchor_fraction=${QRL_MQE_TERMINAL_ANCHOR_FRACTION:-0.1}" in script
    assert "train_qrl_mqe_waypoint_consistency)" in script


def test_diagnostic_shell_has_separate_anchor_stop_loss_ablation():
    script = (
        Path(__file__).with_name("run_comm_inspection_diagnostic.sh")
        .read_text(encoding="utf-8")
    )

    assert "train_qrl_mqe_separate_anchor_stop_loss()" in script
    assert "variant=mqe_separately_normalized_terminal_anchor_stop_loss" in script
    assert "QRL_MQE_FAMILY_NORMALIZATION=separate" in script
    assert (
        'QRL_MQE_TERMINAL_ANCHOR_LOSS_WEIGHT="${QRL_MQE_TERMINAL_ANCHOR_LOSS_WEIGHT:-1.0}"'
        in script
    )
    assert (
        'qrl_training_mqe_separate_anchor_stop_loss}'
        in script
    )
    assert "train_qrl_mqe_separate_anchor_stop_loss)" in script
