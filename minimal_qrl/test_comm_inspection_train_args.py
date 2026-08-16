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
        ],
    )

    train_module.main()
    args = captured["args"]
    config = train_module._comm_inspection_global_push_conf(args)

    assert np.isclose(config.softplus_offset, 300.0)
    assert np.isclose(config.softplus_beta, 0.01)
    assert np.isclose(config.abstract_goal_ratio, 0.5)
    assert np.isclose(config.state_goal_ratio, 0.5)


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
    assert np.isclose(args.qrl_success_transition_weight, 4.0)


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
    assert '--qrl-success-transition-weight "${QRL_SUCCESS_TRANSITION_WEIGHT:-4.0}"' in script
    assert 'teacher_ratio="0.0"' in script
    assert '--task-aware-teacher-ratio "$teacher_ratio"' in script
    assert 'local_nav_eval()' in script
    assert 'LOCAL_NAV_REUSE_ORACLE_JSON' in script
    assert '--reuse-oracle-json "$LOCAL_NAV_REUSE_ORACLE_JSON"' in script
    assert '--astar-heuristic-weight "${LOCAL_NAV_ASTAR_HEURISTIC_WEIGHT:-1.0}"' in script
