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
