from __future__ import annotations

import numpy as np

from minimal_qrl.industry_exp.full_graph_checkpoint_audit import (
    summarize_constraint_family,
)


def test_constraint_family_audit_reports_strict_and_qrl_epsilon_violations():
    result = summarize_constraint_family(
        np.asarray([1.0, 1.2, 1.6, 0.5]),
        np.asarray([1.0, 1.0, 1.0, 1.0]),
        epsilon=0.25,
        numerical_tolerance=1e-6,
    )
    assert result["count"] == 4
    assert result["violation_count"] == 2
    assert result["violation_fraction"] == 0.5
    assert result["epsilon_violation_count"] == 1
    assert result["epsilon_violation_fraction"] == 0.25
    assert np.isclose(result["positive_excess_mean"], 0.2)
    assert np.isclose(result["positive_excess_max"], 0.6)
    assert np.isclose(result["squared_excess_mean"], 0.1)
    assert np.isclose(result["dual_residual_squared"], 0.1 - 0.25**2)
    assert not result["all_constraints_satisfied"]


def test_constraint_family_audit_accepts_numerical_slack():
    result = summarize_constraint_family(
        np.asarray([1.0 + 1e-8, 0.8]),
        np.asarray([1.0, 1.0]),
        epsilon=0.0,
        numerical_tolerance=1e-6,
    )
    assert result["violation_count"] == 0
    assert result["all_constraints_satisfied"]


def test_diagnostic_shell_exposes_targeted_supervised_full_graph_audit():
    script = open("minimal_qrl/run_comm_inspection_diagnostic.sh", encoding="utf-8").read()
    assert "targeted_supervised_full_graph_audit)" in script
    assert "full_graph_checkpoint_audit" in script
    assert "--expected-transitions 352243" in script


def test_diagnostic_shell_exposes_supervised_to_qrl_warm_start():
    script = open("minimal_qrl/run_comm_inspection_diagnostic.sh", encoding="utf-8").read()
    assert "targeted_supervised_qrl_warm_start)" in script
    assert "--init-agent-only" in script
    assert "--full-graph-checkpoint-audit" in script
    assert "WARM_START_TOTAL_STEPS:-20000" in script
    assert "WARM_START_SAVE_INTERVAL:-2000" in script
    assert '--qrl-checkpoints "${checkpoints[@]}"' in script
