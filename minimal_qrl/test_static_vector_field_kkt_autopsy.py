from __future__ import annotations

from pathlib import Path

import numpy as np

from minimal_qrl.industry_exp.joint_feasible_iqe import JointFeasibleProblem
from minimal_qrl.industry_exp.static_vector_field_kkt_autopsy import (
    compact_reverse_dijkstra,
    family_constraint_gradient,
    full_surrogate_gradients,
    solve_exact_lp_kkt,
)


ROOT = Path(__file__).resolve().parents[1]


def _chain_problem() -> JointFeasibleProblem:
    return JointFeasibleProblem(
        states=np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
        observations=np.zeros((3, 4), dtype=np.float32),
        goal_observation=np.zeros(4, dtype=np.float32),
        reference_values=np.asarray([2.0, 1.0, 0.0], dtype=np.float32),
        sources=np.asarray([0, 0, 1, 2], dtype=np.int64),
        destinations=np.asarray([1, -1, -1, -1], dtype=np.int64),
        costs=np.asarray([1.0, 3.0, 1.0, 0.0], dtype=np.float32),
        families=np.asarray([0, 1, 1, 2], dtype=np.int8),
        terminal_indices=np.asarray([2], dtype=np.int64),
        u_trap_mask=np.asarray([True, False, False]),
        graph_stats={},
    )


def test_exact_lp_edge_duals_close_the_kkt_system():
    problem = _chain_problem()
    assert np.allclose(compact_reverse_dijkstra(problem), problem.reference_values)
    values, duals, audit = solve_exact_lp_kkt(problem, time_limit_sec=10.0)
    assert np.allclose(values, problem.reference_values)
    assert np.all(duals >= -1e-12)
    assert audit["primal"]["max_edge_violation"] <= 1e-10
    assert audit["complementarity"]["edge_linf"] <= 1e-10
    assert audit["stationarity"]["linf"] <= 1e-10


def test_squared_hinge_is_flat_but_linear_active_subgradient_is_not():
    problem = _chain_problem()
    values = problem.reference_values.astype(np.float64)
    ordinary = np.asarray([0], dtype=np.int64)
    squared, squared_audit = family_constraint_gradient(
        problem,
        values,
        ordinary,
        surrogate="squared_hinge",
        active_tolerance=1e-6,
    )
    linear, linear_audit = family_constraint_gradient(
        problem,
        values,
        ordinary,
        surrogate="linear_hinge_right_subgradient",
        active_tolerance=1e-6,
    )
    assert np.allclose(squared, 0.0)
    assert squared_audit["strict_positive_excess_edges"] == 0
    assert np.allclose(linear, [1.0, -1.0, 0.0])
    assert linear_audit["active_subgradient_edges"] == 1


def test_best_family_scalars_can_recover_chain_stationarity_for_linear_hinge():
    problem = _chain_problem()
    result = full_surrogate_gradients(
        problem,
        problem.reference_values.astype(np.float64),
        surrogate="linear_hinge_right_subgradient",
        lambdas={"ordinary": 1.0, "direct_goal": 1.0, "terminal_goal": 1.0},
        active_tolerance=1e-6,
    )
    assert result["fitted_residual_norm"] <= 1e-10
    assert np.linalg.norm(result["fitted_total"]) <= 1e-10


def test_shell_exposes_static_autopsy_phase():
    script = (ROOT / "minimal_qrl/run_comm_inspection_diagnostic.sh").read_text()
    assert "static_vector_field_kkt_autopsy()" in script
    assert "static_vector_field_kkt_autopsy)" in script
    assert "STATIC_AUTOPSY_MINIBATCH_SAMPLES" in script
