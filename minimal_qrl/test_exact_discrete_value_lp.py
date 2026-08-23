from __future__ import annotations

import numpy as np

from minimal_qrl.baselines import HybridAStarConfig, HybridAStarValueOracle
from minimal_qrl.dataset import DenseUTrapTransitionConfig
from minimal_qrl.envs import CommInspectionDubinsUAV2D
from minimal_qrl.industry_exp.diagnostic_scenario import build_diagnostic_scenario
from minimal_qrl.industry_exp.exact_discrete_value_lp import (
    DIRECT_GOAL,
    audit_dense_local_graph,
    build_discrete_value_graph,
    reverse_dijkstra,
    solve_exact_value_lp,
    validate_dense_real_dynamics,
)
from minimal_qrl.industry_exp.scalability_scenarios import scenario_to_env_kwargs


FAILED_STARTS = (
    (4.75, 3.15, 3.0),
    (4.75, 4.15, -3.0),
    (4.9, 3.45, 3.1),
    (4.9, 3.9, -3.1),
)


def test_exact_value_lp_recovers_shortest_path_values():
    # 0 -> 1 -> goal costs 2 + 3; the alternative 0 -> 2 -> goal costs 10 + 1.
    terminal = np.array([False, False, False, True])
    sources = np.array([0, 0, 1, 2], dtype=np.int64)
    destinations = np.array([1, 2, 3, DIRECT_GOAL], dtype=np.int64)
    costs = np.array([2.0, 10.0, 3.0, 1.0])
    values, diagnostics = solve_exact_value_lp(
        terminal=terminal,
        sources=sources,
        destinations=destinations,
        costs=costs,
        included=np.ones(4, dtype=bool),
        time_limit_sec=10.0,
    )
    assert diagnostics["success"]
    assert diagnostics["max_constraint_violation"] <= 1e-9
    assert np.allclose(values, np.array([5.0, 3.0, 1.0, 0.0]))


def _small_dense_setup():
    scenario = build_diagnostic_scenario()
    env = CommInspectionDubinsUAV2D(**scenario_to_env_kwargs(scenario))
    env.reset(seed=7, options={"device_id": "u_trap_target"})
    dense = DenseUTrapTransitionConfig(
        position_resolution=0.5,
        heading_bins=12,
        primitive_steps=5,
        primitive_scales=(-1.0, 1.0),
        diagnostic_regions=scenario["metadata"]["exploration_diagnostic_regions"],
        failed_start_states=FAILED_STARTS,
    )
    astar = HybridAStarConfig(
        position_resolution=0.5,
        heading_bins=12,
        primitive_steps=5,
        primitive_scales=(-1.0, 1.0),
    )
    graph = build_discrete_value_graph(env, astar)
    helper = HybridAStarValueOracle(astar)
    _states, helper._grid_shape = helper._grid(env)
    return env, dense, graph, helper


def test_current_dense_local_seed_graph_has_no_terminal_and_is_not_closed():
    env, dense, graph, helper = _small_dense_setup()
    _indices, _strata, audit = audit_dense_local_graph(
        graph, helper, env, dense
    )
    assert audit["seed_lattice_states"] > 0
    assert audit["terminal_goal_states"] == 0
    assert audit["valid_edges_leaving_local_nodes"] > 0
    assert audit["lp_well_posed"] is False
    assert audit["lp_status"] == "not_solved_unbounded_no_terminal_set"


def test_vectorized_lattice_edges_match_real_dense_environment_rollouts():
    env, dense, graph, helper = _small_dense_setup()
    local_indices, _strata, _audit = audit_dense_local_graph(
        graph, helper, env, dense
    )
    metrics = validate_dense_real_dynamics(
        graph,
        helper,
        env,
        dense,
        local_indices,
        seed=11,
    )
    assert metrics["primitive_rollouts_checked"] > 0
    assert metrics["validity_match_rate"] > 0.95
    assert metrics["destination_match_rate_on_valid"] == 1.0
    assert metrics["cost_max_abs_error_on_valid_primitives"] <= 1e-4


def test_reverse_dijkstra_marks_terminal_reachable_subset():
    env, _dense, graph, _helper = _small_dense_setup()
    values = reverse_dijkstra(graph)
    assert np.all(values[graph.terminal] == 0.0)
    assert np.any(np.isfinite(values) & graph.valid)
    assert np.all(np.isfinite(values[graph.terminal]))
    assert env.active_device_id == "u_trap_target"
