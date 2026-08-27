from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import torch

from minimal_qrl.baselines import HybridAStarConfig
from minimal_qrl.envs import CommInspectionDubinsUAV2D
from minimal_qrl.industry_exp.diagnostic_scenario import build_diagnostic_scenario
from minimal_qrl.industry_exp.scalability_scenarios import scenario_to_env_kwargs
from minimal_qrl.industry_exp.tabular_potential_qrl import (
    CELL_SPECS,
    TabularPotentialProblem,
    TabularTrainConfig,
    build_tabular_problem,
    evaluate_values,
    potential_distances,
    train_tabular_potential,
)


def _toy_problem() -> TabularPotentialProblem:
    # 0 -> 1 -> terminal costs 2 + 3; 0 -> 2 -> G costs 10 + 1.
    return TabularPotentialProblem(
        global_state_indices=np.arange(4, dtype=np.int64),
        states=np.zeros((4, 3), dtype=np.float32),
        sources=np.asarray([0, 0, 1, 2, 3], dtype=np.int64),
        destinations=np.asarray([1, 2, 3, -1, -1], dtype=np.int64),
        costs=np.asarray([2.0, 10.0, 3.0, 1.0, 0.0], dtype=np.float32),
        families=np.asarray([0, 0, 0, 1, 2], dtype=np.int8),
        terminal_indices=np.asarray([3], dtype=np.int64),
        push_indices=np.asarray([0, 1, 2], dtype=np.int64),
        graph_stats={
            "family_counts": {
                "ordinary": 3,
                "direct_goal": 1,
                "terminal_goal": 1,
            }
        },
    )


def test_potential_distance_is_asymmetric_and_obeys_triangle_inequality():
    values = torch.tensor([5.0, 3.0, 1.0, 0.0])
    sources = torch.tensor([0, 1, 0, 1, 3])
    destinations = torch.tensor([1, 3, 3, 0, -1])
    distances = potential_distances(values, sources, destinations)
    assert torch.allclose(distances, torch.tensor([2.0, 3.0, 5.0, 0.0, 0.0]))
    assert distances[2] <= distances[0] + distances[1]


def test_trainer_api_cannot_receive_dijkstra_or_oracle_labels():
    parameters = inspect.signature(train_tabular_potential).parameters
    assert "reference_values" not in parameters
    assert "oracle_values" not in parameters
    assert set(CELL_SPECS) == {
        "full_zero",
        "minibatch_zero",
        "full_practical",
        "minibatch_practical",
    }


def test_full_zero_tabular_optimizer_recovers_toy_shortest_path():
    problem = _toy_problem()
    reference = np.asarray([5.0, 3.0, 1.0, 0.0])
    result = train_tabular_potential(
        problem,
        TabularTrainConfig(
            sampling_mode="full",
            epsilon_mode="zero",
            total_steps=5_000,
            ordinary_batch_size=3,
            primal_lr=0.02,
            dual_lr=0.02,
            eval_interval=0,
        ),
        seed=3,
        device=torch.device("cpu"),
    )
    metrics = evaluate_values(
        problem,
        result.values,
        reference,
        {"ordinary": 0.0, "direct_goal": 0.0, "terminal_goal": 0.0},
    )
    assert metrics["dijkstra_regression"]["max_abs_error"] <= 0.15
    assert metrics["constraint_families"]["ordinary"]["max_excess"] <= 0.1
    assert metrics["constraint_families"]["direct_goal"]["max_excess"] <= 0.1
    assert metrics["terminal_max_abs"] == 0.0


def test_small_problem_matches_full_graph_provenance_and_counts():
    scenario = build_diagnostic_scenario()
    env = CommInspectionDubinsUAV2D(**scenario_to_env_kwargs(scenario))
    env.reset(seed=17, options={"device_id": "u_trap_target"})
    problem, _graph = build_tabular_problem(
        env,
        HybridAStarConfig(
            position_resolution=0.5,
            heading_bins=12,
            primitive_steps=5,
        ),
    )
    counts = problem.graph_stats["family_counts"]
    assert problem.num_edges == sum(counts.values())
    assert problem.num_states == problem.graph_stats["goal_reachable_states"]
    assert len(problem.push_indices) == problem.graph_stats[
        "goal_reachable_nonterminal_states"
    ]
    assert counts["terminal_goal"] == len(problem.terminal_indices)
    assert counts["ordinary"] > counts["direct_goal"] > 0
    assert problem.graph_stats["topology_only_reachability"] is True


def test_diagnostic_shell_exposes_tabular_potential_phase():
    script = Path(__file__).with_name(
        "run_comm_inspection_diagnostic.sh"
    ).read_text(encoding="utf-8")
    assert "tabular_potential_qrl()" in script
    assert "tabular_potential_qrl)" in script
    assert "minimal_qrl.industry_exp.tabular_potential_qrl" in script
