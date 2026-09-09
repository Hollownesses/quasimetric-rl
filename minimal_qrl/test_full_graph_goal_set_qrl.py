from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from minimal_qrl.baselines import HybridAStarConfig
from minimal_qrl.dataset import (
    FullGraphGoalSetQRLConfig,
    FullGraphPointGoalQRLConfig,
    create_full_graph_goal_set_qrl_dataset,
    create_full_graph_point_goal_qrl_dataset,
)
from minimal_qrl.envs import CommInspectionDubinsUAV2D
from minimal_qrl.industry_exp.diagnostic_scenario import build_diagnostic_scenario
from minimal_qrl.industry_exp.exact_discrete_value_lp import (
    build_discrete_point_goal_graph,
    reverse_dijkstra,
    select_fixed_terminal_lattice_goal,
)
from minimal_qrl.industry_exp.scalability_scenarios import scenario_to_env_kwargs


def _small_full_graph_dataset(*, stratified_constraints: bool = False):
    scenario = build_diagnostic_scenario()
    env = CommInspectionDubinsUAV2D(**scenario_to_env_kwargs(scenario))
    stats = {}
    dataset = create_full_graph_goal_set_qrl_dataset(
        env,
        FullGraphGoalSetQRLConfig(
            position_resolution=0.5,
            heading_bins=12,
            primitive_steps=5,
            uniform_push_seed=17,
            stratified_constraints=stratified_constraints,
        ),
        collection_stats=stats,
    )
    return env, dataset, stats["full_graph_goal_set_qrl"]


def test_full_graph_dataset_is_exact_macro_graph_plus_terminal_zero_edges():
    _env, dataset, stats = _small_full_graph_dataset()
    infos = dataset.raw_data.transition_infos
    assert len(dataset) == stats["training_transitions"]
    assert len(dataset) == (
        stats["ordinary_macro_edges"]
        + stats["direct_macro_edges_to_goal"]
        + stats["terminal_zero_edges_to_goal"]
    )
    assert bool(infos["full_graph_macro_transition"].all())
    abstract = infos["abstract_goal_edge"].to(dtype=torch.bool)
    assert int(abstract.sum()) == stats["terminal_lattice_states"]
    assert torch.all(dataset.raw_data.rewards[abstract] == 0.0)
    assert torch.all(dataset.raw_data.terminals[abstract])
    assert not any("oracle_value" in key for key in infos)
    assert stats["continuous_env_step_used"] is False
    assert stats["oracle_value_labels_used"] is False
    assert stats["hybrid_astar_trajectories_used"] is False


def test_full_graph_goal_node_is_one_explicit_extended_state_token():
    env, dataset, stats = _small_full_graph_dataset()
    audit = stats["goal_node_audit"]
    goal_index = audit["abstract_goal_feature_index"]
    abstract = dataset.raw_data.transition_infos["abstract_goal_edge"].to(
        dtype=torch.bool
    )
    assert audit["construction"] == "one explicit extended-state observation G"
    assert audit["shared_encoder_and_iqe"] is True
    assert audit["fixed_single_terminal_point_used"] is False
    assert audit["goal_feature_value"] == 1.0
    assert audit["physical_state_goal_feature_max"] == 0.0
    goal_rows = dataset.raw_data.next_observations[abstract]
    assert torch.all(goal_rows[:, goal_index] == 1.0)
    assert torch.all(dataset.raw_data.observations[:, goal_index] == 0.0)
    assert np.isclose(env.abstract_goal_observation()[goal_index], 1.0)


def test_full_graph_global_push_sources_are_uniform_and_independent_of_edges():
    _env, dataset, stats = _small_full_graph_dataset()
    assert stats["uniform_global_push_max_count"] - stats[
        "uniform_global_push_min_count"
    ] <= 1
    infos = dataset.raw_data.transition_infos
    non_abstract = ~infos["abstract_goal_edge"].to(dtype=torch.bool)
    sources = infos["global_push_task_source_observations"][non_abstract]
    edge_sources = dataset.raw_data.observations[non_abstract]
    assert sources.shape == edge_sources.shape
    assert bool(torch.any(torch.any(sources != edge_sources, dim=-1)))


def test_full_graph_batches_keep_independent_edge_endpoints_and_macro_costs():
    _env, dataset, _stats = _small_full_graph_dataset()
    batch = next(
        iter(
            dataset.get_dataloader(
                batch_size=64,
                shuffle=True,
                drop_last=True,
            )
        )
    )
    assert batch.observations.shape == batch.next_observations.shape
    assert batch.actions.shape == (64, 1)
    assert torch.all(batch.rewards <= 0.0)
    assert torch.equal(batch.future_observations, batch.next_observations)
    assert "global_push_task_source_observations" in batch.transition_infos


def test_stratified_batches_include_every_goal_bound_edge_and_sample_ordinary():
    _env, dataset, stats = _small_full_graph_dataset(
        stratified_constraints=True
    )
    ordinary_batch_size = 32
    batch = next(
        iter(
            dataset.get_dataloader(
                batch_size=ordinary_batch_size,
                shuffle=True,
                drop_last=True,
                full_graph_stratified_constraints=True,
            )
        )
    )
    terminal = batch.transition_infos["abstract_goal_edge"].to(dtype=torch.bool)
    direct = batch.transition_infos["full_graph_direct_goal_edge"].to(
        dtype=torch.bool
    ) & ~terminal
    ordinary = ~direct & ~terminal
    assert int(ordinary.sum()) == ordinary_batch_size
    assert int(direct.sum()) == stats["direct_macro_edges_to_goal"]
    assert int(terminal.sum()) == stats["terminal_zero_edges_to_goal"]
    assert len(batch.rewards) == (
        ordinary_batch_size
        + stats["direct_macro_edges_to_goal"]
        + stats["terminal_zero_edges_to_goal"]
    )
    assert torch.equal(
        batch.transition_infos["full_graph_constraint_population_counts"],
        torch.tensor(
            [
                stats["ordinary_macro_edges"],
                stats["direct_macro_edges_to_goal"],
                stats["terminal_zero_edges_to_goal"],
            ],
            dtype=torch.float32,
        ),
    )


def test_diagnostic_shell_exposes_stratified_constraint_phase():
    script = Path(__file__).with_name(
        "run_comm_inspection_diagnostic.sh"
    ).read_text(encoding="utf-8")
    assert "full_graph_goal_set_qrl_stratified_constraints)" in script
    assert "full_graph_goal_set_stratified_constraints" in script
    assert (
        "full_graph_baseline_goal_set_qrl_linear_push_stratified_constraints"
        in script
    )


def test_point_goal_graph_uses_one_physical_goal_and_no_synthetic_edges():
    scenario = build_diagnostic_scenario()
    env = CommInspectionDubinsUAV2D(**scenario_to_env_kwargs(scenario))
    stats = {}
    dataset = create_full_graph_point_goal_qrl_dataset(
        env,
        FullGraphPointGoalQRLConfig(
            position_resolution=0.5,
            heading_bins=12,
            primitive_steps=5,
            uniform_push_seed=17,
            point_goal_candidate_index=0,
        ),
        collection_stats=stats,
    )
    audit = stats["full_graph_point_goal_qrl"]
    infos = dataset.raw_data.transition_infos
    goal_bound = infos["full_graph_direct_goal_edge"].to(dtype=torch.bool)
    goal_obs = infos["task_goal_observations"][0]
    abstract_index = env.task_context_indices["abstract_goal"]

    assert audit["original_terminal_lattice_states"] > 1
    assert audit["synthetic_goal_edges"] == 0
    assert audit["goal_node_audit"]["fixed_single_terminal_point_used"] is True
    assert audit["goal_node_audit"]["construction"] == "one physical lattice observation g*"
    assert not bool(infos["abstract_goal_edge"].any())
    assert bool(goal_bound.any())
    assert torch.all(dataset.raw_data.next_observations[goal_bound] == goal_obs)
    assert torch.all(dataset.raw_data.terminals == goal_bound)
    assert float(goal_obs[abstract_index]) == 0.0
    assert torch.all(infos["task_goal_observations"] == goal_obs)


def test_point_goal_selection_is_explicit_and_reproducible():
    scenario = build_diagnostic_scenario()
    env = CommInspectionDubinsUAV2D(**scenario_to_env_kwargs(scenario))
    first_stats = {}
    second_stats = {}
    config = FullGraphPointGoalQRLConfig(
        position_resolution=0.5,
        heading_bins=12,
        uniform_push_seed=17,
        point_goal_candidate_index=1,
    )
    create_full_graph_point_goal_qrl_dataset(env, config, collection_stats=first_stats)
    create_full_graph_point_goal_qrl_dataset(env, config, collection_stats=second_stats)
    first = first_stats["full_graph_point_goal_qrl"]
    second = second_stats["full_graph_point_goal_qrl"]
    assert first["point_goal_state"] == second["point_goal_state"]
    assert first["training_graph_digest"] == second["training_graph_digest"]

    env.reset(seed=17, options={"device_id": "u_trap_target"})
    standard_config = HybridAStarConfig(
        position_resolution=0.25,
        heading_bins=24,
        primitive_steps=5,
    )
    goal, _goal_index, candidates = select_fixed_terminal_lattice_goal(
        env,
        standard_config,
        candidate_index=0,
    )
    assert len(candidates) == 24
    assert np.allclose(goal, [7.25, 3.5, -3.010693], atol=1e-6)


def test_point_goal_dijkstra_has_exactly_one_physical_zero_node():
    scenario = build_diagnostic_scenario()
    env = CommInspectionDubinsUAV2D(**scenario_to_env_kwargs(scenario))
    env.reset(seed=17, options={"device_id": "u_trap_target"})
    config = HybridAStarConfig(
        position_resolution=0.5,
        heading_bins=12,
        primitive_steps=5,
    )
    _goal, goal_index, candidates = select_fixed_terminal_lattice_goal(
        env,
        config,
        candidate_index=0,
    )
    graph = build_discrete_point_goal_graph(
        env,
        config,
        goal_state_index=goal_index,
    )
    values = reverse_dijkstra(graph)
    other_candidates = candidates[candidates != goal_index]

    assert int(graph.terminal.sum()) == 1
    assert bool(graph.terminal[goal_index])
    assert not bool(np.any(graph.destinations < 0))
    assert values[goal_index] == 0.0
    assert bool(np.any(np.isfinite(values[other_candidates])))
    assert bool(np.all(values[other_candidates][np.isfinite(values[other_candidates])] > 0.0))


def test_diagnostic_shell_exposes_point_goal_control_phase():
    script = Path(__file__).with_name(
        "run_comm_inspection_diagnostic.sh"
    ).read_text(encoding="utf-8")
    assert "full_graph_point_goal_qrl)" in script
    assert "--comm-dataset-mode full_graph_point_goal" in script
    assert "minimal_qrl.industry_exp.point_goal_qrl_diagnostics" in script
