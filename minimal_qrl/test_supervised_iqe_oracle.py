from __future__ import annotations

import numpy as np
import torch

from minimal_qrl.envs import CommInspectionDubinsUAV2D
from minimal_qrl.industry_exp.diagnostic_scenario import build_diagnostic_scenario
from minimal_qrl.industry_exp.scalability_scenarios import scenario_to_env_kwargs
from minimal_qrl.industry_exp.supervised_iqe_oracle import (
    TARGETED_U_TRAP_STRATA,
    _coverage_sample,
    _load_failed_start_states,
    _make_agent,
    _ranking_accuracy,
    _regression_metrics,
    _sample_supervised_batch_indices,
    _targeted_u_trap_pools,
    train_supervised,
)


def test_supervised_metrics_and_successor_ranking_are_exact_for_perfect_predictions():
    targets = np.asarray([3.0, 1.0, 2.0, 8.0, 5.0], dtype=np.float32)
    metrics = _regression_metrics(targets.copy(), targets)
    accuracy, pairs = _ranking_accuracy(
        targets.copy(),
        targets,
        np.asarray([0, 0, 0, 1, 1], dtype=np.int32),
    )
    assert metrics["mae"] == 0.0
    assert np.isclose(metrics["pearson"], 1.0)
    assert np.isclose(metrics["spearman"], 1.0)
    assert accuracy == 1.0
    assert pairs == 4


def test_supervised_objective_updates_encoder_and_iqe_but_not_latent_dynamics():
    scenario = build_diagnostic_scenario()
    env = CommInspectionDubinsUAV2D(**scenario_to_env_kwargs(scenario))
    env.reset(seed=3, options={"device_id": "u_trap_target"})
    agent = _make_agent(env, scenario, num_critics=2, total_steps=2)
    rng = np.random.default_rng(7)
    observations = np.stack(
        [env.state_to_observation(env.sample_valid_state(seed=100 + index)) for index in range(8)]
    ).astype(np.float32)
    goal = env.abstract_goal_observation().astype(np.float32)
    dataset = {
        "train_observation": observations,
        "train_goal": np.repeat(goal[None, :], len(observations), axis=0),
        "train_value": rng.uniform(5.0, 20.0, size=len(observations)).astype(np.float32),
    }
    encoder_before = [parameter.detach().clone() for parameter in agent.critics[0].encoder.parameters()]
    dynamics_before = [
        parameter.detach().clone() for parameter in agent.critics[0].latent_dynamics.parameters()
    ]
    history = train_supervised(
        agent,
        dataset,
        device=torch.device("cpu"),
        steps=2,
        batch_size=4,
        learning_rate=1e-4,
        loss_name="huber",
        huber_delta=10.0,
        seed=9,
        log_interval=1,
    )
    assert history[-1]["step"] == 2.0
    assert any(
        not torch.equal(before, after)
        for before, after in zip(encoder_before, agent.critics[0].encoder.parameters())
    )
    assert all(
        torch.equal(before, after)
        for before, after in zip(dynamics_before, agent.critics[0].latent_dynamics.parameters())
    )


def test_targeted_batch_sampler_enforces_exact_local_global_mix():
    dataset = {
        "train_value": np.zeros(20, dtype=np.float32),
        "train_sampling_group": np.asarray([0] * 10 + [1] * 10, dtype=np.int8),
    }
    indices = _sample_supervised_batch_indices(
        dataset,
        12,
        rng=np.random.default_rng(4),
        local_fraction=0.5,
    )
    groups = dataset["train_sampling_group"][indices]
    assert int(np.sum(groups == 0)) == 6
    assert int(np.sum(groups == 1)) == 6


def test_coverage_sampler_visits_every_lattice_state_before_repeating():
    pool = np.arange(7, dtype=np.int64)
    sampled = _coverage_sample(pool, 17, rng=np.random.default_rng(5))
    counts = np.bincount(sampled, minlength=len(pool))
    assert set(sampled[:7]) == set(pool)
    assert np.all((counts == 2) | (counts == 3))


def test_targeted_u_trap_strata_cover_requested_geometry_and_failed_heading():
    scenario = build_diagnostic_scenario()
    states = np.asarray(
        [
            [4.0, 3.65, 0.0],   # deep interior
            [4.9, 3.65, 0.0],   # east closed wall
            [2.4, 3.65, 0.0],   # west exit
            [3.2, 3.65, 0.0],   # deep-to-exit transition
            [4.75, 3.15, 3.0],  # failed start neighborhood
        ],
        dtype=np.float32,
    )
    pools = _targeted_u_trap_pools(
        states,
        scenario,
        np.asarray([[4.75, 3.15, 3.0]], dtype=np.float32),
        failure_position_radius=0.2,
        failure_heading_radius=0.2,
    )
    assert tuple(pools) == TARGETED_U_TRAP_STRATA
    assert 0 in pools["deep_interior"]
    assert 1 in pools["east_closed_wall"]
    assert 2 in pools["west_exit"]
    assert 3 in pools["deep_to_exit_transition"]
    assert np.array_equal(pools["failed_start_neighborhood"], np.asarray([4]))


def test_failed_starts_are_loaded_from_prior_mppi_results(tmp_path):
    path = tmp_path / "baseline_results.json"
    path.write_text(
        '{"episode_results": ['
        '{"stratum": "u_trap", "success": 0.0, "start": [4.75, 3.15, 3.0]},'
        '{"stratum": "u_trap", "success": 1.0, "start": [3.0, 3.0, 0.0]},'
        '{"stratum": "easy_open", "success": 0.0, "start": [1.0, 1.0, 0.0]}'
        ']}'
    )
    failed = _load_failed_start_states(path)
    assert failed.shape == (1, 3)
    assert np.allclose(failed[0], [4.75, 3.15, 3.0])
