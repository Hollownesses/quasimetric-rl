from __future__ import annotations

import numpy as np
import torch

from minimal_qrl.envs import CommInspectionDubinsUAV2D
from minimal_qrl.industry_exp.diagnostic_scenario import build_diagnostic_scenario
from minimal_qrl.industry_exp.scalability_scenarios import scenario_to_env_kwargs
from minimal_qrl.industry_exp.supervised_iqe_oracle import (
    _make_agent,
    _ranking_accuracy,
    _regression_metrics,
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
