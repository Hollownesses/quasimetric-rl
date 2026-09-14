from __future__ import annotations

import uuid

import gym
import numpy as np
import pytest
import torch

from quasimetric_rl.data import Dataset, EpisodeData, register_offline_env


class _TinyEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(-100.0, 100.0, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)


def _episode(rewards, *, success):
    n = len(rewards)
    observations = np.arange(n, dtype=np.float32).reshape(-1, 1)
    next_observations = np.arange(1, n + 1, dtype=np.float32).reshape(-1, 1)
    return EpisodeData.from_simple_trajectory(
        observations=observations,
        actions=np.zeros((n, 1), dtype=np.float32),
        next_observations=next_observations,
        rewards=-np.asarray(rewards, dtype=np.float32),
        terminals=np.asarray([False] * (n - 1) + [True], dtype=np.bool_),
        timeouts=np.zeros((n,), dtype=np.bool_),
        transition_infos={
            "task_success_episode": np.full((n,), success, dtype=np.bool_),
        },
    )


def _dataset():
    key = f"temporal_test_{uuid.uuid4().hex}"
    episodes = (
        _episode([1.0, 2.0, 3.0], success=False),
        _episode([4.0, 5.0], success=True),
    )
    register_offline_env(
        key,
        key,
        create_env_fn=_TinyEnv,
        load_episodes_fn=lambda: iter(episodes),
    )
    return Dataset(key, key, future_observation_discount=0.99)


def test_temporal_future_cost_uses_same_episode_prefix(monkeypatch):
    dataset = _dataset()
    assert torch.allclose(
        dataset.obs_indices_to_cumulative_cost,
        torch.tensor([0.0, 1.0, 3.0, 6.0, 0.0, 4.0, 9.0]),
    )

    monkeypatch.setattr(
        torch.distributions.Categorical,
        "sample",
        lambda self: torch.tensor([2, 1], dtype=torch.int64),
    )
    batch = dataset[torch.tensor([0, 3])]
    assert torch.allclose(batch.transition_infos["temporal_future_cost"], torch.tensor([6.0, 9.0]))
    assert torch.equal(batch.transition_infos["temporal_future_steps"], torch.tensor([3, 2]))


def test_success_weight_changes_only_successful_transition_sampling():
    dataset = _dataset()
    loader = dataset.get_dataloader(batch_size=2, successful_transition_weight=7.0)
    weights = loader.sampler.sampler.weights
    assert torch.equal(weights, torch.tensor([1.0, 1.0, 1.0, 7.0, 7.0], dtype=torch.float64))


def _mqe_episode(
    values,
    costs,
    *,
    success,
    task_goal=99.0,
    abstract_edge=False,
    terminal=True,
):
    values = np.asarray(values, dtype=np.float32)
    n = len(costs)
    assert len(values) == n + 1
    return EpisodeData.from_simple_trajectory(
        observations=values[:-1, None],
        actions=np.zeros((n, 1), dtype=np.float32),
        next_observations=values[1:, None],
        rewards=-np.asarray(costs, dtype=np.float32),
        terminals=np.asarray([False] * (n - 1) + [terminal], dtype=np.bool_),
        timeouts=np.asarray([False] * (n - 1) + [not terminal], dtype=np.bool_),
        transition_infos={
            "abstract_goal_edge": np.full(n, abstract_edge, dtype=np.bool_),
            "source_terminal_goal_state": np.full(n, abstract_edge, dtype=np.bool_),
            "task_success_episode": np.full(n, success, dtype=np.bool_),
            "task_goal_observations": np.full((n, 1), task_goal, dtype=np.float32),
        },
    )


def _mqe_dataset(*episodes):
    key = f"mqe_temporal_test_{uuid.uuid4().hex}"
    register_offline_env(
        key,
        key,
        create_env_fn=_TinyEnv,
        load_episodes_fn=lambda: iter(episodes),
    )
    return Dataset(key, key, future_observation_discount=0.99)


def test_mqe_sampling_parameters_validate_boundaries():
    config = Dataset.MQEInspiredWaypointSampling(
        goal_discount=0.0,
        waypoint_lambda=0.0,
        next_state_probability=0.0,
        terminal_anchor_fraction=0.0,
    )
    assert config.goal_discount == 0.0
    assert config.terminal_anchor_fraction == 0.0
    with pytest.raises(ValueError):
        Dataset.MQEInspiredWaypointSampling(goal_discount=1.0)
    with pytest.raises(ValueError):
        Dataset.MQEInspiredWaypointSampling(waypoint_lambda=-0.1)
    with pytest.raises(ValueError):
        Dataset.MQEInspiredWaypointSampling(next_state_probability=1.1)
    with pytest.raises(ValueError):
        Dataset.MQEInspiredWaypointSampling(terminal_anchor_fraction=-0.1)


def test_mqe_bernoulli_geometric_mixture_preserves_controlled_k1_mass():
    dataset = _mqe_dataset(
        _mqe_episode(np.arange(1001), np.ones(1000), success=False)
    )
    dataset.configure_mqe_inspired_waypoint_sampling(
        Dataset.MQEInspiredWaypointSampling(
            goal_discount=0.999,
            waypoint_lambda=0.95,
            next_state_probability=0.2,
            terminal_anchor_fraction=0.0,
        )
    )
    torch.manual_seed(20260914)
    batch = dataset[torch.zeros(20_000, dtype=torch.int64)]
    infos = batch.transition_infos
    forced_ratio = infos["mqe_waypoint_forced_one_step"].float().mean()
    k1_ratio = (infos["mqe_waypoint_steps"] == 1).float().mean()
    assert abs(float(forced_ratio) - 0.2) < 0.02
    assert abs(float(k1_ratio) - 0.24) < 0.025


def test_mqe_waypoints_stay_in_episode_and_accumulate_variable_cost(monkeypatch):
    dataset = _mqe_dataset(
        _mqe_episode([10, 11, 12, 13], [1.0, 2.0, 4.0], success=False),
        _mqe_episode([20, 21, 22], [8.0, 16.0], success=False),
    )
    dataset.configure_mqe_inspired_waypoint_sampling(
        Dataset.MQEInspiredWaypointSampling(
            goal_discount=0.9,
            waypoint_lambda=0.9,
            next_state_probability=0.0,
            terminal_anchor_fraction=0.0,
        )
    )
    samples = iter(
        [
            torch.full((4,), 100, dtype=torch.int64),
            torch.full((4,), 100, dtype=torch.int64),
        ]
    )
    monkeypatch.setattr(dataset, "_geometric_steps", lambda *_args: next(samples))
    infos = dataset[torch.tensor([0, 1, 3, 4])].transition_infos
    assert torch.equal(infos["mqe_waypoint_steps"], torch.tensor([3, 2, 2, 1]))
    assert torch.equal(
        infos["mqe_waypoint_observations"].flatten(),
        torch.tensor([13.0, 13.0, 22.0, 22.0]),
    )
    assert torch.equal(
        infos["mqe_waypoint_cost"],
        torch.tensor([7.0, 6.0, 24.0, 16.0]),
    )
    assert torch.equal(
        infos["mqe_waypoint_episode_index"], torch.tensor([0, 0, 1, 1])
    )


def test_mqe_terminal_anchor_quota_is_independent_of_replay_success_rate():
    physical_success = _mqe_episode(
        [0, 1, 2, 3], [1.0, 2.0, 3.0], success=True
    )
    synthetic_abstract = _mqe_episode(
        [3, 99], [0.0], success=False, abstract_edge=True
    )
    physical_failure = _mqe_episode(
        [10, 11, 12], [4.0, 5.0], success=False
    )
    dataset = _mqe_dataset(
        physical_success,
        synthetic_abstract,
        physical_failure,
    )
    dataset.configure_mqe_inspired_waypoint_sampling(
        Dataset.MQEInspiredWaypointSampling(
            goal_discount=0.0,
            waypoint_lambda=0.0,
            next_state_probability=0.0,
            terminal_anchor_fraction=0.1,
        )
    )
    torch.manual_seed(7)
    # The main replay batch contains only unsuccessful transitions.  MQE still
    # reserves exactly ceil(256 * .1) legal terminal-anchor slots.
    replay_indices = torch.tensor([4, 5], dtype=torch.int64).repeat(128)
    infos = dataset[replay_indices].transition_infos
    anchors = infos["mqe_waypoint_terminal_anchor"]
    assert int(anchors.sum()) == 26
    assert torch.all(infos["mqe_waypoint_source_transition_index"][anchors] < 3)
    assert torch.equal(
        infos["mqe_waypoint_observations"][anchors].flatten(),
        torch.full((26,), 3.0),
    )
    assert torch.equal(
        infos["mqe_waypoint_goal_observations"][anchors].flatten(),
        torch.full((26,), 99.0),
    )
    expected_costs = torch.tensor([6.0, 5.0, 3.0])
    anchor_sources = infos["mqe_waypoint_source_transition_index"][anchors]
    assert torch.equal(
        infos["mqe_waypoint_cost"][anchors],
        expected_costs[anchor_sources],
    )


def test_mqe_anchor_configuration_fails_loudly_without_legal_success():
    dataset = _mqe_dataset(
        _mqe_episode([0, 1, 2], [1.0, 1.0], success=False)
    )
    with pytest.raises(ValueError, match="naturally successful"):
        dataset.configure_mqe_inspired_waypoint_sampling(
            Dataset.MQEInspiredWaypointSampling(terminal_anchor_fraction=0.1)
        )
