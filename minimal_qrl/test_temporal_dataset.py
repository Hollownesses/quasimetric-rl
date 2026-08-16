from __future__ import annotations

import uuid

import gym
import numpy as np
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
