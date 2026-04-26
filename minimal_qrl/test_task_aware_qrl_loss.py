#!/usr/bin/env python3
"""
Lightweight tests for task-aware QRL local costs.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from quasimetric_rl.data import BatchData
from quasimetric_rl.modules.quasimetric_critic.losses import CriticBatchInfo
from quasimetric_rl.modules.quasimetric_critic.losses.local_constraint import LocalConstraintLoss


def make_batch(rewards):
    n = len(rewards)
    obs = torch.zeros(n, 2)
    return BatchData(
        observations=obs,
        actions=torch.zeros(n, 1),
        next_observations=obs.clone(),
        rewards=torch.tensor(rewards, dtype=torch.float32),
        terminals=torch.zeros(n, dtype=torch.bool),
        timeouts=torch.zeros(n, dtype=torch.bool),
        future_observations=obs.clone(),
    )


def test_fixed_mode_uses_step_cost():
    loss = LocalConstraintLoss(epsilon=0.25, step_cost=1.5, cost_source="fixed", init_lagrange_multiplier=0.01)
    costs = loss._target_cost(make_batch([-0.1, -2.0, 1.0]), torch.zeros(3))
    assert torch.allclose(costs, torch.tensor([1.5, 1.5, 1.5]))


def test_negative_reward_mode_uses_nonnegative_costs():
    loss = LocalConstraintLoss(epsilon=0.25, step_cost=1.0, cost_source="negative_reward", init_lagrange_multiplier=0.01)
    costs = loss._target_cost(make_batch([-0.1, -2.0, 0.0]), torch.zeros(3))
    assert torch.allclose(costs, torch.tensor([0.1, 2.0, 0.0]))


def test_positive_rewards_are_clipped_to_zero_and_reported():
    class DummyQuasimetric(torch.nn.Module):
        def forward(self, zx, zy):
            return torch.tensor([0.2, 0.3, 0.4], dtype=torch.float32)

    class DummyCritic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.quasimetric_model = DummyQuasimetric()

    loss = LocalConstraintLoss(epsilon=0.25, step_cost=1.0, cost_source="negative_reward", init_lagrange_multiplier=0.01)
    data = make_batch([1.0, -2.0, 0.0])
    info = CriticBatchInfo(critic=DummyCritic(), zx=torch.zeros(3, 2), zy=torch.zeros(3, 2))

    result = loss(data, info)

    assert torch.isclose(result.info["target_cost_min"], torch.tensor(0.0))
    assert torch.isclose(result.info["target_cost_max"], torch.tensor(2.0))
    assert torch.isclose(result.info["target_cost_mean"], torch.tensor(2.0 / 3.0))


if __name__ == "__main__":
    test_fixed_mode_uses_step_cost()
    test_negative_reward_mode_uses_nonnegative_costs()
    test_positive_rewards_are_clipped_to_zero_and_reported()
    print("All task-aware QRL loss tests passed.")
