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
from quasimetric_rl.modules.quasimetric_critic.losses.global_push import GlobalPushLoss
from quasimetric_rl.modules.quasimetric_critic.losses.latent_dynamics import (
    LatentDynamicsLoss,
)
from quasimetric_rl.modules.quasimetric_critic.losses.temporal_path import (
    GoalReturnConstraintLoss,
    NstepGoalConsistencyLoss,
    TemporalPathConstraintLoss,
)


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


def test_full_graph_constraint_families_have_independent_duals():
    class FixedQuasimetric(torch.nn.Module):
        def forward(self, zx, zy):
            return torch.tensor([2.0, 3.0, 3.0, 0.5, 2.0])

    class DummyCritic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.quasimetric_model = FixedQuasimetric()

    data = make_batch([-1.0, -4.0, -1.0, -1.0, 0.0])
    data.transition_infos = {
        "full_graph_direct_goal_edge": torch.tensor(
            [False, False, True, True, True]
        ),
        "abstract_goal_edge": torch.tensor(
            [False, False, False, False, True]
        ),
        "full_graph_constraint_population_counts": torch.tensor(
            [2.0, 2.0, 1.0]
        ),
    }
    loss = LocalConstraintLoss(
        epsilon=0.25,
        step_cost=1.0,
        cost_source="negative_reward",
        init_lagrange_multiplier=0.01,
        constraint_mode="full_graph_stratified",
        direct_goal_epsilon=0.25,
        terminal_goal_epsilon=0.0,
    )
    result = loss(
        data,
        CriticBatchInfo(
            critic=DummyCritic(),
            zx=torch.zeros(5, 2),
            zy=torch.zeros(5, 2),
        ),
    )

    assert len(list(loss.parameters())) == 3
    assert torch.isclose(
        result.info["ordinary"]["sq_deviation"], torch.tensor(0.5)
    )
    assert torch.isclose(
        result.info["direct_goal"]["sq_deviation"], torch.tensor(2.0)
    )
    assert torch.isclose(
        result.info["terminal_goal"]["sq_deviation"], torch.tensor(4.0)
    )
    assert torch.isclose(
        result.info["ordinary"]["lagrange_mult"], torch.tensor(0.01)
    )
    assert torch.isclose(
        result.info["direct_goal"]["lagrange_mult"], torch.tensor(0.01)
    )
    assert torch.isclose(
        result.info["terminal_goal"]["lagrange_mult"], torch.tensor(0.01)
    )
    result.loss.backward()
    dual_gradients = [parameter.grad for parameter in loss.parameters()]
    assert all(gradient is not None for gradient in dual_gradients)
    assert all(bool(torch.isfinite(gradient)) for gradient in dual_gradients)
    assert len({round(float(gradient), 8) for gradient in dual_gradients}) == 3


def test_stratified_batch_preserves_uniform_graph_latent_loss_weighting():
    class IdentityDynamics(torch.nn.Module):
        def forward(self, states, actions):
            return states

    class NextValueQuasimetric(torch.nn.Module):
        def forward(self, predicted, target, bidirectional=False):
            assert bidirectional
            value = target[:, 0]
            return torch.stack([value, value], dim=-1)

    class DummyCritic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.latent_dynamics = IdentityDynamics()
            self.quasimetric_model = NextValueQuasimetric()

    data = make_batch([-1.0, -1.0, -1.0, 0.0])
    data.transition_infos = {
        "full_graph_direct_goal_edge": torch.tensor(
            [False, False, True, True]
        ),
        "abstract_goal_edge": torch.tensor([False, False, False, True]),
        "full_graph_constraint_population_counts": torch.tensor(
            [100.0, 1.0, 1.0]
        ),
    }
    result = LatentDynamicsLoss(weight=1.0)(
        data,
        CriticBatchInfo(
            critic=DummyCritic(),
            zx=torch.zeros(4, 2),
            zy=torch.tensor(
                [[1.0, 0.0], [3.0, 0.0], [10.0, 0.0], [20.0, 0.0]]
            ),
        ),
    )

    expected = torch.tensor((50.0 * 1.0 + 50.0 * 9.0 + 1.0 * 100.0) / 101.0)
    assert torch.isclose(result.info["sq_dists"], expected)


def test_global_push_prefers_explicit_free_state_pairs():
    class IdentityEncoder(torch.nn.Module):
        def forward(self, value):
            return value

    class L1Quasimetric(torch.nn.Module):
        def forward(self, source, goal):
            return torch.abs(goal - source).sum(dim=-1)

    class DummyCritic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = IdentityEncoder()
            self.quasimetric_model = L1Quasimetric()

    data = make_batch([-1.0, -1.0])
    data.transition_infos = {
        "task_goal_observations": torch.zeros(2, 2),
        "abstract_goal_edge": torch.zeros(2, dtype=torch.bool),
        "source_terminal_goal_state": torch.zeros(2, dtype=torch.bool),
        "global_push_source_observations": torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
        "global_push_goal_observations": torch.tensor([[3.0, 0.0], [1.0, 4.0]]),
        "global_push_pair_mask": torch.ones(2, dtype=torch.bool),
    }
    critic = DummyCritic()
    batch_info = CriticBatchInfo(critic=critic, zx=torch.zeros(2, 2), zy=torch.zeros(2, 2))
    loss = GlobalPushLoss(
        softplus_beta=0.1,
        softplus_offset=15.0,
        abstract_goal_ratio=0.0,
        state_goal_ratio=1.0,
    )
    result = loss(data, batch_info)
    assert torch.isclose(result.info["global_push_state_state/dist"], torch.tensor(3.0))
    assert torch.isclose(result.info["global_push_state_state/loss"], torch.tensor(-3.0))


def test_global_push_uses_explicit_uniform_task_sources():
    class IdentityEncoder(torch.nn.Module):
        def forward(self, value):
            return value

    class L1Quasimetric(torch.nn.Module):
        def forward(self, source, goal):
            return torch.abs(goal - source).sum(dim=-1)

    class DummyCritic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = IdentityEncoder()
            self.quasimetric_model = L1Quasimetric()

    data = make_batch([-1.0, -1.0])
    data.observations = torch.zeros(2, 2)
    data.transition_infos = {
        "task_goal_observations": torch.zeros(2, 2),
        "global_push_task_source_observations": torch.tensor(
            [[3.0, 0.0], [0.0, 5.0]]
        ),
        "abstract_goal_edge": torch.zeros(2, dtype=torch.bool),
        "source_terminal_goal_state": torch.zeros(2, dtype=torch.bool),
    }
    critic = DummyCritic()
    batch_info = CriticBatchInfo(
        critic=critic,
        zx=torch.zeros(2, 2),
        zy=torch.zeros(2, 2),
    )
    result = GlobalPushLoss(
        softplus_beta=0.1,
        softplus_offset=15.0,
        abstract_goal_ratio=1.0,
        state_goal_ratio=0.0,
    )(data, batch_info)
    assert torch.isclose(
        result.info["global_push_task_set/dist"], torch.tensor(4.0)
    )
    assert torch.isclose(
        result.info["global_push_task_set/loss"], torch.tensor(-4.0)
    )


def test_global_push_linear_objective_has_constant_unsaturated_gradient():
    distances = torch.tensor([2.0, 7.0], requires_grad=True)
    loss = GlobalPushLoss(
        softplus_beta=0.1,
        softplus_offset=15.0,
        abstract_goal_ratio=1.0,
        state_goal_ratio=0.0,
    )._push_loss(distances)

    assert torch.isclose(loss, torch.tensor(-4.5))
    loss.backward()
    assert torch.allclose(distances.grad, torch.tensor([-0.5, -0.5]))


def test_temporal_path_uses_executed_multistep_cost_as_one_sided_bound():
    class IdentityEncoder(torch.nn.Module):
        def forward(self, value):
            return value

    class L1Quasimetric(torch.nn.Module):
        def forward(self, source, goal):
            return torch.abs(goal - source).sum(dim=-1)

    class DummyCritic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = IdentityEncoder()
            self.quasimetric_model = L1Quasimetric()

    data = make_batch([-1.0, -1.0])
    data.future_observations = torch.tensor([[3.0, 0.0], [1.0, 0.0]])
    data.transition_infos = {
        "temporal_future_cost": torch.tensor([2.0, 5.0]),
        "temporal_future_steps": torch.tensor([3, 3]),
        "abstract_goal_edge": torch.zeros(2, dtype=torch.bool),
    }
    critic = DummyCritic()
    batch_info = CriticBatchInfo(critic=critic, zx=torch.zeros(2, 2), zy=torch.zeros(2, 2))
    result = TemporalPathConstraintLoss(weight=1.0, min_future_steps=2)(data, batch_info)

    # Only the first behavior path violates its bound: ((3 - 2) / (2 + 1))^2 / 2.
    assert torch.isclose(result.loss, torch.tensor(1.0 / 18.0))
    assert torch.isclose(result.info["count"], torch.tensor(2.0))


def test_goal_return_uses_only_naturally_successful_transitions():
    class IdentityEncoder(torch.nn.Module):
        def forward(self, value):
            return value

    class L1Quasimetric(torch.nn.Module):
        def forward(self, source, goal):
            return torch.abs(goal - source).sum(dim=-1)

    class DummyCritic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = IdentityEncoder()
            self.quasimetric_model = L1Quasimetric()

    data = make_batch([-1.0, -1.0])
    data.transition_infos = {
        "task_goal_observations": torch.tensor([[4.0, 0.0], [100.0, 0.0]]),
        "goal_return_cost": torch.tensor([2.0, 0.1]),
        "goal_return_mask": torch.tensor([True, False]),
        "abstract_goal_edge": torch.zeros(2, dtype=torch.bool),
    }
    critic = DummyCritic()
    batch_info = CriticBatchInfo(critic=critic, zx=torch.zeros(2, 2), zy=torch.zeros(2, 2))
    result = GoalReturnConstraintLoss(weight=1.0)(data, batch_info)

    assert torch.isclose(result.loss, torch.tensor(4.0 / 9.0))
    assert torch.isclose(result.info["count"], torch.tensor(1.0))


def test_optional_nstep_goal_uses_frozen_future_goal_estimate():
    class IdentityEncoder(torch.nn.Module):
        def forward(self, value):
            return value

    class L1Quasimetric(torch.nn.Module):
        def forward(self, source, goal):
            return torch.abs(goal - source).sum(dim=-1)

    class DummyCritic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = IdentityEncoder()
            self.quasimetric_model = L1Quasimetric()

        def forward(self, source, goal):
            return self.quasimetric_model(self.encoder(source), self.encoder(goal))

    data = make_batch([-1.0])
    data.future_observations = torch.tensor([[1.0, 0.0]])
    data.transition_infos = {
        "task_goal_observations": torch.tensor([[4.0, 0.0]]),
        "temporal_future_cost": torch.tensor([0.5]),
        "temporal_future_steps": torch.tensor([3]),
        "abstract_goal_edge": torch.zeros(1, dtype=torch.bool),
    }
    critic = DummyCritic()
    batch_info = CriticBatchInfo(critic=critic, zx=torch.zeros(1, 2), zy=torch.zeros(1, 2))
    loss = NstepGoalConsistencyLoss(
        critic=critic,
        weight=1.0,
        min_future_steps=2,
        target_tau=0.005,
    )
    result = loss(data, batch_info)

    # d(s, G)=4, while the semi-gradient bound is 0.5+d_target(s_3,G)=3.5.
    assert torch.isclose(result.loss, torch.tensor(1.0 / 81.0))
    assert torch.isclose(result.info["target_future_dist"], torch.tensor(3.0))


if __name__ == "__main__":
    test_fixed_mode_uses_step_cost()
    test_negative_reward_mode_uses_nonnegative_costs()
    test_positive_rewards_are_clipped_to_zero_and_reported()
    print("All task-aware QRL loss tests passed.")
