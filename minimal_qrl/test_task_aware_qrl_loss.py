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
from quasimetric_rl.modules.quasimetric_critic.losses import QuasimetricCriticLosses
from quasimetric_rl.modules.quasimetric_critic.losses.local_constraint import LocalConstraintLoss
from quasimetric_rl.modules.quasimetric_critic.losses.global_push import GlobalPushLoss
from quasimetric_rl.modules.optim import AdamWSpec
from quasimetric_rl.modules.utils import LossResult
from quasimetric_rl.modules.quasimetric_critic.losses.temporal_path import (
    GoalReturnConstraintLoss,
    MQEInspiredWaypointConsistencyLoss,
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


def test_kkt_functional_has_nonzero_primal_gradient_at_active_boundary():
    critic = _ScalarCritic(1.0)
    data = make_batch([-1.0])
    batch_info = CriticBatchInfo(
        critic=critic,
        zx=torch.zeros(1, 2),
        zy=torch.ones(1, 2),
    )
    loss = LocalConstraintLoss(
        epsilon=0.25,
        step_cost=1.0,
        cost_source="negative_reward",
        init_lagrange_multiplier=0.2,
        mode="kkt_functional",
        augmented_lagrangian_rho=1.0,
        dual_hidden_sizes=(8,),
        dual_max=10.0,
        dual_steps=3,
        observation_size=2,
    )

    result = loss(data, batch_info)
    result.loss.backward()

    # At d=c the augmented hinge is flat, but lambda*(d-c) still supplies the
    # KKT boundary reaction to the critic.
    assert torch.isclose(
        critic.quasimetric_model.value.grad,
        torch.tensor(0.2),
        atol=1e-6,
    )
    assert all(parameter.grad is None for parameter in loss.parameters())


def test_kkt_functional_dual_update_is_detached_from_critic_and_increases_on_violation():
    critic = _ScalarCritic(2.0)
    data = make_batch([-1.0, -1.0])
    batch_info = CriticBatchInfo(
        critic=critic,
        zx=torch.zeros(2, 2),
        zy=torch.ones(2, 2),
    )
    loss = LocalConstraintLoss(
        epsilon=0.25,
        step_cost=1.0,
        cost_source="negative_reward",
        init_lagrange_multiplier=0.2,
        mode="kkt_functional",
        augmented_lagrangian_rho=1.0,
        dual_hidden_sizes=(8,),
        dual_max=10.0,
        dual_steps=3,
        observation_size=2,
    )
    optimizer = torch.optim.SGD(loss.parameters(), lr=0.1)

    before = float(loss.dual_loss(data, batch_info).info["lagrange_mult_mean"])
    optimizer.zero_grad()
    dual_result = loss.dual_loss(data, batch_info)
    dual_result.loss.backward()
    optimizer.step()
    after = float(loss.dual_loss(data, batch_info).info["lagrange_mult_mean"])

    assert critic.quasimetric_model.value.grad is None
    assert after > before


def test_kkt_functional_keeps_dual_frozen_until_violations_wake_it_up():
    critic = _ScalarCritic(0.0)
    data = make_batch([-1.0, -1.0])
    batch_info = CriticBatchInfo(
        critic=critic,
        zx=torch.zeros(2, 2),
        zy=torch.ones(2, 2),
    )
    loss = LocalConstraintLoss(
        epsilon=0.25,
        step_cost=1.0,
        cost_source="negative_reward",
        init_lagrange_multiplier=0.2,
        mode="kkt_functional",
        augmented_lagrangian_rho=1.0,
        dual_hidden_sizes=(8,),
        dual_max=10.0,
        dual_steps=1,
        dual_active_margin=1.0,
        dual_start_violation_fraction=0.05,
        observation_size=2,
    )
    optimizer = torch.optim.SGD(loss.parameters(), lr=0.1)

    before = float(loss.dual_loss(data, batch_info).info["lagrange_mult_mean"])
    for _ in range(5):
        optimizer.zero_grad()
        result = loss.dual_loss(data, batch_info)
        result.loss.backward()
        optimizer.step()
    after_result = loss.dual_loss(data, batch_info)
    after = float(after_result.info["lagrange_mult_mean"])

    assert not bool(loss.dual_updates_started)
    assert float(after_result.info["update_enabled"]) == 0.0
    assert torch.isclose(torch.tensor(after), torch.tensor(before))


def test_kkt_functional_raw_floor_keeps_recovery_gradient_alive():
    loss = LocalConstraintLoss(
        epsilon=0.25,
        step_cost=1.0,
        cost_source="negative_reward",
        init_lagrange_multiplier=0.2,
        mode="kkt_functional",
        augmented_lagrangian_rho=1.0,
        dual_hidden_sizes=(8,),
        dual_max=10.0,
        dual_steps=1,
        dual_raw_min=-10.0,
        observation_size=2,
    )
    assert loss.dual_network is not None
    last_layer = loss.dual_network.module[-1]
    with torch.no_grad():
        last_layer.bias.fill_(-100.0)

    lagrange_mult, raw_unbounded, raw_effective = (
        loss._functional_lagrange_multiplier(
            torch.zeros(2, 2),
            torch.zeros(2, 2),
            torch.ones(2),
        )
    )
    (-lagrange_mult.mean()).backward()

    assert torch.all(raw_unbounded < -10.0)
    assert torch.allclose(raw_effective, torch.full_like(raw_effective, -10.0))
    assert last_layer.bias.grad is not None
    assert float(last_layer.bias.grad.abs().sum()) > 0.0


def test_kkt_functional_runs_configured_dual_steps_per_critic_step():
    class ZeroLoss(torch.nn.Module):
        def forward(self, data, critic_batch_info):
            return LossResult(
                loss=critic_batch_info.critic.quasimetric_model.value * 0.0,
                info={},
            )

    class ZeroTargetLoss(ZeroLoss):
        def update_target(self, critic):
            return None

    critic = _ScalarCritic(2.0)
    local = LocalConstraintLoss(
        epsilon=0.25,
        step_cost=1.0,
        cost_source="negative_reward",
        init_lagrange_multiplier=0.2,
        mode="kkt_functional",
        augmented_lagrangian_rho=1.0,
        dual_hidden_sizes=(8,),
        dual_max=10.0,
        dual_steps=3,
        observation_size=2,
    )
    zero = ZeroLoss()
    zero_target = ZeroTargetLoss()
    losses = QuasimetricCriticLosses(
        critic,
        total_optim_steps=2,
        global_push=zero,
        local_constraint=local,
        latent_dynamics=zero,
        abstract_goal_edge=zero,
        temporal_path=zero,
        goal_return=zero,
        nstep_goal=zero_target,
        mqe_waypoint=zero_target,
        critic_optim_spec=AdamWSpec.Conf(lr=1e-3).make(),
        lagrange_mult_optim_spec=AdamWSpec.Conf(lr=1e-2).make(),
    )
    data = make_batch([-1.0, -1.0])
    batch_info = CriticBatchInfo(
        critic=critic,
        zx=torch.zeros(2, 2),
        zy=torch.ones(2, 2),
    )

    result = losses(data, batch_info, optimize=True)

    optimizer_steps = {
        int(state["step"].item())
        for state in losses.lagrange_mult_optim.optim.state.values()
    }
    assert optimizer_steps == {3}
    assert "dual_update" in result.info["local_constraint"]


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


def test_global_push_remains_bounded_softplus_objective():
    distances = torch.tensor([2.0, 7.0], requires_grad=True)
    loss = GlobalPushLoss(
        softplus_beta=0.1,
        softplus_offset=15.0,
        abstract_goal_ratio=1.0,
        state_goal_ratio=0.0,
    )._push_loss(distances)
    expected = torch.nn.functional.softplus(
        15.0 - distances,
        beta=0.1,
    ).mean()
    assert torch.isclose(loss, expected)
    assert loss.item() > 0.0


def test_global_push_linear_objective_has_constant_unsaturated_gradient():
    distances = torch.tensor([2.0, 7.0], requires_grad=True)
    loss = GlobalPushLoss(
        objective="linear",
        softplus_beta=0.1,
        softplus_offset=15.0,
        abstract_goal_ratio=1.0,
        state_goal_ratio=0.0,
    )._push_loss(distances)

    loss.backward()

    assert torch.isclose(loss, torch.tensor(-4.5))
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
    assert torch.isclose(result.info["future_steps"], torch.tensor(3.0))


class _ScalarQuasimetric(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(float(value)))

    def forward(self, source, goal):
        return self.value.expand(source.shape[0])


class _ScalarCritic(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.encoder = torch.nn.Identity()
        self.quasimetric_model = _ScalarQuasimetric(value)

    def forward(self, source, goal):
        return self.quasimetric_model(self.encoder(source), self.encoder(goal))


def _mqe_loss_batch(cost=0.0, *, terminal_anchor=False):
    data = make_batch([-1.0])
    data.transition_infos = {
        "mqe_waypoint_valid": torch.tensor([True]),
        "mqe_waypoint_source_observations": torch.tensor([[0.0, 0.0]]),
        "mqe_waypoint_observations": torch.tensor([[1.0, 0.0]]),
        "mqe_waypoint_goal_observations": torch.tensor([[4.0, 0.0]]),
        "mqe_waypoint_cost": torch.tensor([cost]),
        "mqe_waypoint_steps": torch.tensor([3]),
        "mqe_waypoint_goal_steps": torch.tensor([5]),
        "mqe_waypoint_forced_one_step": torch.tensor([False]),
        "mqe_waypoint_physical_goal": torch.tensor([not terminal_anchor]),
        "mqe_waypoint_terminal_anchor": torch.tensor([terminal_anchor]),
    }
    return data


def _mqe_family_batch():
    data = make_batch([-1.0, -1.0, -1.0])
    data.transition_infos = {
        "mqe_waypoint_valid": torch.ones(3, dtype=torch.bool),
        "mqe_waypoint_source_observations": torch.zeros(3, 2),
        "mqe_waypoint_observations": torch.ones(3, 2),
        "mqe_waypoint_goal_observations": torch.full((3, 2), 4.0),
        "mqe_waypoint_cost": torch.tensor([0.0, 0.0, 10.0]),
        "mqe_waypoint_steps": torch.tensor([1, 2, 3]),
        "mqe_waypoint_goal_steps": torch.tensor([2, 3, 4]),
        "mqe_waypoint_forced_one_step": torch.tensor([True, False, False]),
        "mqe_waypoint_physical_goal": torch.tensor([True, True, False]),
        "mqe_waypoint_terminal_anchor": torch.tensor([False, False, True]),
        "mqe_waypoint_device_index": torch.tensor([0, 1, 0]),
    }
    return data


def test_mqe_two_sided_huber_has_gradients_for_over_and_under_estimates():
    gradients = []
    for current_value in (2.0, -2.0):
        critic = _ScalarCritic(0.0)
        loss = MQEInspiredWaypointConsistencyLoss(
            critic=critic,
            weight=1.0,
            target_tau=0.25,
            huber_delta=1.0,
        )
        critic.quasimetric_model.value.data.fill_(current_value)
        info = CriticBatchInfo(
            critic=critic,
            zx=torch.zeros(1, 2),
            zy=torch.zeros(1, 2),
        )
        result = loss(_mqe_loss_batch(), info)
        result.loss.backward()
        gradients.append(float(critic.quasimetric_model.value.grad))

    assert gradients[0] > 0.0
    assert gradients[1] < 0.0


def test_mqe_terminal_anchor_uses_exact_zero_terminal_to_goal_distance():
    critic = _ScalarCritic(4.0)
    loss = MQEInspiredWaypointConsistencyLoss(
        critic=critic,
        weight=1.0,
        target_tau=0.25,
        huber_delta=1.0,
    )
    loss.target_critic.quasimetric_model.value.data.fill_(99.0)
    info = CriticBatchInfo(
        critic=critic,
        zx=torch.zeros(1, 2),
        zy=torch.zeros(1, 2),
    )
    result = loss(_mqe_loss_batch(cost=3.0, terminal_anchor=True), info)

    assert torch.isclose(result.info["target_waypoint_dist"], torch.tensor(0.0))
    assert torch.isclose(result.info["target"], torch.tensor(3.0))
    assert torch.isclose(result.info["terminal_anchor_count"], torch.tensor(1.0))
    assert torch.isclose(result.info["terminal_anchor_fraction"], torch.tensor(1.0))


def test_mqe_target_is_stop_gradient_and_updates_only_by_ema():
    critic = _ScalarCritic(0.0)
    loss = MQEInspiredWaypointConsistencyLoss(
        critic=critic,
        weight=1.0,
        target_tau=0.25,
        huber_delta=1.0,
    )
    critic.quasimetric_model.value.data.fill_(4.0)
    info = CriticBatchInfo(
        critic=critic,
        zx=torch.zeros(1, 2),
        zy=torch.zeros(1, 2),
    )
    result = loss(_mqe_loss_batch(cost=1.0), info)
    result.loss.backward()

    target_parameter = loss.target_critic.quasimetric_model.value
    assert target_parameter.grad is None
    assert torch.isclose(target_parameter.detach(), torch.tensor(0.0))
    loss.update_target(critic)
    assert torch.isclose(target_parameter.detach(), torch.tensor(1.0))


def test_mqe_mixed_family_normalization_preserves_original_batch_mean():
    critic = _ScalarCritic(0.0)
    loss = MQEInspiredWaypointConsistencyLoss(
        critic=critic,
        weight=1.0,
        target_tau=0.25,
        huber_delta=1.0,
    )
    critic.quasimetric_model.value.data.fill_(2.0)
    info = CriticBatchInfo(
        critic=critic,
        zx=torch.zeros(3, 2),
        zy=torch.zeros(3, 2),
    )
    result = loss(_mqe_family_batch(), info)

    # Two physical samples have Huber 1.5 and one anchor has Huber 7.5.
    assert torch.isclose(result.loss, torch.tensor(3.5))
    assert torch.isclose(result.info["mixed_huber"], torch.tensor(3.5))
    assert torch.isclose(
        result.info["terminal_anchor_loss_contribution"],
        torch.tensor(2.5),
    )


def test_mqe_separate_family_normalization_preserves_physical_coefficient():
    critic = _ScalarCritic(0.0)
    loss = MQEInspiredWaypointConsistencyLoss(
        critic=critic,
        weight=1.0,
        target_tau=0.25,
        huber_delta=1.0,
        family_normalization="separate",
        terminal_anchor_loss_weight=1.0,
        diagnostic_device_names=("u_trap_target", "easy_north"),
    )
    critic.quasimetric_model.value.data.fill_(2.0)
    info = CriticBatchInfo(
        critic=critic,
        zx=torch.zeros(3, 2),
        zy=torch.zeros(3, 2),
    )
    result = loss(_mqe_family_batch(), info)

    # Preserve the old 2/3 physical coefficient, but independently normalize
    # the anchor family: 2/3 * (1.5 + 7.5) = 6.0.
    assert torch.isclose(result.loss, torch.tensor(6.0))
    assert torch.isclose(
        result.info["physical_loss_contribution"], torch.tensor(1.0)
    )
    assert torch.isclose(
        result.info["terminal_anchor_loss_contribution"], torch.tensor(5.0)
    )
    assert torch.isclose(
        result.info["terminal_anchor_u_trap_target_count"], torch.tensor(1.0)
    )
    assert torch.isclose(
        result.info["terminal_anchor_u_trap_target_huber"], torch.tensor(7.5)
    )
    assert torch.isclose(
        result.info["terminal_anchor_u_trap_target_residual"], torch.tensor(-8.0)
    )
    assert torch.isclose(
        result.info["terminal_anchor_u_trap_target_underestimate_fraction"],
        torch.tensor(1.0),
    )
    assert torch.isclose(
        result.info["terminal_anchor_easy_north_count"], torch.tensor(0.0)
    )


if __name__ == "__main__":
    test_fixed_mode_uses_step_cost()
    test_negative_reward_mode_uses_nonnegative_costs()
    test_positive_rewards_are_clipped_to_zero_and_reported()
    print("All task-aware QRL loss tests passed.")
