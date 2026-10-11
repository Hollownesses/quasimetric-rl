"""Checks the QP boundary reaction and the complete finite critic step."""

from types import SimpleNamespace

import torch

from quasimetric_rl.data import BatchData
from quasimetric_rl.modules.utils import LossResult
from quasimetric_rl.modules.quasimetric_critic.losses import CriticBatchInfo
from quasimetric_rl.modules.quasimetric_critic.losses.feasible_direction import (
    FeasibleDirectionConf, FeasibleDirectionUpdater, solve_direction,
)
from quasimetric_rl.modules.quasimetric_critic.losses.local_constraint import LocalConstraintLoss


def test_two_edge_boundary_qp_has_current_jacobian_reaction():
    g = torch.tensor([0.5, 0.5])
    A = torch.tensor([[1.0, -1.0], [0.0, 1.0]])
    v, alpha, slack = solve_direction(g, A, torch.zeros(2), kappa=1.0)
    assert torch.allclose(v, torch.zeros_like(v), atol=1e-5)
    assert torch.allclose(alpha, torch.tensor([0.5, 1.0]), atol=1e-5)
    assert float(slack.max()) < 1e-5


def test_degenerate_anchor_reports_elastic_slack():
    v, alpha, slack = solve_direction(
        torch.zeros(1), torch.zeros(1, 1), torch.tensor([0.2]),
        kappa=1.0, elastic_penalty=10.0,
    )
    assert torch.allclose(v, torch.zeros_like(v))
    assert torch.allclose(alpha, torch.zeros_like(alpha))
    assert torch.allclose(slack, torch.tensor([0.2]))


class _Replay:
    def __init__(self, rewards):
        self.raw_data = SimpleNamespace(
            all_observations=torch.zeros(2, 2), rewards=rewards)
        self.indices_to_episode_indices = torch.zeros(1, dtype=torch.long)

    def __len__(self):
        return 1


class _ScalarCritic(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(float(value)))
        self.encoder = torch.nn.Identity()
        self.quasimetric_model = self._distance

    def _distance(self, x, y):
        return self.value.expand(x.shape[0])

    def forward(self, x, y):
        return self._distance(x, y)


class _QuadraticCritic(_ScalarCritic):
    def _distance(self, x, y):
        return self.value.square().expand(x.shape[0])


class _Push:
    def __call__(self, data, batch_info):
        return LossResult(loss=-batch_info.critic.value, info={})


def _fixture(value, cost):
    critic = _ScalarCritic(value)
    obs = torch.zeros(1, 2)
    data = BatchData(
        observations=obs, actions=torch.zeros(1, 1), next_observations=obs,
        rewards=torch.tensor([-cost]), terminals=torch.zeros(1, dtype=torch.bool),
        timeouts=torch.zeros(1, dtype=torch.bool), future_observations=obs,
    )
    replay = _Replay(data.rewards)
    local = LocalConstraintLoss(
        epsilon=0.25, step_cost=1.0, cost_source="negative_reward",
        init_lagrange_multiplier=0.01, mode="feasible_direction",
    )
    updater = FeasibleDirectionUpdater(FeasibleDirectionConf(
        work_edges=1, candidate_multiplier=1, audit_interval=1,
        step_size=0.1, trust_radius=1.0,
    ))
    updater.set_replay(replay)
    info = CriticBatchInfo(critic=critic, zx=obs, zy=obs)
    return critic, updater, data, info, local


def test_boundary_step_does_not_escape_cost_constraint():
    critic, updater, data, info, local = _fixture(1.0, 1.0)
    result = updater.step(data, info, _Push(), local)
    assert float(critic.value.detach()) <= 1.0 + 1e-6
    assert result.info["feasible_direction"]["qp_slack_max"] < 1e-5


def test_infeasible_zero_cost_anchor_enters_restoration():
    critic, updater, data, info, local = _fixture(0.1, 0.0)
    result = updater.step(data, info, _Push(), local)
    stats = result.info["feasible_direction"]
    assert stats["restoration"] == 1.0
    assert stats["accepted"] == 1.0
    assert 0.0 <= float(critic.value.detach()) < 0.1
    assert stats["work_anchor_count"] == 1.0


def test_nonlinear_residual_is_rechecked_during_backtracking():
    _, updater, data, _, local = _fixture(0.9, 1.0)
    critic = _QuadraticCritic(0.9)
    info = CriticBatchInfo(critic=critic, zx=data.observations,
                           zy=data.next_observations)
    updater.conf.step_size = 2.0
    updater.conf.trust_radius = 10.0
    result = updater.step(data, info, _Push(), local)
    stats = result.info["feasible_direction"]
    assert stats["accepted"] == 1.0
    assert stats["backtrack_rejections"] >= 1.0
    assert float(critic.value.detach().square()) <= 1.0 + 1e-6
