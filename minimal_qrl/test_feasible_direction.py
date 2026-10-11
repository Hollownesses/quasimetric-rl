"""Checks the QP boundary reaction and the complete finite critic step."""

from types import SimpleNamespace

import torch

from quasimetric_rl.data import BatchData
from quasimetric_rl.data.env_spec import EnvSpec
from quasimetric_rl.modules.utils import LossResult
from quasimetric_rl.modules.quasimetric_critic.losses import CriticBatchInfo
from quasimetric_rl.modules.quasimetric_critic.models.encoder import Encoder
from quasimetric_rl.modules.quasimetric_critic.models.quasimetric_model import QuasimetricModel
from quasimetric_rl.modules.quasimetric_critic.losses.feasible_direction import (
    FeasibleDirectionConf, FeasibleDirectionUpdater, solve_direction,
)
from quasimetric_rl.modules.quasimetric_critic.losses.goal_set_barrier import (
    GoalSetBarrierConf, GoalSetBarrierUpdater,
)
from quasimetric_rl.modules.quasimetric_critic.losses.local_constraint import LocalConstraintLoss
from minimal_qrl.test_comm_inspection_dubins_uav import make_env


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


def test_goal_set_quotient_makes_real_terminal_anchor_exact_and_survives_checkpoint():
    env = make_env()
    env.set_task_by_device_id("tank_01")
    terminal = torch.from_numpy(env.state_to_observation(
        env.sample_task_terminal_state(seed=7))).unsqueeze(0)
    goal = torch.from_numpy(env.abstract_goal_observation()).unsqueeze(0)
    spec = EnvSpec.from_env(env)
    encoder = Encoder(env_spec=spec, arch=(16,), latent_size=8)
    encoder.enable_comm_goal_set_quotient()
    head = QuasimetricModel(input_size=8, projector_arch=(16,),
                            quasimetric_head_spec="iqe(dim=32,components=4)")
    assert torch.equal(encoder._canonicalize_comm_goal_set(terminal), goal)
    assert torch.equal(encoder(terminal), encoder(goal))
    assert float(head(encoder(terminal), encoder(goal)).detach()) == 0.0

    restored = Encoder(env_spec=spec, arch=(16,), latent_size=8)
    restored.load_state_dict(encoder.state_dict())
    assert bool(restored.comm_goal_set_quotient)
    assert torch.equal(restored(terminal), restored(goal))

    legacy = {k: v for k, v in encoder.state_dict().items()
              if k != "comm_goal_set_quotient"}
    plain = Encoder(env_spec=spec, arch=(16,), latent_size=8)
    plain.load_state_dict(legacy)
    assert not bool(plain.comm_goal_set_quotient)


def test_audit_cache_prioritizes_real_edges_over_satisfied_zero_cost_anchors():
    class _SourceCritic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.value = torch.nn.Parameter(torch.tensor(0.9))

        def forward(self, x, y):
            return x[:, 0] * self.value

    class _TwoEdgeReplay:
        raw_data = SimpleNamespace(
            all_observations=torch.tensor([[0.0], [0.0], [1.0], [0.0]]),
            rewards=torch.tensor([0.0, -1.0]))
        indices_to_episode_indices = torch.tensor([0, 1])

        def __len__(self):
            return 2

    replay = _TwoEdgeReplay()
    updater = FeasibleDirectionUpdater(FeasibleDirectionConf(work_edges=1))
    updater.set_replay(replay)
    local = LocalConstraintLoss(epsilon=0.25, step_cost=1.0,
                                cost_source="negative_reward",
                                init_lagrange_multiplier=0.01,
                                mode="feasible_direction")
    stats = updater.audit(_SourceCritic(), local)
    assert updater.audit_high_indices.tolist() == [1]
    assert updater.audit_near_indices.tolist() == [1]
    assert stats["audit_anchor_violation_fraction"] == 0.0
    assert stats["audit_real_violation_fraction"] == 0.0


def test_goal_set_barrier_has_boundary_reaction_without_dual_network():
    critic, _, data, info, local = _fixture(1.0, 1.0)
    updater = GoalSetBarrierUpdater(GoalSetBarrierConf(
        weight=5.0, work_edges=1, candidate_multiplier=1,
        audit_interval=1, topk=1))
    updater.set_replay(_Replay(data.rewards))
    result = updater.step(data, info, _Push(), local)
    result.loss.backward()
    assert float(critic.value.grad) > 0.0
    assert result.info["goal_set_barrier"]["top_penalty"] > 0.0


def test_goal_set_barrier_caches_observed_violations():
    _, _, data, info, local = _fixture(1.1, 1.0)
    updater = GoalSetBarrierUpdater(GoalSetBarrierConf(
        work_edges=1, candidate_multiplier=1, audit_interval=1))
    updater.set_replay(_Replay(data.rewards))
    updater.step(data, info, _Push(), local)
    assert updater.history_indices.tolist() == [0]


def test_goal_set_barrier_workset_keeps_full_high_violation_cache_quota():
    class _LargeReplay:
        raw_data = SimpleNamespace(
            all_observations=torch.zeros(101, 2),
            rewards=-torch.ones(100))
        indices_to_episode_indices = torch.zeros(100, dtype=torch.long)

        def __len__(self):
            return 100

    updater = GoalSetBarrierUpdater(GoalSetBarrierConf(work_edges=64))
    updater.set_replay(_LargeReplay())
    updater.audit_high_indices = torch.arange(40, 72)
    local = LocalConstraintLoss(epsilon=0.25, step_cost=1.0,
                                cost_source="negative_reward",
                                init_lagrange_multiplier=0.01,
                                mode="goal_set_barrier")
    indices = updater._workset(_ScalarCritic(0.0), local)
    assert set(range(40, 72)).issubset(set(indices.tolist()))


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


def test_restoration_uses_its_own_step_size():
    critic, updater, data, info, local = _fixture(1.0, 0.0)
    updater.conf.step_size = 1e-5
    updater.conf.restoration_step_size = 1.0
    result = updater.step(data, info, _Push(), local)
    stats = result.info["feasible_direction"]
    assert stats["nominal_step_size"] == 1.0
    assert stats["accepted"] == 1.0
    assert float(critic.value.detach()) < 0.1


def test_restoration_retains_push_ascent_on_unconstrained_parameter():
    class _FreeCritic(_ScalarCritic):
        def __init__(self):
            super().__init__(0.1)
            self.free = torch.nn.Parameter(torch.tensor(0.0))

    class _TwoParameterPush:
        def __call__(self, data, batch_info):
            return LossResult(loss=-(batch_info.critic.value +
                                     batch_info.critic.free), info={})

    _, updater, data, _, local = _fixture(0.1, 0.0)
    critic = _FreeCritic()
    info = CriticBatchInfo(critic=critic, zx=data.observations,
                           zy=data.next_observations)
    result = updater.step(data, info, _TwoParameterPush(), local)
    stats = result.info["feasible_direction"]
    assert stats["restoration"] == 1.0
    assert stats["accepted"] == 1.0
    assert float(critic.value.detach()) < 0.1
    assert float(critic.free.detach()) > 0.0
    assert stats["actual_ascent"] > 0.0


def test_restoration_falls_back_when_push_curvature_blocks_finite_step():
    class _CurvedCritic(_ScalarCritic):
        def __init__(self):
            super().__init__(0.1)
            self.free = torch.nn.Parameter(torch.tensor(0.0))

        def _distance(self, x, y):
            return (self.value + 100.0 * self.free.square()).expand(x.shape[0])

    class _FreeOnlyPush:
        def __call__(self, data, batch_info):
            return LossResult(loss=-batch_info.critic.free, info={})

    _, updater, data, _, local = _fixture(0.1, 0.0)
    critic = _CurvedCritic()
    info = CriticBatchInfo(critic=critic, zx=data.observations,
                           zy=data.next_observations)
    result = updater.step(data, info, _FreeOnlyPush(), local)
    stats = result.info["feasible_direction"]
    assert stats["restoration_fallback"] == 1.0
    assert stats["accepted"] == 1.0
    assert stats["backtrack_rejections"] >= 6.0
    assert float(critic.value.detach()) < 0.1
    assert float(critic.free.detach()) == 0.0


def test_feasible_ascent_uses_regular_step_size():
    critic, updater, data, info, local = _fixture(0.0, 1.0)
    updater.conf.step_size = 0.001
    updater.conf.restoration_step_size = 1.0
    result = updater.step(data, info, _Push(), local)
    stats = result.info["feasible_direction"]
    assert stats["restoration"] == 0.0
    assert stats["nominal_step_size"] == 0.001
    assert abs(float(critic.value.detach()) - 0.001) < 1e-6


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
