#!/usr/bin/env python3
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn

from minimal_qrl.envs import CommInspectionDubinsUAV2D
from minimal_qrl.high_level_policy import (
    CostAwareSubgoalPolicy,
    FrozenQRLNavigationFeatures,
    decode_relative_subgoal,
)


class DummyEncoder(nn.Module):
    def __init__(self, obs_dim: int, latent_dim: int):
        super().__init__()
        self.linear = nn.Linear(obs_dim, latent_dim, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                    ],
                    dtype=torch.float32,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class DummyQuasimetric(nn.Module):
    def forward(self, zx: torch.Tensor, zy: torch.Tensor) -> torch.Tensor:
        return torch.norm(zx - zy, dim=-1)


class DummyCritic(nn.Module):
    def __init__(self, obs_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = DummyEncoder(obs_dim, latent_dim)
        self.quasimetric_model = DummyQuasimetric()


class DummyQRLAgent(nn.Module):
    def __init__(self, obs_dim: int, latent_dim: int):
        super().__init__()
        self.critics = nn.ModuleList([DummyCritic(obs_dim, latent_dim)])


def _make_env(**kwargs) -> CommInspectionDubinsUAV2D:
    default = dict(
        bounds=(0.0, 0.0, 10.0, 10.0),
        omega_max=1.0,
        v=1.0,
        dt=0.1,
        max_steps=20,
        observation_mode="cos_sin",
        start=(9.7, 9.7, 0.0),
        goal=(8.0, 8.0, 0.0),
        inspection_target=(5.0, 5.0),
        ground_station=(1.5, 2.0),
        observation_radius=1.8,
        fov_angle=np.pi / 2.0,
        require_target_los=True,
        comm_alpha=2.0,
        comm_bias=5.0,
        comm_occlusion_penalty=6.0,
        comm_threshold=0.5,
        goal_sampling_mode="valid",
        goal_position_tolerance=0.15,
        goal_heading_tolerance=0.2,
    )
    default.update(kwargs)
    return CommInspectionDubinsUAV2D(**default)


def test_frozen_qrl_navigation_features_shape_and_values():
    obs_dim = 4
    latent_dim = 3
    qrl_agent = DummyQRLAgent(obs_dim=obs_dim, latent_dim=latent_dim)
    features = FrozenQRLNavigationFeatures(
        qrl_agent,
        obs_dim=obs_dim,
        device=torch.device("cpu"),
        critic_index=0,
        use_distance=True,
        use_latent=True,
    )

    obs = np.array([1.0, 2.0, 0.5, -0.5], dtype=np.float32)
    goal = np.array([4.0, 6.0, -0.5, 0.5], dtype=np.float32)
    high_state = features.build_state(obs, goal)

    expected_dim = 2 * obs_dim + 1 + 3 * latent_dim
    assert high_state.shape == (expected_dim,)
    z_t = np.array([1.0, 2.0, 0.5], dtype=np.float32)
    z_g = np.array([4.0, 6.0, -0.5], dtype=np.float32)
    expected_distance = np.linalg.norm(z_t - z_g)
    assert np.isclose(high_state[2 * obs_dim], expected_distance, atol=1e-6)
    assert np.allclose(high_state[-3:], z_g - z_t, atol=1e-6)


def test_frozen_qrl_features_do_not_require_grad_or_mutate_qrl_params():
    obs_dim = 4
    qrl_agent = DummyQRLAgent(obs_dim=obs_dim, latent_dim=3)
    before = [param.detach().clone() for param in qrl_agent.parameters()]
    features = FrozenQRLNavigationFeatures(
        qrl_agent,
        obs_dim=obs_dim,
        device=torch.device("cpu"),
        critic_index=0,
        use_distance=True,
        use_latent=True,
    )

    state_t = features.build_state_tensor(
        np.zeros((2, obs_dim), dtype=np.float32),
        np.ones((2, obs_dim), dtype=np.float32),
    )
    assert state_t.requires_grad is False

    after = [param.detach().clone() for param in qrl_agent.parameters()]
    for lhs, rhs in zip(before, after):
        assert torch.equal(lhs, rhs)


def test_decode_relative_subgoal_repairs_to_valid_state():
    env = _make_env()
    env.reset(seed=0)
    choice = decode_relative_subgoal(
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        env,
        subgoal_max_radius=2.0,
    )
    assert not choice["raw_valid"]
    assert env.is_valid_state(choice["executed_subgoal"])
    assert choice["repair_distance"] > 0.0


def test_high_level_sac_update_uses_segment_discount():
    policy = CostAwareSubgoalPolicy(
        state_dim=6,
        action_dim=3,
        hidden_dim=16,
        actor_lr=1e-3,
        critic_lr=1e-3,
        tau=0.01,
        init_alpha=0.2,
        device=torch.device("cpu"),
    )

    for module in [policy.q1, policy.q2]:
        for param in module.parameters():
            nn.init.constant_(param, 0.0)

    def _sample_zero(state: torch.Tensor, deterministic: bool = False):
        del deterministic
        batch = state.shape[0]
        action = torch.zeros((batch, policy.action_dim), dtype=state.dtype, device=state.device)
        log_prob = torch.zeros((batch,), dtype=state.dtype, device=state.device)
        return action, log_prob

    def _constant_target(state: torch.Tensor, action: torch.Tensor):
        del action
        return torch.full((state.shape[0],), 2.0, dtype=state.dtype, device=state.device)

    policy.actor.sample = _sample_zero  # type: ignore[method-assign]
    policy.q1_targ.forward = _constant_target  # type: ignore[method-assign]
    policy.q2_targ.forward = _constant_target  # type: ignore[method-assign]

    batch = {
        "state": torch.zeros((4, 6), dtype=torch.float32),
        "action": torch.zeros((4, 3), dtype=torch.float32),
        "reward": torch.ones((4,), dtype=torch.float32),
        "next_state": torch.zeros((4, 6), dtype=torch.float32),
        "done": torch.zeros((4,), dtype=torch.float32),
        "discount": torch.full((4,), 0.25, dtype=torch.float32),
        "segment_len": torch.full((4,), 3.0, dtype=torch.float32),
    }
    metrics = policy.update(batch)

    expected_target = 1.0 + 0.25 * 2.0
    expected_q_loss = expected_target ** 2
    assert np.isclose(metrics["mean_discount"], 0.25, atol=1e-6)
    assert np.isclose(metrics["q1_loss"], expected_q_loss, atol=1e-6)
    assert np.isclose(metrics["q2_loss"], expected_q_loss, atol=1e-6)
