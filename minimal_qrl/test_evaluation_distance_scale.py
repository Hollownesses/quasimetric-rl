#!/usr/bin/env python3
"""训练期 QRL 价值评估的距离尺度回归测试。"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from minimal_qrl.eval.evaluation import evaluate_quasimetric


class IdentityEncoder(torch.nn.Module):
    def forward(self, value):
        return value


class AbsoluteDistance(torch.nn.Module):
    def forward(self, source, goal):
        return torch.abs(source[:, 0] - goal[:, 0])


class DummyCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = IdentityEncoder()
        self.quasimetric_model = AbsoluteDistance()


class DummyAgent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.critics = torch.nn.ModuleList([DummyCritic()])


class GoalSetEnvWithLegacyDtScale:
    def reset(self, seed=None):
        return None

    def sample_nonterminal_valid_state(self, seed=None):
        rng = np.random.default_rng(seed)
        return np.array([rng.uniform(0.0, 5.0), 0.0], dtype=np.float32)

    def abstract_goal_observation(self):
        return np.array([10.0, 0.0], dtype=np.float32)

    def sample_task_terminal_state(self, seed=None):
        return np.array([10.0, 0.0], dtype=np.float32)

    def compute_goal_reaching_cost_estimate(self, state, goal):
        return float(abs(goal[0] - state[0]))

    def state_to_observation(self, state):
        return np.asarray(state, dtype=np.float32)

    def get_distance_scale(self):
        return 0.1


def test_explicit_unit_scale_bypasses_legacy_dt_scale():
    agent = DummyAgent()
    env = GoalSetEnvWithLegacyDtScale()

    legacy_metrics = evaluate_quasimetric(
        agent,
        env,
        n_pairs=8,
        seed=7,
    )
    cost_unit_metrics = evaluate_quasimetric(
        agent,
        env,
        n_pairs=8,
        seed=7,
        distance_scale=1.0,
    )

    assert np.isclose(
        cost_unit_metrics["pred_mean"],
        legacy_metrics["pred_mean"] / env.get_distance_scale(),
    )
    assert np.isclose(cost_unit_metrics["mae"], 0.0)
