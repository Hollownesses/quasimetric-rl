#!/usr/bin/env python3
"""训练期 QRL 价值评估的距离尺度回归测试。"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from minimal_qrl.eval.evaluation import evaluate_quasimetric, evaluation_reference


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


def test_goal_set_evaluation_snapshots_source_and_goal_in_their_task_context():
    class SwitchingTaskEnv(GoalSetEnvWithLegacyDtScale):
        def __init__(self):
            self.source_buffer = np.zeros(2, dtype=np.float32)
            self.goal_buffer = np.zeros(2, dtype=np.float32)

        def reset(self, seed=None):
            self.task = seed % 2

        def state_to_observation(self, state):
            self.source_buffer[:] = [state[0], self.task]
            return self.source_buffer

        def abstract_goal_observation(self):
            self.goal_buffer[:] = [10., self.task]
            return self.goal_buffer

    class ContextCheckingDistance(AbsoluteDistance):
        def forward(self, source, goal):
            assert torch.equal(source[:, 1], goal[:, 1])
            assert source[:, 1].unique().numel() == 2
            return super().forward(source, goal)

    agent = DummyAgent()
    agent.critics[0].quasimetric_model = ContextCheckingDistance()
    metrics = evaluate_quasimetric(agent, SwitchingTaskEnv(), n_pairs=8, seed=7, distance_scale=1.)
    assert np.isclose(metrics['mae'], 0.)
    assert np.isclose(metrics['spearman_corr'], 1.)


def test_proxy_and_complete_path_reference_are_explicitly_distinct():
    env = GoalSetEnvWithLegacyDtScale()
    proxy = evaluation_reference(env)
    path = evaluation_reference(env, oracle_bank=True)
    assert proxy['tensorboard_prefix'] == 'eval_geometry_proxy'
    assert not proxy['full_environment_cost']
    assert proxy['target_unit'] != proxy['prediction_unit']
    assert path['tensorboard_prefix'] == 'eval_oracle'
    assert path['full_environment_cost']
    assert path['target_unit'] == path['prediction_unit']
    assert not proxy['optimality_guaranteed'] and not path['optimality_guaranteed']
    assert evaluation_reference(object())['tensorboard_prefix'] == 'eval'
