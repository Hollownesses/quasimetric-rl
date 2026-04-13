#!/usr/bin/env python3
"""
通信巡检执行评估可视化的轻量 smoke test。
不依赖真实 checkpoint，用一个简单的假 agent 跑评估并生成样本图，确保这条新链路以后不容易悄悄坏掉。
"""
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from minimal_qrl.envs import CommInspectionDubinsUAV2D
from minimal_qrl.eval.comm_inspection_execution_eval import (
    VisualizationConfig,
    _evaluate_comm_lookahead_sequences,
    evaluate_execution_mode,
    rollout_execution_episode,
)
from minimal_qrl.eval.dubins_execution_mode_eval import DubinsLookaheadConfig
from minimal_qrl.gc_agents import GoalConditionedAgentBase
from minimal_qrl.envs import CircleObstacle


class ZeroTurnAgent(GoalConditionedAgentBase):
    def act(self, obs: np.ndarray, goal_obs: np.ndarray, eval_mode: bool = True) -> np.ndarray:
        _ = obs, goal_obs, eval_mode
        return np.array([0.0], dtype=np.float32)

    def value(self, obs: np.ndarray, goal_obs: np.ndarray) -> float:
        _ = obs, goal_obs
        return 0.0


class DummyNavFeatures:
    def __init__(self, obs_dim: int):
        self.obs_dim = int(obs_dim)
        self.feature_dim = 0

    def build_state(self, obs: np.ndarray, goal_obs: np.ndarray) -> np.ndarray:
        return np.concatenate([obs, goal_obs], axis=0).astype(np.float32)

    def distance_to_goal(self, obs: np.ndarray, goal_obs: np.ndarray) -> float:
        return float(np.linalg.norm(np.asarray(obs[:2], dtype=np.float32) - np.asarray(goal_obs[:2], dtype=np.float32)))


class FixedActionHighLevelPolicy:
    def __init__(self, raw_action: np.ndarray, *, subgoal_max_radius: float = 1.5):
        self.raw_action = np.asarray(raw_action, dtype=np.float32)
        self.subgoal_max_radius = float(subgoal_max_radius)

    def act(self, state: np.ndarray, *, eval_mode: bool = True) -> np.ndarray:
        _ = state, eval_mode
        return self.raw_action.copy()


def make_env(**kwargs) -> CommInspectionDubinsUAV2D:
    default = dict(
        bounds=(0.0, 0.0, 10.0, 10.0),
        omega_max=1.0,
        v=1.0,
        dt=0.1,
        max_steps=12,
        observation_mode="task_context",
        inspection_target=(5.0, 5.0),
        ground_station=(1.5, 2.0),
        observation_radius=1.8,
        fov_angle=np.pi / 2.0,
        require_target_los=True,
        comm_alpha=2.0,
        comm_bias=5.0,
        comm_occlusion_penalty=6.0,
        comm_threshold=0.5,
        goal_sampling_mode="task_feasible",
        goal_position_tolerance=0.15,
        goal_heading_tolerance=0.2,
    )
    default.update(kwargs)
    return CommInspectionDubinsUAV2D(**default)


def test_execution_eval_visualization_smoke(tmp_path: Path):
    env = make_env()
    agent = ZeroTurnAgent()
    viz_cfg = VisualizationConfig(
        save_visualizations=True,
        max_successes=1,
        max_failures=1,
        save_gif=False,
        gif_fps=8,
    )

    metrics, visualizations = evaluate_execution_mode(
        agent,
        env,
        "greedy",
        n_trials=3,
        seed=7,
        lookahead_cfg=None,
        output_dir=tmp_path,
        viz_cfg=viz_cfg,
    )

    assert "success_rate" in metrics
    assert len(visualizations["success"]) <= 1
    assert len(visualizations["failure"]) <= 1

    saved_entries = visualizations["success"] + visualizations["failure"]
    assert saved_entries
    saved_png = tmp_path / saved_entries[0]["png"]
    assert saved_png.exists()


def test_hierarchical_execution_eval_smoke(tmp_path: Path):
    env = make_env()
    agent = ZeroTurnAgent()
    policy = FixedActionHighLevelPolicy(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    nav_features = DummyNavFeatures(int(env.observation_space.shape[0]))
    viz_cfg = VisualizationConfig(
        save_visualizations=False,
        max_successes=0,
        max_failures=0,
        save_gif=False,
        gif_fps=8,
    )
    lookahead_cfg = DubinsLookaheadConfig(
        horizon=4,
        num_sequences=16,
        biased_sequences=4,
        alpha_subgoal=1.0,
        alpha_final=0.3,
        alpha_task_terminal=0.5,
        use_env_stage_cost=True,
    )

    metrics, visualizations = evaluate_execution_mode(
        agent,
        env,
        "hierarchical",
        n_trials=2,
        seed=11,
        lookahead_cfg=lookahead_cfg,
        output_dir=tmp_path,
        viz_cfg=viz_cfg,
        high_level_policy=policy,
        nav_features=nav_features,
        high_level_period=3,
    )

    assert "raw_actor_output_valid_rate" in metrics
    assert "mean_repair_distance" in metrics
    assert visualizations == {"success": [], "failure": []}


def test_hierarchical_metrics_and_rollout_recording(tmp_path: Path):
    env = make_env(
        start=(9.7, 9.7, 0.0),
        goal=(8.5, 8.5, 0.0),
        goal_sampling_mode="valid",
    )
    agent = ZeroTurnAgent()
    policy = FixedActionHighLevelPolicy(np.array([1.0, 0.0, 0.0], dtype=np.float32), subgoal_max_radius=2.0)
    nav_features = DummyNavFeatures(int(env.observation_space.shape[0]))
    lookahead_cfg = DubinsLookaheadConfig(
        horizon=3,
        num_sequences=8,
        biased_sequences=2,
        alpha_subgoal=1.0,
        alpha_final=0.3,
        alpha_task_terminal=0.5,
        use_env_stage_cost=True,
    )
    viz_cfg = VisualizationConfig(
        save_visualizations=True,
        max_successes=1,
        max_failures=1,
        save_gif=False,
        gif_fps=8,
    )

    rollout = rollout_execution_episode(
        agent,
        env,
        "hierarchical",
        episode_seed=5,
        lookahead_cfg=lookahead_cfg,
        high_level_policy=policy,
        nav_features=nav_features,
        high_level_period=2,
    )
    assert len(rollout["high_level_events"]) >= 2
    assert {"raw_subgoal", "repaired_subgoal", "executed_subgoal"} <= set(rollout["high_level_events"][0].keys())
    assert "raw_action" in rollout["high_level_events"][0]

    metrics, visualizations = evaluate_execution_mode(
        agent,
        env,
        "hierarchical",
        n_trials=1,
        seed=5,
        lookahead_cfg=lookahead_cfg,
        output_dir=tmp_path,
        viz_cfg=viz_cfg,
        high_level_policy=policy,
        nav_features=nav_features,
        high_level_period=2,
    )

    assert metrics["raw_actor_output_valid_rate"] == 0.0
    assert metrics["mean_repair_distance"] > 0.0
    saved_entries = visualizations["success"] + visualizations["failure"]
    assert saved_entries
    assert saved_entries[0]["high_level_events"]
    assert "raw_subgoal" in saved_entries[0]["high_level_events"][0]
    assert (tmp_path / saved_entries[0]["png"]).exists()


def test_planner_uses_env_stage_cost_without_duplicate_collision_penalty():
    env = CommInspectionDubinsUAV2D(
        bounds=(0.0, 0.0, 10.0, 10.0),
        omega_max=1.0,
        v=1.0,
        dt=0.1,
        max_steps=8,
        observation_mode="task_context",
        obstacles=[CircleObstacle(x=4.16, y=5.0, radius=0.12)],
        start=(4.0, 5.0, 0.0),
        goal=(6.0, 5.0, 0.0),
        inspection_target=(5.0, 5.0),
        ground_station=(1.0, 5.0),
        observation_radius=2.0,
        fov_angle=np.pi / 2.0,
        require_target_los=True,
        comm_alpha=2.0,
        comm_bias=5.0,
        comm_occlusion_penalty=8.0,
        comm_threshold=1.0,
        goal_sampling_mode="valid",
    )
    agent = ZeroTurnAgent()
    lookahead_cfg = DubinsLookaheadConfig(
        horizon=1,
        num_sequences=1,
        collision_penalty=123.0,
        alpha_subgoal=0.0,
        alpha_final=0.0,
        alpha_task_terminal=0.0,
        use_env_stage_cost=True,
    )

    env.reset(seed=0)
    _obs, _reward, _done, _truncated, info = env.step(np.array([0.0], dtype=np.float32))
    expected_cost = float(info["cost_total"])

    env.reset(seed=0)
    goal_obs = env.state_to_observation(np.asarray(env.goal, dtype=np.float32))
    costs, _ = _evaluate_comm_lookahead_sequences(
        agent,
        env,
        goal_obs,
        lookahead_cfg,
        np.asarray([[0.0]], dtype=np.float32),
        env.get_state(),
    )
    assert np.isclose(float(costs[0]), expected_cost, atol=1e-6)


if __name__ == "__main__":
    test_execution_eval_visualization_smoke(Path("results/test_comm_inspection_execution_eval"))
    test_hierarchical_execution_eval_smoke(Path("results/test_comm_inspection_execution_eval_hier"))
    test_hierarchical_metrics_and_rollout_recording(Path("results/test_comm_inspection_execution_eval_hier_metrics"))
    test_planner_uses_env_stage_cost_without_duplicate_collision_penalty()
    print("Execution eval visualization smoke test passed.")
