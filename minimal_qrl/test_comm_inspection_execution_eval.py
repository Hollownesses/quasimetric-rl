#!/usr/bin/env python3
"""
通信巡检执行评估可视化的轻量 smoke test。
不依赖真实 checkpoint，用一个简单的假 agent 跑评估并生成样本图，确保这条新链路以后不容易悄悄坏掉。
"""
import os
import json
import math
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

import minimal_qrl.eval.comm_inspection_execution_eval as comm_eval
from minimal_qrl.envs import CommInspectionDubinsUAV2D
from minimal_qrl.eval.comm_inspection_execution_eval import (
    VisualizationConfig,
    _evaluate_comm_lookahead_sequences,
    evaluate_execution_mode,
    rollout_execution_episode,
)
from minimal_qrl.comm_inspection_planner import INVALID_ROLLOUT_COST
from minimal_qrl.eval.dubins_execution_mode_eval import DubinsLookaheadConfig
from minimal_qrl.gc_agents import GoalConditionedAgentBase, QRLGoalValueAdapter
from minimal_qrl.subgoal_actor import SubgoalActor
from minimal_qrl.envs import CircleObstacle


class ZeroTurnAgent(GoalConditionedAgentBase):
    def act(self, obs: np.ndarray, goal_obs: np.ndarray, eval_mode: bool = True) -> np.ndarray:
        _ = obs, goal_obs, eval_mode
        return np.array([0.0], dtype=np.float32)

    def value(self, obs: np.ndarray, goal_obs: np.ndarray) -> float:
        _ = obs, goal_obs
        return 0.0


class DistanceToGoalAgent(GoalConditionedAgentBase):
    def act(self, obs: np.ndarray, goal_obs: np.ndarray, eval_mode: bool = True) -> np.ndarray:
        _ = obs, goal_obs, eval_mode
        return np.array([0.0], dtype=np.float32)

    def value(self, obs: np.ndarray, goal_obs: np.ndarray) -> float:
        return float(np.linalg.norm(np.asarray(obs[:2], dtype=np.float32) - np.asarray(goal_obs[:2], dtype=np.float32)))

    def batch_value(self, obs_batch: np.ndarray, goal_obs_batch: np.ndarray) -> np.ndarray:
        return np.linalg.norm(
            np.asarray(obs_batch[:, :2], dtype=np.float32) - np.asarray(goal_obs_batch[:, :2], dtype=np.float32),
            axis=1,
        ).astype(np.float32)


class FixedInvalidSubgoalActor:
    def __init__(self, raw_state: np.ndarray):
        self.raw_state = np.asarray(raw_state, dtype=np.float32)

    def predict_state(
        self,
        obs: np.ndarray,
        goal_obs: np.ndarray,
        env: CommInspectionDubinsUAV2D,
        device: torch.device,
    ) -> np.ndarray:
        _ = obs, goal_obs, env, device
        return self.raw_state.copy()


class CountingBatchAdapter(QRLGoalValueAdapter):
    def __init__(self, env):
        torch.nn.Module.__init__(self)
        self.env = env
        self.calls = 0

    def value(self, obs, goal_obs):
        raise AssertionError("greedy action selection must not make scalar value calls")

    def batch_value(self, obs_batch, goal_obs_batch):
        self.calls += 1
        return np.asarray(obs_batch[:, 0], dtype=np.float32)


def make_env(**kwargs) -> CommInspectionDubinsUAV2D:
    catalog = {
        "ground_station": {"position": [1.5, 2.0], "los_anchor": [1.5, 2.0]},
        "devices": [
            {
                "id": "tank_01",
                "position": [5.0, 5.0],
                "observation_anchor": [5.0, 5.0],
                "observation": {
                    "min_distance": 0.5,
                    "max_distance": 1.8,
                    "preferred_bearing_rad": math.pi,
                    "bearing_tolerance_rad": math.pi,
                    "fov_angle_rad": math.pi / 2.0,
                    "require_los": False,
                },
            },
            {
                "id": "pipe_02",
                "position": [7.0, 7.0],
                "observation_anchor": [7.0, 7.0],
                "observation": {
                    "min_distance": 0.5,
                    "max_distance": 1.5,
                    "preferred_bearing_rad": -math.pi / 2.0,
                    "bearing_tolerance_rad": math.pi,
                    "fov_angle_rad": math.pi / 2.0,
                    "require_los": False,
                },
            },
        ],
    }
    default = dict(
        device_catalog=catalog,
        bounds=(0.0, 0.0, 10.0, 10.0),
        omega_max=1.0,
        v=1.0,
        dt=0.1,
        max_steps=12,
        comm_alpha=2.0,
        comm_bias=5.0,
        comm_occlusion_penalty=6.0,
        comm_threshold=0.5,
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
        starts_per_device=3,
        seed=7,
        lookahead_cfg=None,
        output_dir=tmp_path,
        viz_cfg=viz_cfg,
    )

    assert "success_rate" in metrics
    assert metrics["num_trials"] == 6.0
    assert set(metrics["per_device"]) == {"tank_01", "pipe_02"}
    assert all(row["num_trials"] == 3 for row in metrics["per_device"].values())
    assert "decision_time_p95_sec" in metrics
    assert len(visualizations["success"]) <= 1
    assert len(visualizations["failure"]) <= 1

    saved_entries = visualizations["success"] + visualizations["failure"]
    assert saved_entries
    saved_png = tmp_path / saved_entries[0]["png"]
    assert saved_png.exists()


def test_qrl_greedy_batches_all_candidate_values_once():
    env = make_env(start=(2.0, 5.0, 0.0))
    obs, _ = env.reset(seed=0, options={"device_id": "tank_01"})
    agent = CountingBatchAdapter(env)
    action = agent.act(obs, env.abstract_goal_observation(), eval_mode=True)
    assert action.shape == (1,)
    assert agent.calls == 1


def test_hierarchical_execution_eval_smoke(tmp_path: Path):
    env = make_env()
    agent = ZeroTurnAgent()
    actor = SubgoalActor(obs_dim=int(env.observation_space.shape[0]), hidden_dim=32)
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
        starts_per_device=2,
        seed=11,
        lookahead_cfg=lookahead_cfg,
        output_dir=tmp_path,
        viz_cfg=viz_cfg,
        subgoal_actor=actor,
        actor_device=torch.device("cpu"),
        high_level_period=3,
        subgoal_candidates=16,
        subgoal_lambda_final=0.3,
        subgoal_lambda_task=1.0,
    )

    assert "raw_actor_output_valid_rate" in metrics
    assert "mean_repair_distance" in metrics
    assert visualizations == {"success": [], "failure": []}


def test_hierarchical_metrics_and_rollout_recording(tmp_path: Path):
    env = make_env()
    agent = ZeroTurnAgent()
    actor = FixedInvalidSubgoalActor(np.array([11.0, 11.0, 0.0], dtype=np.float32))
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
        subgoal_actor=actor,
        actor_device=torch.device("cpu"),
        high_level_period=2,
        subgoal_candidates=8,
        subgoal_lambda_final=0.3,
        subgoal_lambda_task=1.0,
    )
    assert len(rollout["high_level_events"]) >= 2
    assert {"raw_subgoal", "repaired_subgoal", "executed_subgoal"} <= set(rollout["high_level_events"][0].keys())

    metrics, visualizations = evaluate_execution_mode(
        agent,
        env,
        "hierarchical",
        starts_per_device=1,
        seed=5,
        lookahead_cfg=lookahead_cfg,
        output_dir=tmp_path,
        viz_cfg=viz_cfg,
        subgoal_actor=actor,
        actor_device=torch.device("cpu"),
        high_level_period=2,
        subgoal_candidates=8,
        subgoal_lambda_final=0.3,
        subgoal_lambda_task=1.0,
    )

    assert metrics["raw_actor_output_valid_rate"] == 0.0
    assert np.isclose(metrics["mean_repair_distance"], np.sqrt(2.0), atol=1e-4)
    saved_entries = visualizations["success"] + visualizations["failure"]
    assert saved_entries
    assert saved_entries[0]["high_level_events"]
    assert "raw_subgoal" in saved_entries[0]["high_level_events"][0]
    assert (tmp_path / saved_entries[0]["png"]).exists()


def test_planner_rejects_collision_rollouts_even_with_env_stage_cost():
    env = CommInspectionDubinsUAV2D(
        device_catalog=make_env().device_catalog,
        bounds=(0.0, 0.0, 10.0, 10.0),
        omega_max=1.0,
        v=1.0,
        dt=0.1,
        max_steps=8,
        obstacles=[CircleObstacle(x=4.16, y=5.0, radius=0.12)],
        start=(4.0, 5.0, 0.0),
        comm_alpha=2.0,
        comm_bias=5.0,
        comm_occlusion_penalty=8.0,
        comm_threshold=1.0,
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
    goal_obs = env.abstract_goal_observation()
    costs, _ = _evaluate_comm_lookahead_sequences(
        agent,
        env,
        goal_obs,
        lookahead_cfg,
        np.asarray([[0.0]], dtype=np.float32),
        env.get_state(),
    )
    assert float(costs[0]) >= INVALID_ROLLOUT_COST + expected_cost - 0.1


def test_terminal_heuristic_ignores_dense_progress_term():
    env = make_env(
        start=(2.0, 5.0, 0.0),
    )
    agent = DistanceToGoalAgent()
    lookahead_cfg = DubinsLookaheadConfig(
        horizon=1,
        num_sequences=2,
        biased_sequences=0,
        alpha_subgoal=0.0,
        alpha_final=0.0,
        alpha_task_terminal=0.0,
        use_env_stage_cost=False,
        heuristic_mode="terminal",
        qrl_progress_alpha=1000.0,
    )

    env.reset(seed=0)
    goal_obs = env.abstract_goal_observation()
    costs, _ = _evaluate_comm_lookahead_sequences(
        agent,
        env,
        goal_obs,
        lookahead_cfg,
        np.asarray([[0.0], [1.0]], dtype=np.float32),
        env.get_state(),
    )
    assert np.allclose(costs, np.zeros_like(costs), atol=1e-6)


def test_dense_heuristic_prefers_qrl_progress():
    env = make_env(
        start=(2.0, 5.0, 0.0),
    )
    agent = DistanceToGoalAgent()
    lookahead_cfg = DubinsLookaheadConfig(
        horizon=1,
        num_sequences=2,
        biased_sequences=0,
        alpha_subgoal=0.0,
        alpha_final=0.0,
        alpha_task_terminal=0.0,
        use_env_stage_cost=False,
        heuristic_mode="dense",
        qrl_progress_alpha=1.0,
    )

    env.reset(seed=0)
    goal_obs = env.abstract_goal_observation()
    costs, _ = _evaluate_comm_lookahead_sequences(
        agent,
        env,
        goal_obs,
        lookahead_cfg,
        np.asarray([[0.0], [1.0]], dtype=np.float32),
        env.get_state(),
    )
    assert not np.allclose(costs, np.zeros_like(costs), atol=1e-6)
    assert not np.isclose(float(costs[0]), float(costs[1]), atol=1e-6)


def test_cli_outputs_separate_lookahead_heuristic_keys(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint_final.pth"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"fake")
    output_dir = tmp_path / "eval"

    old_argv = sys.argv[:]
    old_make_env = comm_eval.make_comm_inspection_env
    old_build_adapter = comm_eval.build_qrl_adapter
    old_auto_device = comm_eval.auto_device
    try:
        comm_eval.make_comm_inspection_env = lambda args: make_env(max_steps=2)
        comm_eval.build_qrl_adapter = lambda args, device, env: (ZeroTurnAgent(), None)
        comm_eval.auto_device = lambda device: torch.device("cpu")
        sys.argv = [
            "comm_inspection_execution_eval.py",
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(output_dir),
            "--device-catalog",
            str(tmp_path / "unused_catalog.json"),
            "--execution-modes",
            "lookahead",
            "--lookahead-heuristics",
            "terminal,dense",
            "--lookahead-horizon",
            "1",
            "--lookahead-num-sequences",
            "1",
            "--lookahead-biased-sequences",
            "0",
            "--planner-alpha-final",
            "0.0",
            "--planner-alpha-task-terminal",
            "0.0",
            "--planner-qrl-progress-alpha",
            "1.0",
            "--starts-per-device",
            "1",
        ]
        comm_eval.main()
    finally:
        sys.argv = old_argv
        comm_eval.make_comm_inspection_env = old_make_env
        comm_eval.build_qrl_adapter = old_build_adapter
        comm_eval.auto_device = old_auto_device

    payload = json.loads((output_dir / "comm_inspection_execution_eval.json").read_text(encoding="utf-8"))
    assert set(payload["results"].keys()) == {"lookahead_terminal", "lookahead_dense"}
    assert payload["execution_modes"] == ["lookahead_terminal", "lookahead_dense"]


if __name__ == "__main__":
    test_execution_eval_visualization_smoke(Path("results/test_comm_inspection_execution_eval"))
    test_hierarchical_execution_eval_smoke(Path("results/test_comm_inspection_execution_eval_hier"))
    test_hierarchical_metrics_and_rollout_recording(Path("results/test_comm_inspection_execution_eval_hier_metrics"))
    test_planner_rejects_collision_rollouts_even_with_env_stage_cost()
    test_terminal_heuristic_ignores_dense_progress_term()
    test_dense_heuristic_prefers_qrl_progress()
    test_cli_outputs_separate_lookahead_heuristic_keys(Path("/tmp/test_comm_inspection_execution_eval_cli"))
    print("Execution eval visualization smoke test passed.")
