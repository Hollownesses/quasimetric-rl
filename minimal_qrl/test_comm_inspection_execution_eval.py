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

from minimal_qrl.envs import CommInspectionDubinsUAV2D
from minimal_qrl.eval.comm_inspection_execution_eval import (
    VisualizationConfig,
    evaluate_execution_mode,
)
from minimal_qrl.gc_agents import GoalConditionedAgentBase


class ZeroTurnAgent(GoalConditionedAgentBase):
    def act(self, obs: np.ndarray, goal_obs: np.ndarray, eval_mode: bool = True) -> np.ndarray:
        _ = obs, goal_obs, eval_mode
        return np.array([0.0], dtype=np.float32)

    def value(self, obs: np.ndarray, goal_obs: np.ndarray) -> float:
        _ = obs, goal_obs
        return 0.0


def make_env() -> CommInspectionDubinsUAV2D:
    return CommInspectionDubinsUAV2D(
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


def test_execution_eval_visualization_smoke(tmp_path: Path):
    env = make_env()
    agent = ZeroTurnAgent()
    viz_cfg = VisualizationConfig(
        save_visualizations=True,
        num_samples=1,
        save_failures=True,
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
    assert len(visualizations["samples"]) == 1
    assert len(visualizations["failures"]) <= 1

    sample_png = tmp_path / visualizations["samples"][0]["png"]
    assert sample_png.exists()


if __name__ == "__main__":
    test_execution_eval_visualization_smoke(Path("results/test_comm_inspection_execution_eval"))
    print("Execution eval visualization smoke test passed.")
