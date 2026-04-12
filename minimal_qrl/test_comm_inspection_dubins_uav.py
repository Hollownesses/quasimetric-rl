#!/usr/bin/env python3
"""
轻量测试：task-conditioned 通信感知巡检 Dubins UAV 环境。
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from minimal_qrl.envs import CircleObstacle, CommInspectionDubinsUAV2D


def make_env(**kwargs) -> CommInspectionDubinsUAV2D:
    default = dict(
        bounds=(0.0, 0.0, 10.0, 10.0),
        omega_max=1.0,
        v=1.0,
        dt=0.1,
        max_steps=100,
        observation_mode="task_context",
        inspection_target=(5.0, 5.0),
        ground_station=(1.0, 5.0),
        observation_radius=2.0,
        fov_angle=np.pi / 2.0,
        require_target_los=True,
        comm_alpha=2.0,
        comm_bias=5.0,
        comm_occlusion_penalty=8.0,
        comm_threshold=1.0,
        require_ground_station_los=False,
        goal_sampling_mode="task_feasible",
        goal_position_tolerance=0.2,
        goal_heading_tolerance=0.25,
    )
    default.update(kwargs)
    return CommInspectionDubinsUAV2D(**default)


def test_task_context_observation_matches_space():
    env = make_env()
    obs, _ = env.reset(seed=42)
    assert obs.shape == env.observation_space.shape
    assert obs.shape == (20,)


def test_reset_sets_task_entities_and_goal():
    env = make_env()
    obs, info = env.reset(seed=42)
    assert obs.shape == env.observation_space.shape
    assert env.inspection_target is not None
    assert env.ground_station is not None
    assert env.goal is not None
    assert env.is_task_feasible(np.asarray(env.goal, dtype=np.float32))
    assert tuple(info["inspection_target"]) == tuple(env.inspection_target)
    assert tuple(info["ground_station"]) == tuple(env.ground_station)
    assert tuple(info["goal"]) == tuple(env.goal)
    assert info["observation_mode"] == "task_context"


def test_sample_task_feasible_goal_is_valid():
    env = make_env()
    env.reset(seed=0)
    goal = env.sample_task_feasible_goal(seed=1)
    assert env.is_valid_state(goal)
    assert env.is_observation_feasible(goal)
    assert env.is_communication_feasible(goal)
    assert env.is_task_feasible(goal)


def test_reset_resamples_start_and_goal_when_not_fixed():
    env = make_env()
    env.reset(seed=0)
    start1 = tuple(float(v) for v in env.start)
    goal1 = tuple(float(v) for v in env.goal)

    env.reset(seed=1)
    start2 = tuple(float(v) for v in env.start)
    goal2 = tuple(float(v) for v in env.goal)

    assert start1 != start2
    assert goal1 != goal2


def test_observation_score_direction():
    env = make_env()
    env.reset(seed=0)
    good = np.array([4.0, 5.0, 0.0], dtype=np.float32)
    bad = np.array([2.5, 5.0, np.pi], dtype=np.float32)
    assert env.compute_observation_score(good) > env.compute_observation_score(bad)


def test_communication_score_direction():
    env = make_env(
        ground_station=(1.0, 1.0),
        inspection_target=(5.0, 5.0),
        comm_alpha=2.0,
        comm_bias=3.0,
        comm_threshold=1.5,
        require_ground_station_los=False,
        goal_sampling_mode="valid",
    )
    env.reset(seed=0)
    near_station = np.array([1.4, 1.0, 0.0], dtype=np.float32)
    far_station = np.array([9.0, 9.0, 0.0], dtype=np.float32)
    assert env.compute_communication_score(near_station) > env.compute_communication_score(far_station)


def test_collision_penalty_is_negative():
    obstacle = CircleObstacle(x=4.16, y=5.0, radius=0.12)
    env = make_env(
        obstacles=[obstacle],
        start=(4.0, 5.0, 0.0),
        goal=(6.0, 5.0, 0.0),
    )
    env.reset(seed=0)
    _, reward, _, _, info = env.step(np.array([0.0], dtype=np.float32))
    assert info["collision"]
    assert info["cost_collision"] >= 10.0
    assert reward < -1.0


def test_out_of_bounds_penalty_is_negative():
    env = make_env(
        bounds=(0.0, 0.0, 1.0, 1.0),
        start=(0.95, 0.95, 0.0),
        goal=(0.95, 0.95, 0.0),
        inspection_target=(0.8, 0.8),
        ground_station=(0.2, 0.2),
    )
    env.reset(seed=0)
    _, reward, _, _, info = env.step(np.array([0.0], dtype=np.float32))
    assert info["out_of_bounds"]
    assert info["cost_oob"] >= 10.0
    assert reward < -1.0


def test_success_trigger_near_exact_goal():
    env = make_env(
        goal_position_tolerance=0.15,
        goal_heading_tolerance=0.1,
    )
    goal = (4.0, 5.0, 0.0)
    start = (3.9, 5.0, 0.0)
    env.reset(seed=0, options={"start": start, "goal": goal})
    _, _, terminated, _, info = env.step(np.array([0.0], dtype=np.float32))
    assert terminated
    assert info["success"]
    assert info["distance_to_goal"] <= env.goal_position_tolerance
    assert info["heading_error"] <= env.goal_heading_tolerance


def test_goal_reached_but_not_task_feasible_is_not_success():
    env = make_env(
        goal_position_tolerance=0.15,
        goal_heading_tolerance=0.1,
        observation_radius=1.0,
    )
    goal = (4.0, 5.0, 0.0)
    start = (3.9, 5.0, 0.0)
    env.reset(seed=0, options={"start": start, "goal": goal})
    env.inspection_target = (9.0, 9.0)
    _, _, terminated, _, info = env.step(np.array([0.0], dtype=np.float32))
    assert not env.is_task_feasible(env.state)
    assert not terminated
    assert not info["success"]
    assert info["distance_to_goal"] <= env.goal_position_tolerance
    assert info["heading_error"] <= env.goal_heading_tolerance


def test_task_feasible_not_equal_success():
    env = make_env()
    goal = (8.0, 8.0, 0.0)
    feasible_state = (4.0, 5.0, 0.0)
    env.reset(seed=0, options={"start": feasible_state, "goal": goal})
    _, _, terminated, _, info = env.step(np.array([0.0], dtype=np.float32))
    assert info["task_feasible"]
    assert not terminated
    assert info["ever_task_feasible"]
    assert info["first_task_feasible_step"] == 1


def test_zero_communication_break_cost_disables_fixed_penalty():
    env = make_env(
        goal_sampling_mode="valid",
        comm_threshold=4.0,
        communication_break_cost=0.0,
    )
    env.reset(seed=0)
    step_terms = env.compute_step_terms(
        new_state=env.state,
        collision=False,
        out_of_bounds=False,
    )
    assert step_terms["cost_comm_break"] == 0.0

    env.communication_break_cost = 1.25
    step_terms = env.compute_step_terms(
        new_state=env.state,
        collision=False,
        out_of_bounds=False,
    )
    assert step_terms["cost_comm_break"] == 1.25


def test_feasible_state_has_zero_violation_costs():
    env = make_env()
    feasible_state = np.array([4.0, 5.0, 0.0], dtype=np.float32)
    env.reset(seed=0, options={"start": feasible_state, "goal": feasible_state})
    step_terms = env.compute_step_terms(
        new_state=env.state,
        collision=False,
        out_of_bounds=False,
    )
    assert step_terms["cost_obs_violation"] == 0.0
    assert step_terms["cost_comm_violation"] == 0.0
    assert step_terms["cost_obs_fail"] == 0.0
    assert step_terms["cost_comm_break"] == 0.0


def test_taskscore_clips_and_normalizes_margins():
    env = make_env(goal_sampling_mode="valid")
    env.reset(seed=0)
    assert env.normalize_task_margin(3.0) == 1.0
    assert env.normalize_task_margin(-3.0) == -1.0
    assert np.isclose(env.normalize_task_margin(1.0), 0.5)


def test_repair_state_keeps_geometric_validity_without_forcing_task_feasible():
    env = make_env(
        bounds=(0.0, 0.0, 10.0, 10.0),
        observation_radius=0.5,
        goal_sampling_mode="valid",
        obstacles=[CircleObstacle(x=5.0, y=5.0, radius=0.7)],
    )
    env.reset(seed=0)
    raw = np.array([5.0, 5.0, 4.0], dtype=np.float32)
    repaired = env.repair_state(raw)
    assert env.is_valid_state(repaired)
    assert -np.pi <= float(repaired[2]) <= np.pi
    assert not env.is_task_feasible(repaired)


def test_legacy_modes_reset_and_step():
    for mode in ("cos_sin", "state"):
        env = make_env(observation_mode=mode)
        obs, _ = env.reset(seed=0)
        assert obs.shape == env.observation_space.shape
        next_obs, reward, terminated, truncated, _ = env.step(np.array([0.1], dtype=np.float32))
        assert next_obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)


def test_visualize_script_smoke():
    repo_root = Path(__file__).parent.parent
    out_path = repo_root / "results" / "minimal_qrl_inspection_dubins" / "comm_inspection_dubins_uav_vis" / "smoke_test.png"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "minimal_qrl.visualize_comm_inspection_dubins_uav",
            "--out",
            str(out_path),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Saved visualization to" in result.stdout
    assert out_path.exists()


if __name__ == "__main__":
    test_task_context_observation_matches_space()
    test_reset_sets_task_entities_and_goal()
    test_sample_task_feasible_goal_is_valid()
    test_reset_resamples_start_and_goal_when_not_fixed()
    test_observation_score_direction()
    test_communication_score_direction()
    test_collision_penalty_is_negative()
    test_out_of_bounds_penalty_is_negative()
    test_success_trigger_near_exact_goal()
    test_goal_reached_but_not_task_feasible_is_not_success()
    test_task_feasible_not_equal_success()
    test_zero_communication_break_cost_disables_fixed_penalty()
    test_feasible_state_has_zero_violation_costs()
    test_taskscore_clips_and_normalizes_margins()
    test_repair_state_keeps_geometric_validity_without_forcing_task_feasible()
    test_legacy_modes_reset_and_step()
    test_visualize_script_smoke()
    print("All comm inspection Dubins UAV tests passed.")
