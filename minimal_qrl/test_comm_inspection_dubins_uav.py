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
from minimal_qrl.dataset import (
    collect_goal_set_comm_episode_pair,
    collect_task_aware_comm_teacher_episode_pair,
    create_dataset,
)


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
        randomize_inspection_target=False,
        randomize_ground_station=False,
    )
    default.update(kwargs)
    return CommInspectionDubinsUAV2D(**default)


def test_task_context_observation_matches_space():
    env = make_env()
    obs, _ = env.reset(seed=42)
    assert obs.shape == env.observation_space.shape
    assert obs.shape == (28,)
    assert obs[-1] == 0.0
    abstract = env.abstract_goal_observation()
    assert abstract.shape == obs.shape
    assert abstract[-1] == 1.0


def test_reset_sets_task_context_and_abstract_goal():
    env = make_env()
    obs, info = env.reset(seed=42)
    assert obs.shape == env.observation_space.shape
    assert env.inspection_target is not None
    assert env.ground_station is not None
    assert env.goal is None
    assert tuple(info["inspection_target"]) == tuple(env.inspection_target)
    assert tuple(info["ground_station"]) == tuple(env.ground_station)
    assert "abstract_goal_observation" in info
    assert info["observation_mode"] == "task_context"
    assert not env.is_terminal_goal_state(env.state)


def test_sample_task_terminal_state_is_valid():
    env = make_env()
    env._ensure_valid_task_entities(seed=0)
    terminal = env.sample_task_terminal_state(seed=1)
    assert env.is_valid_state(terminal)
    assert env.is_observation_feasible(terminal)
    assert env.is_communication_feasible(terminal)
    assert env.is_terminal_goal_state(terminal)


def test_random_context_changes_abstract_goal():
    env = make_env(randomize_inspection_target=True, randomize_ground_station=True)
    env.reset(seed=0)
    start1 = tuple(float(v) for v in env.start)
    abstract1 = env.abstract_goal_observation().copy()

    env.reset(seed=1)
    start2 = tuple(float(v) for v in env.start)
    abstract2 = env.abstract_goal_observation().copy()

    assert start1 != start2
    assert not np.allclose(abstract1, abstract2)


def test_random_context_start_is_not_inspection_target():
    env = make_env(randomize_inspection_target=True, randomize_ground_station=True)
    env.reset(seed=0)
    start_xy = np.asarray(env.start[:2], dtype=np.float32)
    target_xy = np.asarray(env.inspection_target, dtype=np.float32)
    assert float(np.linalg.norm(start_xy - target_xy)) >= env.min_start_target_distance


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
    )
    env._ensure_valid_task_entities(seed=0)
    near_station = np.array([1.4, 1.0, 0.0], dtype=np.float32)
    far_station = np.array([9.0, 9.0, 0.0], dtype=np.float32)
    assert env.compute_communication_score(near_station) > env.compute_communication_score(far_station)


def test_collision_penalty_is_negative():
    obstacle = CircleObstacle(x=4.16, y=5.0, radius=0.12)
    env = make_env(
        obstacles=[obstacle],
        start=(4.0, 5.0, 0.0),
    )
    env.reset(seed=0)
    _, reward, terminated, truncated, info = env.step(np.array([0.0], dtype=np.float32))
    assert terminated
    assert not truncated
    assert not info["success"]
    assert info["collision"]
    assert info["cost_collision"] >= 10.0
    assert reward < -1.0


def test_out_of_bounds_penalty_is_negative():
    env = make_env(
        bounds=(0.0, 0.0, 1.0, 1.0),
        start=(0.95, 0.95, 0.0),
        inspection_target=(0.8, 0.8),
        ground_station=(0.2, 0.2),
    )
    env.reset(seed=0)
    _, reward, terminated, truncated, info = env.step(np.array([0.0], dtype=np.float32))
    assert terminated
    assert not truncated
    assert not info["success"]
    assert info["out_of_bounds"]
    assert info["cost_oob"] >= 10.0
    assert reward < -1.0


def test_success_trigger_on_task_terminal_set():
    env = make_env()
    start = (3.9, 5.0, 0.0)
    env.reset(seed=0, options={"start": start})
    _, _, terminated, _, info = env.step(np.array([0.0], dtype=np.float32))
    assert terminated
    assert info["success"]
    assert env.is_terminal_goal_state(env.state)


def test_nonterminal_state_is_not_success():
    env = make_env(
        observation_radius=1.0,
    )
    start = (2.0, 5.0, 0.0)
    env.reset(seed=0, options={"start": start})
    _, _, terminated, _, info = env.step(np.array([0.0], dtype=np.float32))
    assert not env.is_task_feasible(env.state)
    assert not terminated
    assert not info["success"]


def test_task_feasible_equals_success():
    env = make_env()
    feasible_state = (4.0, 5.0, 0.0)
    env.reset(seed=0, options={"start": feasible_state})
    _, _, terminated, _, info = env.step(np.array([0.0], dtype=np.float32))
    assert info["task_feasible"]
    assert terminated
    assert info["success"]
    assert info["ever_task_feasible"]
    assert info["first_task_feasible_step"] in (0, 1)


def test_zero_communication_break_cost_disables_fixed_penalty():
    env = make_env(
        comm_threshold=1.0,
        communication_break_cost=0.0,
    )
    env.reset(seed=0)
    env.state = np.array([9.0, 9.0, 0.0], dtype=np.float32)
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
    env.reset(seed=0, options={"start": feasible_state})
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
    env = make_env()
    env.reset(seed=0)
    assert env.normalize_task_margin(3.0) == 1.0
    assert env.normalize_task_margin(-3.0) == -1.0
    assert np.isclose(env.normalize_task_margin(1.0), 0.5)


def test_repair_state_keeps_geometric_validity_without_forcing_task_feasible():
    env = make_env(
        bounds=(0.0, 0.0, 10.0, 10.0),
        observation_radius=0.5,
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


def test_goal_set_dataset_adds_abstract_edge_for_success():
    env = make_env(
        start=(3.9, 5.0, 0.0),
        max_steps=5,
    )
    episode, abstract_episode = collect_goal_set_comm_episode_pair(env, max_steps=5, seed=0)
    assert episode.transition_infos["abstract_goal_edge"].shape[0] == episode.num_transitions
    assert not bool(episode.transition_infos["abstract_goal_edge"].any())
    assert abstract_episode is not None
    assert bool(abstract_episode.transition_infos["abstract_goal_edge"][0])
    assert bool(abstract_episode.transition_infos["source_terminal_goal_state"][0])
    assert float(abstract_episode.rewards[0]) == 0.0
    assert float(abstract_episode.all_observations[-1, -1]) == 1.0


def test_goal_set_dataset_adds_abstract_edge_without_rollout_success():
    env = make_env(
        start=(2.0, 5.0, 0.0),
        max_steps=1,
    )
    episode, abstract_episode = collect_goal_set_comm_episode_pair(env, max_steps=1, seed=0)
    assert not bool(episode.terminals[0])
    assert abstract_episode is not None
    assert bool(abstract_episode.transition_infos["abstract_goal_edge"][0])
    assert bool(abstract_episode.transition_infos["source_terminal_goal_state"][0])
    terminal_state = env.observation_to_state(abstract_episode.all_observations[0])
    assert env.is_terminal_goal_state(terminal_state)
    assert float(abstract_episode.rewards[0]) == 0.0
    assert float(abstract_episode.all_observations[-1, -1]) == 1.0


def test_task_aware_teacher_collects_success_chain():
    env = make_env(
        start=(2.0, 5.0, 0.0),
        omega_max=3.0,
        max_steps=80,
        obstacles=[],
    )
    episode, abstract_episode = collect_task_aware_comm_teacher_episode_pair(
        env,
        max_steps=80,
        seed=3,
        context_id=7,
    )
    assert episode is not None
    assert abstract_episode is not None
    assert bool(episode.terminals[-1])
    assert bool(episode.transition_infos["teacher_guided"].all())
    assert not bool(episode.transition_infos["abstract_goal_edge"].any())
    final_state = env.observation_to_state(episode.all_observations[-1])
    assert env.is_terminal_goal_state(final_state)
    assert bool(abstract_episode.transition_infos["abstract_goal_edge"][0])
    assert bool(abstract_episode.transition_infos["source_terminal_goal_state"][0])
    assert float(abstract_episode.rewards[0]) == 0.0
    assert int(abstract_episode.transition_infos["context_id"][0]) == 7


def test_dataset_teacher_shares_random_context_id():
    num_episodes = 3
    env = make_env(
        randomize_inspection_target=True,
        randomize_ground_station=True,
        omega_max=3.0,
        max_steps=80,
        obstacles=[],
    )
    episodes = list(
        create_dataset(
            env,
            num_episodes=num_episodes,
            max_steps_per_episode=80,
            seed=11,
            task_aware_teacher_ratio=1.0,
        )
    )

    teacher_contexts = set()
    random_contexts = set()
    for episode in episodes:
        infos = episode.transition_infos
        if "context_id" not in infos:
            continue
        context_ids = set(int(v) for v in infos["context_id"].tolist())
        assert context_ids <= set(range(num_episodes))
        if bool(infos.get("teacher_guided", np.array([False])).any()):
            teacher_contexts.update(context_ids)
        elif not bool(infos.get("abstract_goal_edge", np.array([False])).any()):
            random_contexts.update(context_ids)

    assert teacher_contexts
    assert teacher_contexts <= random_contexts


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
    test_reset_sets_task_context_and_abstract_goal()
    test_sample_task_terminal_state_is_valid()
    test_random_context_changes_abstract_goal()
    test_random_context_start_is_not_inspection_target()
    test_observation_score_direction()
    test_communication_score_direction()
    test_collision_penalty_is_negative()
    test_out_of_bounds_penalty_is_negative()
    test_success_trigger_on_task_terminal_set()
    test_nonterminal_state_is_not_success()
    test_task_feasible_equals_success()
    test_zero_communication_break_cost_disables_fixed_penalty()
    test_feasible_state_has_zero_violation_costs()
    test_taskscore_clips_and_normalizes_margins()
    test_repair_state_keeps_geometric_validity_without_forcing_task_feasible()
    test_legacy_modes_reset_and_step()
    test_goal_set_dataset_adds_abstract_edge_for_success()
    test_goal_set_dataset_adds_abstract_edge_without_rollout_success()
    test_task_aware_teacher_collects_success_chain()
    test_dataset_teacher_shares_random_context_id()
    test_visualize_script_smoke()
    print("All comm inspection Dubins UAV tests passed.")
