"""Tests for the device-catalog industrial inspection environment."""

from __future__ import annotations

import math

import numpy as np
import pytest

from minimal_qrl.dataset import collect_goal_set_comm_episode_pair, create_dataset
from minimal_qrl.envs import (
    CircleObstacle,
    CommInspectionDubinsUAV2D,
    DeviceTaskSpec,
    IndustrialInspectionCatalog,
    TaskContextInfeasibleError,
)


def catalog_dict(*, require_los: bool = False):
    return {
        "ground_station": {"position": [0.5, 0.5], "los_anchor": [0.5, 0.5]},
        "devices": [
            {
                "id": "tank_01",
                "position": [5.0, 5.0],
                "observation_anchor": [5.0, 5.0],
                "observation": {
                    "min_distance": 1.0,
                    "max_distance": 2.0,
                    "preferred_bearing_rad": math.pi,
                    "bearing_tolerance_rad": 0.5,
                    "fov_angle_rad": math.pi / 2.0,
                    "require_los": require_los,
                },
            },
            {
                "id": "pipe_02",
                "position": [7.0, 7.0],
                "observation_anchor": [7.0, 7.0],
                "observation": {
                    "min_distance": 0.8,
                    "max_distance": 1.5,
                    "preferred_bearing_rad": -math.pi / 2.0,
                    "bearing_tolerance_rad": 0.4,
                    "fov_angle_rad": math.pi / 2.0,
                    "require_los": require_los,
                },
            },
        ],
    }


def make_env(**kwargs) -> CommInspectionDubinsUAV2D:
    defaults = dict(
        device_catalog=catalog_dict(),
        bounds=(0.0, 0.0, 10.0, 10.0),
        omega_max=3.0,
        v=1.0,
        dt=0.1,
        max_steps=100,
        comm_bias=20.0,
        comm_threshold=0.0,
    )
    defaults.update(kwargs)
    return CommInspectionDubinsUAV2D(**defaults)


def test_catalog_validation_rejects_duplicates_and_bad_angles():
    duplicate = catalog_dict()
    duplicate["devices"][1]["id"] = "tank_01"
    with pytest.raises(ValueError, match="duplicate device"):
        IndustrialInspectionCatalog.from_dict(duplicate)

    bad = catalog_dict()
    bad["devices"][0]["observation"]["bearing_tolerance_rad"] = 0.0
    with pytest.raises(ValueError, match="bearing_tolerance"):
        IndustrialInspectionCatalog.from_dict(bad)


def test_named_task_context_and_abstract_goal():
    env = make_env()
    obs, info = env.reset(seed=3, options={"device_id": "tank_01"})
    assert obs.shape == (len(env.TASK_CONTEXT_FIELDS),)
    assert obs.shape == env.observation_space.shape
    assert obs[env.task_context_indices["abstract_goal"]] == 0.0
    goal = env.abstract_goal_observation()
    assert goal[env.task_context_indices["abstract_goal"]] == 1.0
    assert info["device_id"] == "tank_01"


def test_fixed_ground_station_and_uniform_catalog_sampling():
    env = make_env()
    station = env.ground_station
    sampled = set()
    for seed in range(20):
        _, info = env.reset(seed=seed)
        sampled.add(info["device_id"])
        assert env.ground_station == station
    assert sampled == set(env.device_ids)


def test_explicit_device_and_task_context_are_never_overwritten():
    env = make_env()
    for seed in range(5):
        env.reset(seed=seed, options={"device_id": "pipe_02"})
        assert env.active_device_id == "pipe_02"

    custom = DeviceTaskSpec.from_dict(
        {
            "id": "alarm_exact",
            "position": [4.0, 4.0],
            "observation_anchor": [4.0, 4.0],
            "observation": {
                "min_distance": 0.5,
                "max_distance": 1.0,
                "preferred_bearing_rad": 0.0,
                "bearing_tolerance_rad": 0.4,
                "fov_angle_rad": 1.2,
                "require_los": False,
            },
        }
    )
    env.reset(seed=9, options={"task_context": custom})
    assert env.active_device_id == "alarm_exact"
    assert env.inspection_target == (4.0, 4.0)


def test_annular_sector_heading_and_terminal_communication_conditions():
    env = make_env()
    env.reset(seed=0, options={"device_id": "tank_01"})
    good = np.array([3.5, 5.0, 0.0], dtype=np.float32)
    too_close = np.array([4.5, 5.0, 0.0], dtype=np.float32)
    wrong_sector = np.array([6.5, 5.0, math.pi], dtype=np.float32)
    wrong_heading = np.array([3.5, 5.0, math.pi], dtype=np.float32)
    assert env.is_observation_feasible(good)
    assert env.is_task_feasible(good)
    assert not env.is_observation_feasible(too_close)
    assert not env.is_observation_feasible(wrong_sector)
    assert not env.is_observation_feasible(wrong_heading)

    env.comm_threshold = 100.0
    assert env.is_observation_feasible(good)
    assert not env.is_task_feasible(good)


def test_semantic_device_can_be_inside_obstacle_and_surface_los_is_open_ended():
    obstacle = CircleObstacle(5.0, 5.0, 1.0)
    catalog = catalog_dict(require_los=True)
    device = catalog["devices"][0]
    device["observation_anchor"] = [4.0, 5.0]
    device["observation"]["min_distance"] = 1.1
    env = make_env(device_catalog=catalog, obstacles=[obstacle])
    env.reset(seed=0, options={"device_id": "tank_01"})

    near_side = np.array([3.5, 5.0, 0.0], dtype=np.float32)
    far_side = np.array([6.5, 5.0, math.pi], dtype=np.float32)
    assert obstacle.contains(*env.inspection_target)
    assert env.is_valid_state(near_side)
    assert not env.is_valid_state(np.array([5.0, 5.0, 0.0], dtype=np.float32))
    assert env._segment_has_los(tuple(near_side[:2]), env.observation_anchor, allow_endpoint_contact=True)
    assert not env._segment_has_los(tuple(far_side[:2]), env.observation_anchor, allow_endpoint_contact=True)


def test_infeasible_device_fails_with_device_id():
    catalog = catalog_dict()
    catalog["devices"] = [catalog["devices"][0]]
    catalog["devices"][0]["observation"]["min_distance"] = 0.1
    catalog["devices"][0]["observation"]["max_distance"] = 0.2
    with pytest.raises(TaskContextInfeasibleError, match="tank_01"):
        make_env(
            device_catalog=catalog,
            obstacles=[CircleObstacle(5.0, 5.0, 1.0)],
            sample_max_attempts=20,
        )


def test_terminal_sampling_and_step_cost_semantics():
    env = make_env(communication_break_cost=1.25)
    env.reset(seed=0, options={"device_id": "tank_01"})
    terminal = env.sample_task_terminal_state(seed=1)
    assert env.is_task_feasible(terminal)

    env.comm_threshold = 100.0
    terms = env.compute_step_terms(terminal, collision=False, out_of_bounds=False)
    assert terms["cost_comm_break"] == 1.25
    assert terms["cost_time"] == env.dt


def test_dataset_contains_independent_global_push_pairs():
    env = make_env(max_steps=5)
    episode, abstract_episode = collect_goal_set_comm_episode_pair(env, max_steps=5, seed=11, context_id=4)
    infos = episode.transition_infos
    assert bool(infos["global_push_pair_mask"].all())
    assert infos["global_push_source_observations"].shape == episode.all_observations[:-1].shape
    assert infos["global_push_goal_observations"].shape == episode.all_observations[:-1].shape
    assert not np.allclose(
        infos["global_push_source_observations"],
        infos["global_push_goal_observations"],
    )
    for key in ("global_push_source_observations", "global_push_goal_observations"):
        for obs in infos[key].numpy():
            assert env.is_valid_state(env.observation_to_state(obs))
            assert obs[env.task_context_indices["device_x"]] == env.inspection_target[0]
    assert abstract_episode is not None
    assert not bool(abstract_episode.transition_infos["global_push_pair_mask"].any())


def test_dataset_task_goals_are_catalog_devices_and_teacher_reuses_device():
    env = make_env(max_steps=80)
    episodes = list(
        create_dataset(
            env,
            num_episodes=3,
            max_steps_per_episode=80,
            seed=2,
            task_aware_teacher_ratio=1.0,
        )
    )
    device_indices = set()
    for episode in episodes:
        infos = episode.transition_infos
        device_indices.update(int(v) for v in infos["device_index"].tolist() if int(v) >= 0)
    assert device_indices <= set(range(len(env.device_ids)))
    assert device_indices
