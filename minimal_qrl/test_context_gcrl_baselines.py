from __future__ import annotations

import numpy as np
import pytest
import torch

from minimal_qrl.baselines import (
    ContextContrastiveRLAgent,
    ContextGCRLConfig,
    ContextHERDDPGAgent,
    ContextHERReplayBuffer,
    MRNContextHERDDPGAgent,
    MRNGoalCritic,
    RawGoalSetEpisode,
    load_context_checkpoint,
    parameter_count,
    save_context_checkpoint,
)
from minimal_qrl.envs import CommInspectionDubinsUAV2D
from minimal_qrl.test_comm_inspection_baselines import make_env


def make_two_device_env() -> CommInspectionDubinsUAV2D:
    catalog = {
        "ground_station": {"position": [1.0, 4.0], "los_anchor": [1.0, 4.0]},
        "devices": [
            {
                "id": "left",
                "position": [2.0, 4.0],
                "observation_anchor": [2.0, 4.0],
                "observation": {
                    "min_distance": 0.0,
                    "max_distance": 0.5,
                    "preferred_bearing_rad": 0.0,
                    "bearing_tolerance_rad": np.pi,
                    "fov_angle_rad": 2.0 * np.pi,
                    "require_los": False,
                },
            },
            {
                "id": "right",
                "position": [6.0, 4.0],
                "observation_anchor": [6.0, 4.0],
                "observation": {
                    "min_distance": 0.0,
                    "max_distance": 0.5,
                    "preferred_bearing_rad": 0.0,
                    "bearing_tolerance_rad": np.pi,
                    "fov_angle_rad": 2.0 * np.pi,
                    "require_los": False,
                },
            },
        ],
    }
    return CommInspectionDubinsUAV2D(
        device_catalog=catalog,
        bounds=(0.0, 0.0, 8.0, 8.0),
        omega_max=2.0,
        v=1.0,
        dt=0.1,
        max_steps=30,
        obstacles=[],
        start=(4.0, 4.0, 0.0),
        comm_threshold=-100.0,
    )


def test_task_conditioned_interfaces_match_active_step_and_restore_context():
    env = make_env(observation_radius=1.2)
    obs, _ = env.reset(seed=7)
    device_id = env.active_device_id
    action = np.array([0.2], dtype=np.float32)
    next_obs, reward, terminated, _truncated, info = env.step(action)
    outcome = env.transition_outcome_for_task(
        env.state,
        device_id,
        collision=bool(info["collision"]),
        out_of_bounds=bool(info["out_of_bounds"]),
    )
    assert np.allclose(next_obs, outcome["next_observation"])
    assert np.isclose(reward, outcome["reward"])
    assert terminated == outcome["terminated"]
    assert env.active_device_id == device_id
    assert np.allclose(obs, env.observation_for_task(env.observation_to_state(obs), device_id))
    assert env.active_device_id == device_id


def test_context_her_relabels_only_to_future_feasible_catalog_task():
    env = make_two_device_env()
    env.reset(seed=2, options={"device_id": "left"})
    right_terminal = env.sample_task_terminal_state(seed=13) if env.active_device_id == "right" else None
    if right_terminal is None:
        with env._task_scope("right"):
            right_terminal = env.sample_task_terminal_state(seed=13)
    episode = RawGoalSetEpisode(
        states=np.stack([
            np.array([4.0, 4.0, 0.0], dtype=np.float32),
            np.asarray(right_terminal, dtype=np.float32),
        ]),
        actions=np.array([[0.0]], dtype=np.float32),
        device_id="left",
        collisions=np.array([False]),
        out_of_bounds=np.array([False]),
        truncated=np.array([True]),
    )
    replay = ContextHERReplayBuffer(env, 20, torch.device("cpu"), her_k=4, seed=5)
    replay.add_episode(episode)
    batch = replay.sample_numpy(4, positive_only=True)
    assert set(batch["device_ids"]) == {"right"}
    assert np.all(batch["relabeled"] == 1.0)
    # The sampled goal is achieved in the future, but not necessarily at t+1 in general.
    assert np.all(batch["done"] == 1.0)
    assert env.active_device_id == "left"


def test_context_her_keeps_original_sample_and_bootstraps_time_limit():
    env = make_two_device_env()
    env.reset(seed=4, options={"device_id": "left"})
    episode = RawGoalSetEpisode(
        states=np.asarray([[4.0, 4.0, 0.0], [4.1, 4.0, 0.0]], dtype=np.float32),
        actions=np.zeros((1, 1), dtype=np.float32),
        device_id="left",
        collisions=np.array([False]),
        out_of_bounds=np.array([False]),
        truncated=np.array([True]),
    )
    replay = ContextHERReplayBuffer(env, 20, torch.device("cpu"), her_k=4, seed=6)
    replay.add_episode(episode)
    batch = replay.sample_numpy(1)
    assert batch["relabeled"].item() == 0.0
    assert batch["done"].item() == 0.0


def _positive_replay(env: CommInspectionDubinsUAV2D) -> ContextHERReplayBuffer:
    env.reset(seed=3)
    terminal = env.sample_task_terminal_state(seed=11)
    episode = RawGoalSetEpisode(
        states=np.stack([env.state.copy(), terminal]),
        actions=np.zeros((1, 1), dtype=np.float32),
        device_id=env.active_device_id,
        collisions=np.array([False]),
        out_of_bounds=np.array([False]),
        truncated=np.array([False]),
        source="teacher",
    )
    replay = ContextHERReplayBuffer(env, 20, torch.device("cpu"), her_k=4, seed=9)
    replay.add_episode(episode)
    return replay


@pytest.mark.parametrize(
    "agent_type,positive_only",
    [
        (ContextHERDDPGAgent, False),
        (MRNContextHERDDPGAgent, False),
        (ContextContrastiveRLAgent, True),
    ],
)
def test_context_agents_take_bounded_actions_and_update(agent_type, positive_only):
    env = make_env(observation_radius=1.2)
    replay = _positive_replay(env)
    cfg = ContextGCRLConfig(
        hidden_dim=32,
        representation_dim=8,
        residual_dim=8,
        batch_size=4,
        replay_size=20,
    )
    agent = agent_type(env, cfg, torch.device("cpu"))
    batch = replay.sample(4, positive_only=positive_only)
    stats = agent.update(batch)
    assert all(np.isfinite(value) for value in stats.values())
    obs = env.state_to_observation(env.state)
    action = agent.act(obs, env.abstract_goal_observation(), eval_mode=True)
    assert env.action_space.contains(action)
    assert np.isfinite(agent.value(obs, env.abstract_goal_observation()))


def test_contrastive_duplicate_devices_are_not_false_negatives():
    env = make_env()
    cfg = ContextGCRLConfig(hidden_dim=16, representation_dim=4)
    agent = ContextContrastiveRLAgent(env, cfg, torch.device("cpu"))
    obs = torch.zeros((3, agent.obs_dim))
    action = torch.zeros((3, agent.act_dim))
    goal = torch.zeros((3, agent.goal_dim))
    loss, stats = agent.contrastive_loss(obs, action, goal, ["same"] * 3)
    assert torch.isfinite(loss)
    assert stats["contrastive_negative_accuracy"] == 1.0
    assert stats["contrastive_negative_score"] == 0.0


def test_mrn_uses_l2_metric_and_default_parameter_budget():
    critic = MRNGoalCritic(4, 4, 1, 16, 4, 4)
    embedding = torch.tensor([[0.0, 0.0, 0.0, 0.0], [3.0, 4.0, 0.0, 0.0]])
    with torch.no_grad():
        critic.metric_projection.weight.copy_(torch.eye(4))
        critic.metric_projection.bias.zero_()
        critic.residual_projection.weight.zero_()
        critic.residual_projection.bias.zero_()
    distance, metric, residual = critic.distance_from_embeddings(embedding, embedding)
    assert torch.allclose(distance, torch.zeros_like(distance))
    assert torch.all(metric >= 0.0) and torch.all(residual >= 0.0)
    pair_distance, pair_metric, _ = critic.distance_from_embeddings(embedding[:1], embedding[1:])
    assert torch.allclose(pair_distance, torch.tensor([5.0]))
    assert torch.allclose(pair_metric, torch.tensor([5.0]))

    env = make_env()
    cfg = ContextGCRLConfig()
    monolithic = ContextHERDDPGAgent(env, cfg, torch.device("cpu"))
    mrn = MRNContextHERDDPGAgent(env, cfg, torch.device("cpu"))
    relative_difference = abs(parameter_count(monolithic.critic) - parameter_count(mrn.critic)) / parameter_count(monolithic.critic)
    assert relative_difference <= 0.05


def test_context_checkpoint_roundtrip_and_catalog_guard(tmp_path):
    env = make_env()
    cfg = ContextGCRLConfig(hidden_dim=16, representation_dim=4, residual_dim=4)
    agent = ContextHERDDPGAgent(env, cfg, torch.device("cpu"))
    obs, _ = env.reset(seed=8)
    goal = env.abstract_goal_observation()
    expected_action = agent.act(obs, goal, eval_mode=True)
    path = tmp_path / "agent.pth"
    save_context_checkpoint(
        path,
        "context_her_ddpg",
        agent,
        env,
        seed=3,
        env_steps=12,
        teacher_steps=7,
        updates=5,
        replay_diagnostics={"relabel_count": 2.0},
    )
    loaded, metadata = load_context_checkpoint(path, env, torch.device("cpu"))
    assert np.allclose(expected_action, loaded.act(obs, goal, eval_mode=True))
    assert metadata["env_steps"] == 12

    other_env = make_two_device_env()
    with pytest.raises(ValueError, match="catalog_hash"):
        load_context_checkpoint(path, other_env, torch.device("cpu"))
