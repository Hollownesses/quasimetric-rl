from __future__ import annotations

import json

import numpy as np
import torch

from minimal_qrl.baselines import (
    GoalSetReplayBuffer,
    GoalSetSACAgent,
    GoalSetSACConfig,
    HybridAStarConfig,
    HybridAStarController,
    HybridAStarValueOracle,
    MPPIConfig,
    MPPIController,
    simulate_action_sequences,
)
from minimal_qrl.envs import CircleObstacle, CommInspectionDubinsUAV2D
from minimal_qrl.eval.comm_inspection_baseline_eval import (
    IncrementalResultWriter,
    _evaluate_controller,
)
from minimal_qrl.gc_agents import GoalConditionedAgentBase


def make_env(**kwargs) -> CommInspectionDubinsUAV2D:
    observation_radius = float(kwargs.pop("observation_radius", 0.6))
    catalog = {
        "ground_station": {"position": [1.0, 4.0], "los_anchor": [1.0, 4.0]},
        "devices": [
            {
                "id": "device_01",
                "position": [5.0, 4.0],
                "observation_anchor": [5.0, 4.0],
                "observation": {
                    "min_distance": 0.0,
                    "max_distance": observation_radius,
                    "preferred_bearing_rad": 0.0,
                    "bearing_tolerance_rad": np.pi,
                    "fov_angle_rad": np.pi / 2.0,
                    "require_los": True,
                },
            }
        ],
    }
    defaults = dict(
        device_catalog=catalog,
        bounds=(0.0, 0.0, 8.0, 8.0),
        omega_max=2.0,
        v=1.0,
        dt=0.1,
        max_steps=80,
        obstacles=[],
        start=(2.0, 4.0, 0.0),
        comm_threshold=-100.0,
    )
    defaults.update(kwargs)
    return CommInspectionDubinsUAV2D(**defaults)


class ZeroValueAgent(GoalConditionedAgentBase):
    def act(self, obs, goal_obs, eval_mode=True):
        del obs, goal_obs, eval_mode
        return np.zeros((1,), dtype=np.float32)

    def value(self, obs, goal_obs):
        del obs, goal_obs
        return 0.0

    def batch_value(self, obs_batch, goal_obs_batch):
        del goal_obs_batch
        return np.zeros((len(obs_batch),), dtype=np.float32)


class ConstantTerminalValue:
    def __init__(self, value=1.0):
        self.value = float(value)
        self.queries = 0

    def begin_episode(self, env, *, seed):
        del env, seed
        self.queries = 0
        return {"oracle_value_source": "test"}

    def batch_value(self, env, states):
        del env
        self.queries += len(states)
        return np.full((len(states),), self.value, dtype=np.float32)

    def end_episode(self):
        return {"oracle_value_queries": self.queries}


def test_incremental_result_writer_flushes_each_episode(tmp_path):
    writer = IncrementalResultWriter(tmp_path)
    record = {
        "method": "qrl_greedy",
        "model_run": "qrl_0",
        "device_id": "device_01",
        "episode_seed": 17,
        "success": 1.0,
        "num_steps": 12.0,
    }
    writer.write(record)

    csv_text = (tmp_path / "baseline_results.partial.csv").read_text(encoding="utf-8")
    jsonl_text = (tmp_path / "baseline_results.partial.jsonl").read_text(encoding="utf-8")
    progress = json.loads(
        (tmp_path / "baseline_progress.json").read_text(encoding="utf-8")
    )
    assert "qrl_greedy" in csv_text
    assert '"device_id": "device_01"' in jsonl_text
    assert progress["status"] == "running"
    assert progress["completed_records"] == 1

    writer.mark_complete()
    writer.close()
    progress = json.loads(
        (tmp_path / "baseline_progress.json").read_text(encoding="utf-8")
    )
    assert progress["status"] == "complete"


def test_incremental_result_writer_resume_preserves_existing_records(tmp_path):
    first = {
        "method": "mppi_no_terminal",
        "model_run": "model",
        "device_id": "device_01",
        "episode_seed": 17,
        "success": 1.0,
    }
    writer = IncrementalResultWriter(tmp_path)
    writer.write(first)
    writer.close()

    resumed = IncrementalResultWriter(tmp_path, resume=True)
    assert resumed.existing_records == [first]
    assert len(resumed.completed_keys) == 1
    second = {**first, "episode_seed": 18, "success": 0.0}
    resumed.write(second)
    resumed.close()

    jsonl_records = [
        json.loads(line)
        for line in (tmp_path / "baseline_results.partial.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert jsonl_records == [first, second]
    assert len(
        (tmp_path / "baseline_results.partial.csv")
        .read_text(encoding="utf-8")
        .splitlines()
    ) == 3


def test_evaluate_controller_skips_completed_episode(tmp_path):
    env = make_env()
    completed_key = ("mppi_no_terminal", "model", "device_01", 17)
    controller = MPPIController(
        MPPIConfig(horizon=2, num_samples=2),
        terminal_mode="none",
    )
    records = _evaluate_controller(
        "mppi_no_terminal",
        controller,
        env,
        [("device_01", 17)],
        model_run="model",
        output_dir=tmp_path,
        viz_cfg=type(
            "VizConfig",
            (),
            {
                "save_visualizations": False,
                "max_successes": 0,
                "max_failures": 0,
            },
        )(),
        counters={},
        completed_keys={completed_key},
    )
    assert records == []


def test_formal_baselines_do_not_require_point_goal():
    env = make_env()
    obs, _ = env.reset(seed=3)
    assert env.goal is None
    goal_obs = env.abstract_goal_observation()

    astar = HybridAStarController(
        HybridAStarConfig(max_expansions=5000, timeout_sec=2.0, terminal_samples=8)
    )
    diagnostics = astar.begin_episode(env, goal_obs, seed=3)
    assert diagnostics["planner_success"]
    action, _ = astar.act(obs, env)
    assert env.action_space.contains(action)

    mppi = MPPIController(
        MPPIConfig(horizon=4, num_samples=8, terminal_samples=4),
        terminal_mode="model",
    )
    mppi.begin_episode(env, goal_obs, seed=3)
    action, _ = mppi.act(obs, env)
    assert env.action_space.contains(action)


def test_same_seed_produces_same_task_and_start():
    env_a = make_env(
        start=None,
        observation_radius=1.5,
        comm_threshold=0.0,
    )
    env_b = make_env(
        start=None,
        observation_radius=1.5,
        comm_threshold=0.0,
    )
    env_a.reset(seed=17)
    env_b.reset(seed=17)
    assert np.allclose(env_a.state, env_b.state)
    assert np.allclose(env_a.inspection_target, env_b.inspection_target)
    assert np.allclose(env_a.ground_station, env_b.ground_station)


def test_hybrid_astar_and_dijkstra_find_same_straight_path_cost():
    env = make_env()
    env.reset(seed=0)
    goal_obs = env.abstract_goal_observation()
    common = dict(
        position_resolution=0.2,
        heading_bins=24,
        primitive_steps=4,
        max_expansions=10_000,
        timeout_sec=3.0,
        terminal_samples=0,
    )
    astar = HybridAStarController(HybridAStarConfig(**common, heuristic_weight=1.0))
    dijkstra = HybridAStarController(HybridAStarConfig(**common, heuristic_weight=0.0))
    astar_result = astar.begin_episode(env, goal_obs, seed=0)
    dijkstra_result = dijkstra.begin_episode(env, goal_obs, seed=0)
    assert astar_result["planner_success"]
    assert dijkstra_result["planner_success"]
    assert np.isclose(astar_result["planned_cost"], dijkstra_result["planned_cost"], atol=1e-6)


def test_hybrid_astar_path_does_not_cross_obstacle():
    env = make_env(
        obstacles=[CircleObstacle(3.5, 4.0, 0.45)],
        observation_radius=0.8,
        max_steps=120,
    )
    obs, _ = env.reset(seed=0)
    controller = HybridAStarController(
        HybridAStarConfig(
            position_resolution=0.25,
            heading_bins=24,
            primitive_steps=4,
            max_expansions=30_000,
            timeout_sec=5.0,
            terminal_samples=16,
        )
    )
    result = controller.begin_episode(env, env.abstract_goal_observation(), seed=0)
    assert result["planner_success"]
    assert result["terminal_sample_count"] == 16
    done = truncated = False
    while not (done or truncated):
        action, _ = controller.act(obs, env)
        obs, _reward, done, truncated, info = env.step(action)
        assert not info["collision"]
    assert info["success"]


def test_mppi_vectorized_rollout_matches_environment_step_cost():
    env = make_env(
        obstacles=[CircleObstacle(6.5, 6.5, 0.3)],
        observation_radius=1.2,
        comm_threshold=0.5,
    )
    env.reset(seed=5)
    actions = np.array([0.2, -0.5, 1.0, 0.0], dtype=np.float32)
    modeled = simulate_action_sequences(env, env.state, actions[None, :])
    scalar_cost = 0.0
    for action in actions:
        _obs, _reward, done, truncated, info = env.step(
            np.array([action], dtype=np.float32)
        )
        scalar_cost += float(info["cost_total"])
        if done or truncated:
            break
    assert np.allclose(modeled["final_states"][0], env.state, atol=1e-6)
    assert np.isclose(modeled["costs"][0], scalar_cost, atol=1e-5)


def test_mppi_variants_use_identical_rollout_budget():
    env = make_env(observation_radius=1.2)
    obs, _ = env.reset(seed=8)
    goal_obs = env.abstract_goal_observation()
    cfg = MPPIConfig(horizon=4, num_samples=11, terminal_samples=5)
    model = MPPIController(cfg, terminal_mode="model")
    qrl = MPPIController(cfg, terminal_mode="qrl", qrl_agent=ZeroValueAgent())
    none = MPPIController(cfg, terminal_mode="none")
    oracle_value = ConstantTerminalValue()
    oracle = MPPIController(
        cfg,
        terminal_mode="oracle",
        terminal_value_provider=oracle_value,
    )
    model.begin_episode(env, goal_obs, seed=8)
    qrl.begin_episode(env, goal_obs, seed=8)
    none.begin_episode(env, goal_obs, seed=8)
    oracle.begin_episode(env, goal_obs, seed=8)
    _action, model_diag = model.act(obs, env)
    _action, qrl_diag = qrl.act(obs, env)
    _action, none_diag = none.act(obs, env)
    _action, oracle_diag = oracle.act(obs, env)
    assert (
        model_diag["model_rollouts"]
        == qrl_diag["model_rollouts"]
        == none_diag["model_rollouts"]
        == oracle_diag["model_rollouts"]
        == 11
    )


def test_hybrid_astar_value_oracle_builds_and_reuses_exact_lattice_table(tmp_path):
    env = make_env(bounds=(0.0, 0.0, 6.0, 6.0), start=(2.0, 4.0, 0.0))
    env.reset(seed=11)
    config = HybridAStarConfig(
        position_resolution=0.5,
        heading_bins=12,
        primitive_steps=2,
        primitive_scales=(-1.0, 0.0, 1.0),
        terminal_samples=0,
    )
    oracle = HybridAStarValueOracle(config, cache_dir=tmp_path)
    diagnostics = oracle.begin_episode(env, seed=11)
    terminal = env.sample_task_terminal_state(seed=91)
    values = oracle.batch_value(env, np.stack([env.state, terminal]))

    assert diagnostics["oracle_value_source"] == "hybrid_astar_lattice_reverse_dijkstra"
    assert diagnostics["oracle_value_cache_hit"] is False
    assert diagnostics["oracle_value_reachable_fraction"] > 0.0
    assert np.isfinite(values[0]) and values[0] > 0.0
    assert values[1] == 0.0

    reloaded = HybridAStarValueOracle(config, cache_dir=tmp_path)
    reload_diagnostics = reloaded.begin_episode(env, seed=11)
    assert reload_diagnostics["oracle_value_cache_hit"] is True
    assert np.allclose(reloaded.batch_value(env, np.stack([env.state, terminal])), values)


def test_goal_set_sac_action_update_and_abstract_goal_replay():
    env = make_env(observation_radius=1.2)
    obs, _ = env.reset(seed=2)
    assert env.goal is None
    goal_obs = env.abstract_goal_observation()
    cfg = GoalSetSACConfig(hidden_dim=32, batch_size=4, replay_size=16)
    agent = GoalSetSACAgent(env, cfg, torch.device("cpu"))
    action = agent.act(obs, goal_obs, eval_mode=False)
    assert env.action_space.contains(action)

    replay = GoalSetReplayBuffer(
        agent.obs_dim, agent.goal_dim, agent.act_dim, 16, torch.device("cpu")
    )
    for _ in range(6):
        next_obs, reward, terminated, truncated, _ = env.step(action)
        replay.add(obs, goal_obs, action, reward, next_obs, terminated)
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset(seed=3)
            goal_obs = env.abstract_goal_observation()
    batch = replay.sample(4)
    assert torch.allclose(
        batch["goal"],
        torch.as_tensor(goal_obs[None]).repeat(4, 1),
    )
    stats = agent.update(batch)
    assert all(np.isfinite(value) for value in stats.values())
    obs_t = torch.as_tensor(obs[None], dtype=torch.float32)
    goal_t = torch.as_tensor(goal_obs[None], dtype=torch.float32)
    sampled_action, log_prob = agent._sample_action(obs_t, goal_t, deterministic=False)
    assert torch.isfinite(sampled_action).all()
    assert torch.isfinite(log_prob).all()
