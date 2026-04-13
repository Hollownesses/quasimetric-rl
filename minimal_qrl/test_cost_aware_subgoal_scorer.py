#!/usr/bin/env python3
"""
CostAwareSubgoalScorer 单元测试。
"""
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from minimal_qrl.cost_aware_subgoal_scorer import (
    CostAwareSubgoalScorer,
    TOP_MODEL_COST_CONTEXT_KEYS,
    build_cost_aware_candidate_bundle,
    build_cost_aware_inputs,
    build_top_model_cost_context,
    evaluate_candidate_rollout_labels,
    load_cost_aware_subgoal_scorer_checkpoint,
    rollout_cost_label_for_subgoal,
    save_cost_aware_subgoal_scorer_checkpoint,
    select_cost_aware_subgoal,
)
from minimal_qrl.envs import CommInspectionDubinsUAV2D
from minimal_qrl.eval.dubins_execution_mode_eval import DubinsLookaheadConfig
from minimal_qrl.gc_agents import GoalConditionedAgentBase


class ZeroTurnAgent(GoalConditionedAgentBase):
    def act(self, obs: np.ndarray, goal_obs: np.ndarray, eval_mode: bool = True) -> np.ndarray:
        _ = obs, goal_obs, eval_mode
        return np.array([0.0], dtype=np.float32)

    def value(self, obs: np.ndarray, goal_obs: np.ndarray) -> float:
        _ = obs, goal_obs
        return 0.0


class FixedSubgoalActor:
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


class TableScorer:
    def __init__(self, table: np.ndarray):
        self.table = np.asarray(table, dtype=np.float32)

    def predict_costs(
        self,
        obs: np.ndarray,
        goal_obs: np.ndarray,
        cand_obs: np.ndarray,
        nav_features: np.ndarray,
        cost_context: np.ndarray,
        *,
        device: torch.device,
    ) -> np.ndarray:
        _ = obs, goal_obs, cand_obs, nav_features, cost_context, device
        return self.table.copy()


def make_env(*, start=None, goal=None, collision_cost: float = 10.0) -> CommInspectionDubinsUAV2D:
    return CommInspectionDubinsUAV2D(
        bounds=(0.0, 0.0, 10.0, 10.0),
        omega_max=1.0,
        v=1.0,
        dt=0.1,
        max_steps=12,
        observation_mode="task_context",
        start=start,
        goal=goal,
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
        collision_cost=collision_cost,
    )


def test_cost_aware_scorer_forward_and_checkpoint(tmp_path: Path):
    env = make_env()
    agent = ZeroTurnAgent()
    scorer = CostAwareSubgoalScorer(obs_dim=int(env.observation_space.shape[0]), hidden_dim=32)
    lookahead_cfg = DubinsLookaheadConfig(use_env_stage_cost=True)

    obs, _ = env.reset(seed=0)
    goal_obs = env.state_to_observation(np.asarray(env.goal, dtype=np.float32))
    candidates = np.stack(
        [
            np.asarray(env.goal, dtype=np.float32),
            env.sample_valid_state(seed=123),
        ],
        axis=0,
    ).astype(np.float32)
    inputs = build_cost_aware_inputs(scorer, agent, env, obs, goal_obs, candidates, lookahead_cfg)
    pred = scorer(
        torch.as_tensor(inputs["obs"], dtype=torch.float32),
        torch.as_tensor(inputs["goal_obs"], dtype=torch.float32),
        torch.as_tensor(inputs["cand_obs"], dtype=torch.float32),
        torch.as_tensor(inputs["nav_features"], dtype=torch.float32),
        torch.as_tensor(inputs["cost_context"], dtype=torch.float32),
    )

    assert pred.shape == (2,)
    assert inputs["nav_features"].shape == (2, 8)
    assert inputs["cost_context"].shape == (2, len(TOP_MODEL_COST_CONTEXT_KEYS))
    assert build_top_model_cost_context(env, lookahead_cfg).shape == (len(TOP_MODEL_COST_CONTEXT_KEYS),)

    ckpt_path = tmp_path / "cost_aware_scorer.pth"
    save_cost_aware_subgoal_scorer_checkpoint(
        str(ckpt_path),
        scorer,
        train_step=7,
        metadata={"marker": 42},
    )
    loaded, meta = load_cost_aware_subgoal_scorer_checkpoint(str(ckpt_path), torch.device("cpu"))
    loaded_pred = loaded.predict_costs(
        inputs["obs"],
        inputs["goal_obs"],
        inputs["cand_obs"],
        inputs["nav_features"],
        inputs["cost_context"],
        device=torch.device("cpu"),
    )

    assert meta["marker"] == 42
    assert np.allclose(pred.detach().cpu().numpy(), loaded_pred)


def test_rollout_cost_label_matches_single_step_cost_plus_terminal_term():
    env = make_env(
        start=(4.0, 5.0, 0.0),
        goal=(6.0, 5.0, 0.0),
    )
    agent = ZeroTurnAgent()
    lookahead_cfg = DubinsLookaheadConfig(
        horizon=1,
        num_sequences=1,
        biased_sequences=1,
        bias_kp=2.0,
        alpha_subgoal=0.0,
        alpha_final=0.0,
        alpha_task_terminal=0.5,
        use_env_stage_cost=True,
    )

    obs, _ = env.reset(seed=0)
    goal_obs = env.state_to_observation(np.asarray(env.goal, dtype=np.float32))
    subgoal = np.array([6.0, 5.0, 0.0], dtype=np.float32)
    base_state = env.get_state()
    result = rollout_cost_label_for_subgoal(
        agent,
        env,
        goal_obs,
        lookahead_cfg,
        subgoal_state=subgoal,
        rollout_steps=1,
        base_state=base_state,
    )

    env.set_state(base_state)
    _obs, _reward, terminated, _truncated, info = env.step(np.array([0.0], dtype=np.float32))
    expected = float(info["cost_total"])
    if not terminated:
        expected -= float(lookahead_cfg.alpha_task_terminal) * float(env.compute_task_score(env.state))

    assert np.isclose(result["rollout_cost_label"], expected, atol=1e-6)


def test_select_cost_aware_subgoal_matches_lowest_rollout_label():
    env = make_env()
    agent = ZeroTurnAgent()
    actor = FixedSubgoalActor(np.array([11.0, 11.0, 0.0], dtype=np.float32))
    lookahead_cfg = DubinsLookaheadConfig(
        horizon=3,
        num_sequences=8,
        biased_sequences=2,
        alpha_subgoal=1.0,
        alpha_final=0.3,
        alpha_task_terminal=0.5,
        use_env_stage_cost=True,
    )

    obs, _ = env.reset(seed=5)
    goal_obs = env.state_to_observation(np.asarray(env.goal, dtype=np.float32))
    rng_seed = 17
    bundle = build_cost_aware_candidate_bundle(
        actor,
        env,
        obs,
        goal_obs,
        device=torch.device("cpu"),
        num_candidates=8,
        rng=np.random.default_rng(rng_seed),
    )
    np.random.seed(123)
    labels = evaluate_candidate_rollout_labels(
        agent,
        env,
        goal_obs,
        lookahead_cfg,
        bundle["candidates"],
        rollout_steps=2,
    )["labels"]
    scorer = TableScorer(labels)
    np.random.seed(123)
    choice = select_cost_aware_subgoal(
        actor,
        scorer,
        agent,
        env,
        obs,
        goal_obs,
        actor_device=torch.device("cpu"),
        scorer_device=torch.device("cpu"),
        lookahead_cfg=lookahead_cfg,
        num_candidates=8,
        rollout_steps=2,
        rng=np.random.default_rng(rng_seed),
        evaluate_rollout_labels=True,
    )

    assert choice["candidate_count"] == 8
    assert choice["selected_by"] == "cost_aware"
    assert np.isclose(choice["selected_rollout_label"], float(np.min(labels)), atol=1e-6)
    assert choice["top1_match"] == 1.0
