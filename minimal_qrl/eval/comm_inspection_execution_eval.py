#!/usr/bin/env python3
"""
通信感知巡检 Dubins 环境下的 QRL 执行成功率离线评估脚本。

默认评估口径：
- 固定 inspection target / ground station / obstacle 配置
- 每个 episode 随机起点、随机 task terminal goal
- 同时比较 greedy 与 lookahead 两种执行方式
- 可选保存执行轨迹 PNG / GIF
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
from matplotlib import animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quasimetric_rl.data import Dataset
from quasimetric_rl.modules import QRLConf

from minimal_qrl.dataset import create_dataset
from minimal_qrl.envs import CircleObstacle, CommInspectionDubinsUAV2D
from minimal_qrl.eval.dubins_execution_mode_eval import (
    DubinsLookaheadConfig,
)
from minimal_qrl.eval.utils import auto_device, ensure_registered_env
from minimal_qrl.gc_agents import GoalConditionedAgentBase, QRLGoalValueAdapter
from minimal_qrl.cost_aware_subgoal_scorer import (
    CostAwareSubgoalScorer,
    load_cost_aware_subgoal_scorer_checkpoint,
    select_cost_aware_subgoal,
)
from minimal_qrl.subgoal_actor import (
    SubgoalActor,
    load_subgoal_actor_checkpoint,
    select_teacher_subgoal,
)


@dataclass
class VisualizationConfig:
    save_visualizations: bool = False
    max_successes: int = 10
    max_failures: int = 10
    save_gif: bool = False
    gif_fps: int = 8


def _obstacles_from_args(args) -> List[CircleObstacle]:
    if getattr(args, "obstacles", None) and len(args.obstacles) > 0:
        vals = list(args.obstacles)
        if len(vals) % 3 != 0:
            raise ValueError("--obstacles 必须是 3 的倍数个数字 (x, y, radius)")
        return [
            CircleObstacle(x=float(vals[i]), y=float(vals[i + 1]), radius=float(vals[i + 2]))
            for i in range(0, len(vals), 3)
        ]

    config = getattr(args, "obstacle_config", "none") or "none"
    x_min, y_min, x_max, y_max = args.bounds
    cx, cy = 0.5 * (x_min + x_max), 0.5 * (y_min + y_max)
    w, h = x_max - x_min, y_max - y_min

    if config == "none":
        return []
    if config == "simple":
        return [CircleObstacle(x=cx, y=cy, radius=0.12 * min(w, h))]
    if config == "medium":
        r = 0.10 * min(w, h)
        return [
            CircleObstacle(x=x_min + 0.35 * w, y=cy, radius=r),
            CircleObstacle(x=x_min + 0.65 * w, y=cy, radius=r),
            CircleObstacle(x=cx, y=y_min + 0.3 * h, radius=r * 0.8),
        ]
    if config == "hard":
        r = 0.08 * min(w, h)
        return [
            CircleObstacle(x=x_min + 0.25 * w, y=y_min + 0.25 * h, radius=r),
            CircleObstacle(x=x_min + 0.75 * w, y=y_min + 0.25 * h, radius=r),
            CircleObstacle(x=x_min + 0.25 * w, y=y_min + 0.75 * h, radius=r),
            CircleObstacle(x=x_min + 0.75 * w, y=y_min + 0.75 * h, radius=r),
            CircleObstacle(x=cx, y=cy, radius=r * 1.2),
        ]
    raise ValueError(f"未知的 obstacle_config: {config}")


def make_comm_inspection_env(args) -> CommInspectionDubinsUAV2D:
    env = CommInspectionDubinsUAV2D(
        bounds=tuple(float(v) for v in args.bounds),
        omega_max=float(args.omega_max),
        v=float(args.v),
        dt=float(args.dt),
        max_steps=int(args.max_episode_steps),
        observation_mode=str(args.observation_mode),
        obstacles=_obstacles_from_args(args),
        inspection_target=tuple(float(v) for v in args.inspection_target),
        ground_station=tuple(float(v) for v in args.ground_station),
        randomize_inspection_target=bool(args.randomize_inspection_target),
        randomize_ground_station=bool(args.randomize_ground_station),
        observation_radius=float(args.observation_radius),
        fov_angle=float(args.fov_angle),
        require_target_los=bool(args.require_target_los),
        comm_alpha=float(args.comm_alpha),
        comm_bias=float(args.comm_bias),
        comm_occlusion_penalty=float(args.comm_occlusion_penalty),
        comm_threshold=float(args.comm_threshold),
        require_ground_station_los=bool(args.require_ground_station_los),
        goal_sampling_mode=str(args.goal_sampling_mode),
        goal_position_tolerance=float(args.goal_position_tolerance),
        goal_heading_tolerance=float(args.goal_heading_tolerance),
        collision_cost=abs(float(args.collision_cost)),
        out_of_bounds_cost=abs(float(args.out_of_bounds_cost)),
        communication_break_cost=abs(float(args.communication_break_cost)),
        observation_violation_cost_weight=float(args.observation_violation_cost_weight),
        communication_violation_cost_weight=float(args.communication_violation_cost_weight),
        observation_failure_cost=abs(float(args.observation_failure_cost)),
        taskscore_beta_obs=float(args.taskscore_beta_obs),
        taskscore_beta_comm=float(args.taskscore_beta_comm),
        taskscore_beta_feas=float(args.taskscore_beta_feas),
        taskscore_margin_clip=float(args.taskscore_margin_clip),
    )
    try:
        env.sample_goal(seed=int(args.seed))
    except RuntimeError as exc:
        raise ValueError(
            "当前通信巡检环境配置下不存在可采样的 task terminal goal。"
            "请检查 inspection_target / ground_station / obstacle_config / "
            "observation_radius / fov_angle / comm_threshold 等参数。"
        ) from exc
    return env


def build_qrl_adapter(
    args,
    device: torch.device,
    env: CommInspectionDubinsUAV2D,
) -> tuple[GoalConditionedAgentBase, Optional[int]]:
    env_name = str(args.env_name)

    def create_env_fn():
        return make_comm_inspection_env(args)

    def load_episodes_fn():
        e = create_env_fn()
        return create_dataset(e, num_episodes=1, max_steps_per_episode=10, seed=int(args.seed))

    ensure_registered_env(
        "comm_inspection_dubins_uav",
        env_name,
        create_env_fn=create_env_fn,
        load_episodes_fn=load_episodes_fn,
    )

    dataset_conf = Dataset.Conf(
        kind="comm_inspection_dubins_uav",
        name=env_name,
        future_observation_discount=0.99,
    )
    dataset = dataset_conf.make(dummy=True)
    env_spec = dataset.env_spec

    qrl_conf = QRLConf(actor=None, num_critics=int(args.num_critics))
    qrl_agent, _ = qrl_conf.make(env_spec=env_spec, total_optim_steps=1)

    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_step: Optional[int] = None
    state_dict = ckpt
    if isinstance(ckpt, dict) and "agent" in ckpt:
        state_dict = ckpt["agent"]
        ckpt_step = ckpt.get("optim_steps")

    try:
        qrl_agent.load_state_dict(state_dict)
    except RuntimeError as exc:
        obs_dim = int(env.observation_space.shape[0])
        act_dim = int(env.action_space.shape[0])
        raise ValueError(
            "checkpoint 与当前通信巡检环境的 observation/action 维度不匹配。"
            f" 当前环境 obs_dim={obs_dim}, act_dim={act_dim}, "
            f"observation_mode={env.observation_mode}。原始错误: {exc}"
        ) from exc

    qrl_agent.to(device)
    qrl_agent.eval()
    return QRLGoalValueAdapter(qrl_agent, env=env, device=device), ckpt_step


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _evaluate_comm_lookahead_sequences(
    agent: GoalConditionedAgentBase,
    env: CommInspectionDubinsUAV2D,
    goal_obs: np.ndarray,
    cfg: DubinsLookaheadConfig,
    omegas: np.ndarray,
    base_state: dict,
    *,
    subgoal_state: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(omegas.shape[0])
    costs = np.zeros((n,), dtype=np.float32)
    first_actions = omegas[:, 0].astype(np.float32).copy()
    subgoal_obs = env.state_to_observation(subgoal_state) if subgoal_state is not None else None

    for i in range(n):
        env.set_state(base_state)
        total_cost = 0.0
        reached_subgoal = False
        success = False

        for t in range(int(omegas.shape[1])):
            w = float(omegas[i, t])
            action = np.array([w], dtype=np.float32)
            _obs, _reward, terminated, truncated, info = env.step(action)

            if bool(cfg.use_env_stage_cost):
                total_cost += _safe_float(info.get("cost_total"))
            else:
                if cfg.step_cost_weight > 0.0:
                    total_cost += float(cfg.step_cost_weight) * abs(w)
                if cfg.collision_penalty > 0.0 and bool(info.get("collision", False)):
                    total_cost += float(cfg.collision_penalty)

            if subgoal_state is not None and env.is_subgoal_reached(
                env.state,
                subgoal_state,
                pos_tolerance=float(cfg.subgoal_reached_pos_tolerance),
                theta_tolerance=float(cfg.subgoal_reached_theta_tolerance),
            ):
                reached_subgoal = True

            if terminated:
                success = True
                break
            if truncated:
                break

        terminal_obs = env.state_to_observation(env.state)
        terminal_task_score = env.compute_task_score(env.state)
        terminal_cost = 0.0
        if not success:
            terminal_cost += float(cfg.alpha_final) * float(agent.value(terminal_obs, goal_obs))
            terminal_cost -= float(cfg.alpha_task_terminal) * float(terminal_task_score)
            if subgoal_obs is not None and not reached_subgoal:
                terminal_cost += float(cfg.alpha_subgoal) * float(agent.value(terminal_obs, subgoal_obs))

        costs[i] = float(total_cost + terminal_cost)

    env.set_state(base_state)
    return costs, first_actions


def _comm_inspection_lookahead_action(
    agent: GoalConditionedAgentBase,
    env: CommInspectionDubinsUAV2D,
    goal_obs: np.ndarray,
    cfg: DubinsLookaheadConfig,
    *,
    subgoal_state: Optional[np.ndarray] = None,
) -> np.ndarray:
    horizon = max(1, int(cfg.horizon))
    num_sequences = max(1, int(cfg.num_sequences))
    base_state = env.get_state()

    low = float(env.action_space.low[0])
    high = float(env.action_space.high[0])
    n_bias = int(min(max(0, cfg.biased_sequences), num_sequences))
    n_rand = int(max(0, num_sequences - n_bias))
    rand = (
        np.random.uniform(low, high, size=(n_rand, horizon)).astype(np.float32)
        if n_rand > 0
        else np.zeros((0, horizon), dtype=np.float32)
    )
    if n_bias > 0:
        bias = np.zeros((n_bias, horizon), dtype=np.float32)
        desired = subgoal_state if subgoal_state is not None else np.asarray(env.goal, dtype=np.float32)
        dx = float(desired[0] - env.state[0])
        dy = float(desired[1] - env.state[1])
        err = env._normalize_angle(float(np.arctan2(dy, dx) - env.state[2]))
        w0 = float(np.clip(float(cfg.bias_kp) * err, low, high))
        bias[0, :] = w0
        for idx in range(1, n_bias):
            scale = max(0.0, 1.0 - 0.15 * float(idx))
            bias[idx, :] = float(np.clip(w0 * scale, low, high))
    else:
        bias = np.zeros((0, horizon), dtype=np.float32)
    omegas0 = np.concatenate([bias, rand], axis=0) if (n_bias + n_rand) > 0 else np.zeros((1, horizon), dtype=np.float32)

    if cfg.use_cem:
        om_range = float(high - low)
        std = np.full((horizon,), float(cfg.cem_std_init_frac) * 0.5 * om_range, dtype=np.float32)
        mean = np.mean(omegas0, axis=0).astype(np.float32) if omegas0.shape[0] > 0 else np.zeros((horizon,), dtype=np.float32)
        n_elite = max(1, int(float(cfg.cem_elite_frac) * float(num_sequences)))
        best_first = np.array([0.0], dtype=np.float32)
        best_cost = float("inf")

        for _ in range(max(1, int(cfg.cem_iters))):
            samples = np.random.normal(loc=mean[None, :], scale=std[None, :], size=(num_sequences, horizon)).astype(np.float32)
            samples = np.clip(samples, low, high)
            if n_bias > 0:
                samples[: min(n_bias, samples.shape[0])] = bias[: min(n_bias, samples.shape[0])]
            costs, firsts = _evaluate_comm_lookahead_sequences(
                agent,
                env,
                goal_obs,
                cfg,
                samples,
                base_state,
                subgoal_state=subgoal_state,
            )
            idx = int(np.argmin(costs))
            if float(costs[idx]) < best_cost:
                best_cost = float(costs[idx])
                best_first = np.array([float(firsts[idx])], dtype=np.float32)
            elite = samples[np.argsort(costs)[:n_elite]]
            mean = np.mean(elite, axis=0).astype(np.float32)
            std = (np.std(elite, axis=0) + 1e-4).astype(np.float32)

        env.set_state(base_state)
        return best_first

    costs, firsts = _evaluate_comm_lookahead_sequences(
        agent,
        env,
        goal_obs,
        cfg,
        omegas0,
        base_state,
        subgoal_state=subgoal_state,
    )
    best_idx = int(np.argmin(costs))
    env.set_state(base_state)
    return np.array([float(firsts[best_idx])], dtype=np.float32)


def _choose_hierarchical_subgoal(
    actor: SubgoalActor,
    agent: GoalConditionedAgentBase,
    env: CommInspectionDubinsUAV2D,
    obs: np.ndarray,
    goal_obs: np.ndarray,
    *,
    device: torch.device,
    num_candidates: int,
    lambda_final: float,
    lambda_task: float,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    choice = select_teacher_subgoal(
        actor,
        agent,
        env,
        obs,
        goal_obs,
        device=device,
        num_candidates=num_candidates,
        lambda_final=lambda_final,
        lambda_task=lambda_task,
        rng=rng,
    )
    repaired_state = np.asarray(choice["repaired_subgoal"], dtype=np.float32)
    teacher_state = np.asarray(choice["teacher_subgoal"], dtype=np.float32)
    use_teacher_fallback = bool(choice.get("used_global_repair_fallback", False))
    executed_subgoal = teacher_state if use_teacher_fallback else repaired_state
    return {
        **choice,
        "executed_subgoal": executed_subgoal,
        "used_teacher_fallback": bool(use_teacher_fallback),
        "executed_task_score": float(env.compute_task_score(executed_subgoal)),
        "selected_by": "heuristic",
        "selected_pred_cost": None,
        "selected_rollout_label": None,
        "top1_match": None,
        "eval_mse": None,
    }


def rollout_execution_episode(
    agent: GoalConditionedAgentBase,
    env: CommInspectionDubinsUAV2D,
    execution_mode: str,
    *,
    episode_seed: int,
    lookahead_cfg: Optional[DubinsLookaheadConfig],
    subgoal_actor: Optional[SubgoalActor] = None,
    top_model: Optional[CostAwareSubgoalScorer] = None,
    actor_device: Optional[torch.device] = None,
    high_level_period: int = 5,
    subgoal_candidates: int = 64,
    subgoal_lambda_final: float = 0.3,
    subgoal_lambda_task: float = 1.0,
    subgoal_selector: str = "heuristic",
    top_model_rollout_steps: Optional[int] = None,
) -> Dict[str, Any]:
    if execution_mode not in {"greedy", "lookahead", "hierarchical"}:
        raise ValueError(f"未知 execution_mode: {execution_mode}")
    if execution_mode == "lookahead" and lookahead_cfg is None:
        raise ValueError("lookahead 模式需要 lookahead_cfg")
    if execution_mode == "hierarchical" and (lookahead_cfg is None or subgoal_actor is None or actor_device is None):
        raise ValueError("hierarchical 模式需要 lookahead_cfg、subgoal_actor 和 actor_device")
    if subgoal_selector not in {"heuristic", "cost_aware"}:
        raise ValueError(f"未知 subgoal_selector: {subgoal_selector}")
    if execution_mode == "hierarchical" and subgoal_selector == "cost_aware" and top_model is None:
        raise ValueError("hierarchical + cost_aware 需要 top_model")

    np.random.seed(int(episode_seed))
    obs, reset_info = env.reset(seed=int(episode_seed))
    goal_obs = env.state_to_observation(np.asarray(env.goal, dtype=np.float32))
    rng = np.random.default_rng(int(episode_seed))

    states: List[np.ndarray] = [env.state.copy()]
    actions: List[np.ndarray] = []
    rewards: List[float] = []
    step_infos: List[Dict[str, Any]] = []
    task_flags: List[bool] = [bool(reset_info.get("task_feasible", False))]
    high_level_events: List[Dict[str, Any]] = []

    done = False
    truncated = False
    collided = False
    out_of_bounds = False
    final_info: Dict[str, Any] = dict(reset_info)
    current_subgoal: Optional[np.ndarray] = None

    while not (done or truncated):
        if execution_mode == "lookahead":
            action = _comm_inspection_lookahead_action(agent, env, goal_obs, lookahead_cfg)
        elif execution_mode == "hierarchical":
            need_replan = current_subgoal is None or (
                len(actions) % max(1, int(high_level_period)) == 0
            )
            if current_subgoal is not None and env.is_subgoal_reached(
                env.state,
                current_subgoal,
                pos_tolerance=float(lookahead_cfg.subgoal_reached_pos_tolerance),
                theta_tolerance=float(lookahead_cfg.subgoal_reached_theta_tolerance),
            ):
                need_replan = True

            if need_replan:
                if subgoal_selector == "cost_aware":
                    subgoal_choice = select_cost_aware_subgoal(
                        subgoal_actor,
                        top_model,
                        agent,
                        env,
                        obs,
                        goal_obs,
                        actor_device=actor_device,
                        scorer_device=actor_device,
                        lookahead_cfg=lookahead_cfg,
                        num_candidates=int(subgoal_candidates),
                        rollout_steps=(
                            int(top_model_rollout_steps)
                            if top_model_rollout_steps is not None
                            else int(high_level_period)
                        ),
                        rng=rng,
                        evaluate_rollout_labels=True,
                    )
                else:
                    subgoal_choice = _choose_hierarchical_subgoal(
                        subgoal_actor,
                        agent,
                        env,
                        obs,
                        goal_obs,
                        device=actor_device,
                        num_candidates=int(subgoal_candidates),
                        lambda_final=float(subgoal_lambda_final),
                        lambda_task=float(subgoal_lambda_task),
                        rng=rng,
                    )
                current_subgoal = np.asarray(subgoal_choice["executed_subgoal"], dtype=np.float32)
                high_level_events.append(
                    {
                        "step": int(len(actions)),
                        "raw_subgoal": [float(v) for v in np.asarray(subgoal_choice["raw_subgoal"], dtype=np.float32)],
                        "repaired_subgoal": [float(v) for v in np.asarray(subgoal_choice["repaired_subgoal"], dtype=np.float32)],
                        "executed_subgoal": [float(v) for v in np.asarray(current_subgoal, dtype=np.float32)],
                        "raw_valid": bool(subgoal_choice["raw_valid"]),
                        "used_nearby_repair": bool(subgoal_choice.get("used_nearby_repair", False)),
                        "used_global_repair_fallback": bool(subgoal_choice.get("used_global_repair_fallback", False)),
                        "repair_distance": float(subgoal_choice["repair_distance"]),
                        "repair_dtheta": float(subgoal_choice["repair_dtheta"]),
                        "raw_task_score": float(subgoal_choice["raw_task_score"]),
                        "repaired_task_score": float(subgoal_choice["repaired_task_score"]),
                        "executed_task_score": float(subgoal_choice["executed_task_score"]),
                        "used_teacher_fallback": bool(subgoal_choice["used_teacher_fallback"]),
                        "candidate_count": int(subgoal_choice.get("candidate_count", subgoal_candidates)),
                        "selected_by": str(subgoal_choice.get("selected_by", subgoal_selector)),
                        "selected_pred_cost": (
                            None
                            if subgoal_choice.get("selected_pred_cost") is None
                            else float(subgoal_choice["selected_pred_cost"])
                        ),
                        "selected_rollout_label": (
                            None
                            if subgoal_choice.get("selected_rollout_label") is None
                            else float(subgoal_choice["selected_rollout_label"])
                        ),
                        "top_model_top1_match": (
                            None
                            if subgoal_choice.get("top1_match") is None
                            else float(subgoal_choice["top1_match"])
                        ),
                        "top_model_eval_mse": (
                            None
                            if subgoal_choice.get("eval_mse") is None
                            else float(subgoal_choice["eval_mse"])
                        ),
                    }
                )

            action = _comm_inspection_lookahead_action(
                agent,
                env,
                goal_obs,
                lookahead_cfg,
                subgoal_state=current_subgoal,
            )
        else:
            action = agent.act(obs, goal_obs, eval_mode=True)

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        obs, reward, done, truncated, step_info = env.step(action)
        step_info = dict(step_info)
        if current_subgoal is not None:
            step_info["active_subgoal"] = [float(v) for v in np.asarray(current_subgoal, dtype=np.float32)]
            step_info["subgoal_reached"] = bool(
                env.is_subgoal_reached(
                    env.state,
                    current_subgoal,
                    pos_tolerance=float(lookahead_cfg.subgoal_reached_pos_tolerance),
                    theta_tolerance=float(lookahead_cfg.subgoal_reached_theta_tolerance),
                )
            )

        actions.append(action.copy())
        rewards.append(float(reward))
        step_infos.append(step_info)
        states.append(env.state.copy())
        task_flags.append(bool(step_info.get("task_feasible", False)))

        final_info = step_info
        collided = collided or bool(step_info.get("collision", False))
        out_of_bounds = out_of_bounds or bool(step_info.get("out_of_bounds", False))

    return {
        "execution_mode": execution_mode,
        "seed": int(episode_seed),
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "step_infos": step_infos,
        "task_flags": task_flags,
        "success": bool(done),
        "truncated": bool(truncated),
        "num_steps": int(len(actions)),
        "collision": bool(collided),
        "out_of_bounds": bool(out_of_bounds),
        "initial_info": dict(reset_info),
        "final_info": dict(final_info),
        "start": np.asarray(env.start, dtype=np.float32).copy(),
        "goal": np.asarray(env.goal, dtype=np.float32).copy(),
        "inspection_target": np.asarray(env.inspection_target, dtype=np.float32).copy(),
        "ground_station": np.asarray(env.ground_station, dtype=np.float32).copy(),
        "high_level_events": high_level_events,
    }


def _add_heading_arrow(ax, state: np.ndarray, color: str, *, length: float = 0.45) -> None:
    ax.arrow(
        float(state[0]),
        float(state[1]),
        length * np.cos(float(state[2])),
        length * np.sin(float(state[2])),
        head_width=0.12,
        head_length=0.10,
        fc=color,
        ec=color,
        linewidth=1.2,
        zorder=6,
        length_includes_head=True,
    )


def _compute_feasibility_masks(
    env: CommInspectionDubinsUAV2D,
    theta: float,
    *,
    resolution: int = 140,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(env.x_min, env.x_max, resolution)
    ys = np.linspace(env.y_min, env.y_max, resolution)
    obs_mask = np.zeros((resolution, resolution), dtype=np.float32)
    comm_mask = np.zeros((resolution, resolution), dtype=np.float32)
    task_mask = np.zeros((resolution, resolution), dtype=np.float32)

    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            state = np.array([x, y, theta], dtype=np.float32)
            if not env.is_valid_state(state):
                continue
            obs_ok = env.is_observation_feasible(state)
            comm_ok = env.is_communication_feasible(state)
            obs_mask[iy, ix] = 1.0 if obs_ok else 0.0
            comm_mask[iy, ix] = 1.0 if comm_ok else 0.0
            task_mask[iy, ix] = 1.0 if (obs_ok and comm_ok) else 0.0

    return xs, ys, obs_mask, comm_mask, task_mask


def _draw_feasibility_mask(
    ax,
    env: CommInspectionDubinsUAV2D,
    mask: np.ndarray,
    *,
    cmap: str,
    alpha: float = 0.32,
) -> None:
    ax.imshow(
        mask,
        origin="lower",
        extent=[env.x_min, env.x_max, env.y_min, env.y_max],
        cmap=cmap,
        alpha=alpha,
        vmin=0.0,
        vmax=1.0,
        zorder=0,
    )


def _draw_environment_base(ax, env: CommInspectionDubinsUAV2D, rollout: Dict[str, Any]) -> None:
    ax.set_aspect("equal")
    ax.set_xlim(env.x_min - 0.2, env.x_max + 0.2)
    ax.set_ylim(env.y_min - 0.2, env.y_max + 0.2)
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    for obs in env.obstacles:
        if isinstance(obs, CircleObstacle):
            patch = patches.Circle((obs.x, obs.y), obs.radius, color="gray", alpha=0.75, zorder=1)
        elif all(hasattr(obs, attr) for attr in ("x_min", "y_min", "x_max", "y_max")):
            patch = patches.Rectangle(
                (obs.x_min, obs.y_min),
                obs.x_max - obs.x_min,
                obs.y_max - obs.y_min,
                color="gray",
                alpha=0.75,
                zorder=1,
            )
        else:
            continue
        ax.add_patch(patch)

    inspection_target = np.asarray(rollout["inspection_target"], dtype=np.float32)
    ground_station = np.asarray(rollout["ground_station"], dtype=np.float32)
    start = np.asarray(rollout["start"], dtype=np.float32)
    goal = np.asarray(rollout["goal"], dtype=np.float32)

    ax.scatter(start[0], start[1], c="green", s=80, label="start", zorder=5)
    ax.scatter(goal[0], goal[1], c="red", s=120, marker="*", label="task terminal goal", zorder=5)
    ax.scatter(
        inspection_target[0],
        inspection_target[1],
        c="gold",
        s=90,
        marker="X",
        label="inspection target",
        zorder=5,
    )
    ax.scatter(
        ground_station[0],
        ground_station[1],
        c="navy",
        s=90,
        marker="s",
        label="ground station",
        zorder=5,
    )

    obs_circle = patches.Circle(
        (float(inspection_target[0]), float(inspection_target[1])),
        float(env.observation_radius),
        fill=False,
        linestyle="--",
        linewidth=1.0,
        edgecolor="goldenrod",
        alpha=0.9,
    )
    ax.add_patch(obs_circle)
    _add_heading_arrow(ax, start, "green")
    _add_heading_arrow(ax, goal, "red")


def _make_rollout_title(
    rollout: Dict[str, Any],
    *,
    execution_mode: str,
    episode_index: int,
    frame_index: Optional[int] = None,
) -> str:
    line1 = (
        f"{execution_mode} | episode={episode_index:03d} | seed={int(rollout['seed'])} | "
        f"success={bool(rollout['success'])} | steps={int(rollout['num_steps'])}"
    )
    line2 = (
        f"collision={bool(rollout['collision'])} | "
        f"out_of_bounds={bool(rollout['out_of_bounds'])}"
    )
    if frame_index is None:
        return f"{line1}\n{line2}"

    if frame_index <= 0:
        return f"{line1}\n{line2} | step=0"

    info = rollout["step_infos"][frame_index - 1]
    return (
        f"{line1}\n"
        f"{line2} | step={frame_index} | "
        f"comm_margin={_safe_float(info.get('comm_margin')):.3f} | "
        f"obs_margin={_safe_float(info.get('obs_margin')):.3f} | "
        f"task_feasible={bool(info.get('task_feasible', False))}"
    )


def _plot_rollout_png(
    env: CommInspectionDubinsUAV2D,
    rollout: Dict[str, Any],
    out_path: Path,
    *,
    execution_mode: str,
    episode_index: int,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    states = rollout["states"]
    task_flags = rollout["task_flags"]
    traj_x = [float(s[0]) for s in states]
    traj_y = [float(s[1]) for s in states]

    goal = np.asarray(rollout["goal"], dtype=np.float32)
    theta_slice = float(goal[2])
    _, _, obs_mask, comm_mask, task_mask = _compute_feasibility_masks(env, theta_slice)

    fig, axes = plt.subplots(1, 3, figsize=(18.0, 6.2), sharex=True, sharey=True)
    panel_specs = [
        ("Observation Feasible Region", obs_mask, "Greens"),
        ("Communication Feasible Region", comm_mask, "Blues"),
        ("Joint Task Feasible Region", task_mask, "Oranges"),
    ]

    end_state = np.asarray(states[-1], dtype=np.float32)
    first_feasible_idx = next((i for i, flag in enumerate(task_flags) if flag), None)

    for panel_idx, (ax, (panel_title, mask, cmap)) in enumerate(zip(axes, panel_specs)):
        _draw_feasibility_mask(ax, env, mask, cmap=cmap)
        _draw_environment_base(ax, env, rollout)

        for idx in range(1, len(states)):
            seg_color = "darkorange" if task_flags[idx] else "black"
            seg_label = "trajectory" if idx == 1 and panel_idx == 2 else None
            ax.plot(
                traj_x[idx - 1 : idx + 1],
                traj_y[idx - 1 : idx + 1],
                color=seg_color,
                linewidth=2.0,
                label=seg_label,
                zorder=4,
            )

        end_label = "end" if panel_idx == 2 else None
        ax.scatter(end_state[0], end_state[1], c="crimson", s=70, marker="o", label=end_label, zorder=6)
        _add_heading_arrow(ax, end_state, "crimson", length=0.38)

        if first_feasible_idx is not None:
            feasible_state = np.asarray(states[first_feasible_idx], dtype=np.float32)
            marker_label = "first task-feasible state" if panel_idx == 2 else None
            ax.scatter(
                feasible_state[0],
                feasible_state[1],
                c="darkorange",
                s=75,
                marker="o",
                label=marker_label,
                zorder=6,
            )

        if rollout.get("high_level_events"):
            for ev_idx, event in enumerate(rollout["high_level_events"]):
                raw = np.asarray(event["raw_subgoal"], dtype=np.float32)
                repaired = np.asarray(event["repaired_subgoal"], dtype=np.float32)
                executed = np.asarray(event["executed_subgoal"], dtype=np.float32)
                if panel_idx == 2:
                    raw_label = "raw subgoal" if ev_idx == 0 else None
                    repaired_label = "repaired subgoal" if ev_idx == 0 else None
                    executed_label = "executed subgoal" if ev_idx == 0 else None
                else:
                    raw_label = None
                    repaired_label = None
                    executed_label = None
                ax.plot(
                    [raw[0], repaired[0]],
                    [raw[1], repaired[1]],
                    color="mediumpurple",
                    linewidth=1.0,
                    linestyle=":",
                    zorder=5,
                )
                ax.scatter(
                    raw[0],
                    raw[1],
                    c="violet",
                    s=35,
                    marker="x",
                    label=raw_label,
                    zorder=6,
                )
                ax.scatter(
                    repaired[0],
                    repaired[1],
                    c="purple",
                    s=45,
                    marker="s",
                    label=repaired_label,
                    zorder=6,
                )
                ax.scatter(
                    executed[0],
                    executed[1],
                    c="indigo",
                    s=50,
                    marker="D",
                    label=executed_label,
                    zorder=6,
                )

        ax.set_title(panel_title)

    fig.suptitle(
        _make_rollout_title(rollout, execution_mode=execution_mode, episode_index=episode_index)
        + f"\nfeasible-region slice uses goal heading theta={theta_slice:.2f} rad",
        fontsize=14,
        y=0.98,
    )
    axes[-1].legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    fig.subplots_adjust(top=0.84)
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_rollout_gif(
    env: CommInspectionDubinsUAV2D,
    rollout: Dict[str, Any],
    out_path: Path,
    *,
    execution_mode: str,
    episode_index: int,
    fps: int,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    states = rollout["states"]
    task_flags = rollout["task_flags"]
    goal = np.asarray(rollout["goal"], dtype=np.float32)
    theta_slice = float(goal[2])
    _, _, _obs_mask, _comm_mask, task_mask = _compute_feasibility_masks(env, theta_slice)

    fig, ax = plt.subplots(figsize=(8.5, 8.0))
    _draw_feasibility_mask(ax, env, task_mask, cmap="Oranges", alpha=0.28)
    _draw_environment_base(ax, env, rollout)
    ax.legend(loc="upper right", fontsize=8)

    segment_artists = []
    for idx in range(1, len(states)):
        seg_color = "darkorange" if task_flags[idx] else "black"
        (artist,) = ax.plot([], [], color=seg_color, linewidth=2.0, zorder=4)
        segment_artists.append(artist)

    current_point = ax.scatter([], [], c="crimson", s=70, marker="o", zorder=7)
    current_arrow = ax.quiver(
        [],
        [],
        [],
        [],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="crimson",
        width=0.005,
        zorder=7,
    )

    first_feasible_idx = next((i for i, flag in enumerate(task_flags) if flag), None)
    first_feasible_marker = None
    if first_feasible_idx is not None:
        feasible_state = np.asarray(states[first_feasible_idx], dtype=np.float32)
        first_feasible_marker = ax.scatter(
            [feasible_state[0]],
            [feasible_state[1]],
            c="darkorange",
            s=75,
            marker="o",
            label="first task-feasible state",
            zorder=6,
        )
        first_feasible_marker.set_visible(False)

    event_artists = []
    if rollout.get("high_level_events"):
        for event in rollout["high_level_events"]:
            raw = np.asarray(event["raw_subgoal"], dtype=np.float32)
            repaired = np.asarray(event["repaired_subgoal"], dtype=np.float32)
            executed = np.asarray(event["executed_subgoal"], dtype=np.float32)
            (repair_line,) = ax.plot(
                [raw[0], repaired[0]],
                [raw[1], repaired[1]],
                color="mediumpurple",
                linewidth=1.0,
                linestyle=":",
                zorder=5,
            )
            raw_artist = ax.scatter([raw[0]], [raw[1]], c="violet", s=35, marker="x", zorder=6)
            repaired_artist = ax.scatter([repaired[0]], [repaired[1]], c="purple", s=45, marker="s", zorder=6)
            executed_artist = ax.scatter([executed[0]], [executed[1]], c="indigo", s=50, marker="D", zorder=6)
            event_artists.extend([repair_line, raw_artist, repaired_artist, executed_artist])

    def _init():
        ax.set_title(
            _make_rollout_title(
                rollout,
                execution_mode=execution_mode,
                episode_index=episode_index,
                frame_index=0,
            )
            + f"\nJoint task feasible slice at goal theta={theta_slice:.2f} rad"
        )
        current_point.set_offsets(np.asarray([[states[0][0], states[0][1]]], dtype=np.float32))
        current_arrow.set_offsets(np.asarray([[states[0][0], states[0][1]]], dtype=np.float32))
        current_arrow.set_UVC(
            [0.35 * np.cos(float(states[0][2]))],
            [0.35 * np.sin(float(states[0][2]))],
        )
        return [*segment_artists, current_point, current_arrow, *event_artists]

    def _update(frame_idx: int):
        for idx, artist in enumerate(segment_artists, start=1):
            if idx <= frame_idx:
                artist.set_data(
                    [float(states[idx - 1][0]), float(states[idx][0])],
                    [float(states[idx - 1][1]), float(states[idx][1])],
                )
            else:
                artist.set_data([], [])

        state = np.asarray(states[frame_idx], dtype=np.float32)
        current_point.set_offsets(np.asarray([[state[0], state[1]]], dtype=np.float32))
        current_arrow.set_offsets(np.asarray([[state[0], state[1]]], dtype=np.float32))
        current_arrow.set_UVC([0.35 * np.cos(float(state[2]))], [0.35 * np.sin(float(state[2]))])

        if first_feasible_marker is not None:
            first_feasible_marker.set_visible(frame_idx >= first_feasible_idx)

        ax.set_title(
            _make_rollout_title(
                rollout,
                execution_mode=execution_mode,
                episode_index=episode_index,
                frame_index=frame_idx,
            )
            + f"\nJoint task feasible slice at goal theta={theta_slice:.2f} rad"
        )
        artists = [*segment_artists, current_point, current_arrow, *event_artists]
        if first_feasible_marker is not None:
            artists.append(first_feasible_marker)
        return artists

    anim = animation.FuncAnimation(
        fig,
        _update,
        frames=len(states),
        init_func=_init,
        interval=max(1, int(1000 / max(1, fps))),
        blit=False,
        repeat=False,
    )
    writer = animation.PillowWriter(fps=max(1, fps))
    anim.save(out_path, writer=writer)
    plt.close(fig)
    return out_path


def _save_rollout_visualization(
    env: CommInspectionDubinsUAV2D,
    rollout: Dict[str, Any],
    *,
    execution_mode: str,
    episode_index: int,
    category: str,
    category_dir: Path,
    base_output_dir: Path,
    viz_cfg: VisualizationConfig,
) -> Dict[str, Any]:
    status = "success" if bool(rollout["success"]) else "failure"
    stem = f"episode_{episode_index:03d}_seed_{int(rollout['seed'])}_{status}"
    png_path = category_dir / f"{stem}.png"
    gif_path = category_dir / f"{stem}.gif"

    _plot_rollout_png(
        env,
        rollout,
        png_path,
        execution_mode=execution_mode,
        episode_index=episode_index,
    )

    gif_error: Optional[str] = None
    if viz_cfg.save_gif:
        try:
            _plot_rollout_gif(
                env,
                rollout,
                gif_path,
                execution_mode=execution_mode,
                episode_index=episode_index,
                fps=int(viz_cfg.gif_fps),
            )
        except Exception as exc:  # pragma: no cover - 依赖环境差异较大
            gif_error = str(exc)
            print(
                f"[comm_inspection_execution_eval] GIF 保存失败: mode={execution_mode}, "
                f"episode={episode_index}, seed={int(rollout['seed'])}, error={exc}",
                file=sys.stderr,
            )

    final_info = rollout["final_info"]
    gif_saved = bool(viz_cfg.save_gif and gif_error is None and gif_path.exists())
    return {
        "category": category,
        "episode_index": int(episode_index),
        "seed": int(rollout["seed"]),
        "mode": execution_mode,
        "success": bool(rollout["success"]),
        "truncated": bool(rollout["truncated"]),
        "num_steps": int(rollout["num_steps"]),
        "collision": bool(rollout["collision"]),
        "out_of_bounds": bool(rollout["out_of_bounds"]),
        "png": os.path.relpath(png_path, base_output_dir),
        "gif": os.path.relpath(gif_path, base_output_dir) if gif_saved else None,
        "gif_error": gif_error,
        "start": [float(v) for v in np.asarray(rollout["start"], dtype=np.float32)],
        "goal": [float(v) for v in np.asarray(rollout["goal"], dtype=np.float32)],
        "inspection_target": [float(v) for v in np.asarray(rollout["inspection_target"], dtype=np.float32)],
        "ground_station": [float(v) for v in np.asarray(rollout["ground_station"], dtype=np.float32)],
        "final_comm_margin": _safe_float(final_info.get("comm_margin")),
        "final_obs_margin": _safe_float(final_info.get("obs_margin")),
        "final_task_feasible": bool(final_info.get("task_feasible", False)),
        "num_high_level_events": int(len(rollout.get("high_level_events", []))),
        "high_level_events": list(rollout.get("high_level_events", [])),
    }


def evaluate_execution_mode(
    agent: GoalConditionedAgentBase,
    env: CommInspectionDubinsUAV2D,
    execution_mode: str,
    *,
    n_trials: int,
    seed: int,
    lookahead_cfg: Optional[DubinsLookaheadConfig],
    output_dir: Path,
    viz_cfg: VisualizationConfig,
    subgoal_actor: Optional[SubgoalActor] = None,
    top_model: Optional[CostAwareSubgoalScorer] = None,
    actor_device: Optional[torch.device] = None,
    high_level_period: int = 5,
    subgoal_candidates: int = 64,
    subgoal_lambda_final: float = 0.3,
    subgoal_lambda_task: float = 1.0,
    subgoal_selector: str = "heuristic",
    top_model_rollout_steps: Optional[int] = None,
) -> tuple[Dict[str, float], Dict[str, List[Dict[str, Any]]]]:
    if execution_mode not in {"greedy", "lookahead", "hierarchical"}:
        raise ValueError(f"未知 execution_mode: {execution_mode}")
    if execution_mode == "lookahead" and lookahead_cfg is None:
        raise ValueError("lookahead 模式需要 lookahead_cfg")
    if execution_mode == "hierarchical" and (lookahead_cfg is None or subgoal_actor is None or actor_device is None):
        raise ValueError("hierarchical 模式需要 lookahead_cfg、subgoal_actor 和 actor_device")

    np.random.seed(int(seed))

    success_count = 0
    success_steps: List[int] = []
    all_steps: List[int] = []
    ever_task_feasible = 0
    first_task_feasible_steps: List[int] = []
    collision_episodes = 0
    out_of_bounds_episodes = 0
    raw_valid_rates: List[float] = []
    repair_distances: List[float] = []
    repair_dthetas: List[float] = []
    raw_task_scores: List[float] = []
    repaired_task_scores: List[float] = []
    top_model_top1_matches: List[float] = []
    top_model_eval_mses: List[float] = []
    selected_pred_costs: List[float] = []
    selected_rollout_labels: List[float] = []

    visualization_index: Dict[str, List[Dict[str, Any]]] = {
        "success": [],
        "failure": [],
    }

    vis_root = output_dir / "visualizations" / execution_mode
    success_dir = vis_root / "success"
    failure_dir = vis_root / "failure"
    if viz_cfg.save_visualizations:
        success_dir.mkdir(parents=True, exist_ok=True)
        failure_dir.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(n_trials), desc=f"{execution_mode}_success_rate", leave=False):
        rollout = rollout_execution_episode(
            agent,
            env,
            execution_mode,
            episode_seed=int(seed + i),
            lookahead_cfg=lookahead_cfg,
            subgoal_actor=subgoal_actor,
            top_model=top_model,
            actor_device=actor_device,
            high_level_period=int(high_level_period),
            subgoal_candidates=int(subgoal_candidates),
            subgoal_lambda_final=float(subgoal_lambda_final),
            subgoal_lambda_task=float(subgoal_lambda_task),
            subgoal_selector=str(subgoal_selector),
            top_model_rollout_steps=top_model_rollout_steps,
        )

        step_count = int(rollout["num_steps"])
        final_info = rollout["final_info"]

        if rollout["success"]:
            success_count += 1
            success_steps.append(step_count)

        all_steps.append(step_count)

        if bool(final_info.get("ever_task_feasible", False)):
            ever_task_feasible += 1
            first_step = final_info.get("first_task_feasible_step")
            if first_step is not None:
                first_task_feasible_steps.append(int(first_step))

        if rollout["collision"]:
            collision_episodes += 1
        if rollout["out_of_bounds"]:
            out_of_bounds_episodes += 1
        if execution_mode == "hierarchical":
            events = rollout.get("high_level_events", [])
            if events:
                raw_valid_rates.extend(float(ev["raw_valid"]) for ev in events)
                repair_distances.extend(float(ev["repair_distance"]) for ev in events)
                repair_dthetas.extend(float(ev["repair_dtheta"]) for ev in events)
                raw_task_scores.extend(float(ev["raw_task_score"]) for ev in events)
                repaired_task_scores.extend(float(ev["repaired_task_score"]) for ev in events)
                top_model_top1_matches.extend(
                    float(ev["top_model_top1_match"])
                    for ev in events
                    if ev.get("top_model_top1_match") is not None
                )
                top_model_eval_mses.extend(
                    float(ev["top_model_eval_mse"])
                    for ev in events
                    if ev.get("top_model_eval_mse") is not None
                )
                selected_pred_costs.extend(
                    float(ev["selected_pred_cost"])
                    for ev in events
                    if ev.get("selected_pred_cost") is not None
                )
                selected_rollout_labels.extend(
                    float(ev["selected_rollout_label"])
                    for ev in events
                    if ev.get("selected_rollout_label") is not None
                )

        if (
            viz_cfg.save_visualizations
            and rollout["success"]
            and len(visualization_index["success"]) < int(viz_cfg.max_successes)
        ):
            visualization_index["success"].append(
                _save_rollout_visualization(
                    env,
                    rollout,
                    execution_mode=execution_mode,
                    episode_index=i,
                    category="success",
                    category_dir=success_dir,
                    base_output_dir=output_dir,
                    viz_cfg=viz_cfg,
                )
            )

        if (
            viz_cfg.save_visualizations
            and not rollout["success"]
            and len(visualization_index["failure"]) < int(viz_cfg.max_failures)
        ):
            visualization_index["failure"].append(
                _save_rollout_visualization(
                    env,
                    rollout,
                    execution_mode=execution_mode,
                    episode_index=i,
                    category="failure",
                    category_dir=failure_dir,
                    base_output_dir=output_dir,
                    viz_cfg=viz_cfg,
                )
            )

    success_rate = success_count / float(n_trials) if n_trials > 0 else 0.0
    metrics = {
        "success_rate": success_rate,
        "avg_steps_success": float(np.mean(success_steps)) if success_steps else 0.0,
        "avg_steps_all": float(np.mean(all_steps)) if all_steps else 0.0,
        "num_success": float(success_count),
        "num_trials": float(n_trials),
        "ever_task_feasible_rate": ever_task_feasible / float(n_trials) if n_trials > 0 else 0.0,
        "avg_first_task_feasible_step": (
            float(np.mean(first_task_feasible_steps)) if first_task_feasible_steps else 0.0
        ),
        "collision_rate": collision_episodes / float(n_trials) if n_trials > 0 else 0.0,
        "out_of_bounds_rate": out_of_bounds_episodes / float(n_trials) if n_trials > 0 else 0.0,
        "raw_actor_output_valid_rate": float(np.mean(raw_valid_rates)) if raw_valid_rates else 0.0,
        "mean_repair_distance": float(np.mean(repair_distances)) if repair_distances else 0.0,
        "mean_repair_dtheta": float(np.mean(repair_dthetas)) if repair_dthetas else 0.0,
        "mean_taskscore_raw_subgoal": float(np.mean(raw_task_scores)) if raw_task_scores else 0.0,
        "mean_taskscore_repaired_subgoal": float(np.mean(repaired_task_scores)) if repaired_task_scores else 0.0,
        "top_model_top1_match_rate": float(np.mean(top_model_top1_matches)) if top_model_top1_matches else 0.0,
        "top_model_val_mse": float(np.mean(top_model_eval_mses)) if top_model_eval_mses else 0.0,
        "mean_selected_pred_cost": float(np.mean(selected_pred_costs)) if selected_pred_costs else 0.0,
        "mean_selected_rollout_cost_label": float(np.mean(selected_rollout_labels)) if selected_rollout_labels else 0.0,
    }
    return metrics, visualization_index


def _parse_execution_modes(raw: str) -> List[str]:
    modes = [m.strip() for m in str(raw).split(",") if m.strip()]
    if not modes:
        modes = ["greedy", "lookahead"]
    for mode in modes:
        if mode not in {"greedy", "lookahead", "hierarchical"}:
            raise ValueError(f"不支持的 execution mode: {mode}")
    return modes


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="通信巡检 Dubins 环境上的 QRL 成功率离线评估")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="results/minimal_qrl_inspection_dubins/checkpoint_final.pth",
        help="QRL checkpoint 路径",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/minimal_qrl_inspection_dubins",
        help="输出目录（保存 json）",
    )
    parser.add_argument(
        "--env-name",
        type=str,
        default="comm_inspection_dubins_uav_eval",
        help="offline env 注册名",
    )

    parser.add_argument("--bounds", type=float, nargs=4, default=[0.0, 0.0, 10.0, 10.0])
    parser.add_argument("--omega-max", type=float, default=3.0)
    parser.add_argument("--v", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--max-episode-steps", type=int, default=180)
    parser.add_argument("--obstacle-config", type=str, default="medium", choices=["none", "simple", "medium", "hard"])
    parser.add_argument("--obstacles", type=float, nargs="*", default=None)
    parser.add_argument("--observation-mode", type=str, default="task_context", choices=["task_context", "cos_sin", "state"])

    parser.add_argument("--inspection-target", type=float, nargs=2, default=[3.0, 7.5])
    parser.add_argument("--ground-station", type=float, nargs=2, default=[1.5, 2.0])
    parser.add_argument("--randomize-inspection-target", action="store_true")
    parser.add_argument("--randomize-ground-station", action="store_true")
    parser.add_argument("--observation-radius", type=float, default=1.8)
    parser.add_argument("--fov-angle", type=float, default=float(np.pi / 2.0))
    parser.add_argument("--require-target-los", dest="require_target_los", action="store_true", default=True)
    parser.add_argument("--no-require-target-los", dest="require_target_los", action="store_false")
    parser.add_argument("--comm-alpha", type=float, default=2.0)
    parser.add_argument("--comm-bias", type=float, default=5.0)
    parser.add_argument("--comm-occlusion-penalty", type=float, default=6.0)
    parser.add_argument("--comm-threshold", type=float, default=0.5)
    parser.add_argument("--require-ground-station-los", action="store_true")
    parser.add_argument("--goal-sampling-mode", type=str, default="task_feasible", choices=["task_feasible", "valid"])
    parser.add_argument("--goal-position-tolerance", type=float, default=0.15)
    parser.add_argument("--goal-heading-tolerance", type=float, default=0.2)
    parser.add_argument("--collision-cost", type=float, default=10.0)
    parser.add_argument("--out-of-bounds-cost", type=float, default=10.0)
    parser.add_argument("--communication-break-cost", type=float, default=1.0)
    parser.add_argument("--observation-violation-cost-weight", type=float, default=1.0)
    parser.add_argument("--communication-violation-cost-weight", type=float, default=0.5)
    parser.add_argument("--observation-failure-cost", type=float, default=0.25)

    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-critics", type=int, default=2)
    parser.add_argument("--execution-modes", type=str, default="greedy,lookahead")
    parser.add_argument("--subgoal-actor-checkpoint", type=str, default=None)
    parser.add_argument("--subgoal-selector", type=str, default="heuristic", choices=["heuristic", "cost_aware"])
    parser.add_argument("--top-model-checkpoint", type=str, default=None)
    parser.add_argument("--high-level-period", type=int, default=5)
    parser.add_argument("--subgoal-candidates", type=int, default=64)
    parser.add_argument("--subgoal-lambda-final", type=float, default=0.3)
    parser.add_argument("--subgoal-lambda-task", type=float, default=1.0)
    parser.add_argument("--top-model-rollout-steps", type=int, default=None)

    parser.add_argument("--lookahead-horizon", type=int, default=20)
    parser.add_argument("--lookahead-num-sequences", type=int, default=256)
    parser.add_argument("--lookahead-step-cost-weight", type=float, default=0.0)
    parser.add_argument("--lookahead-collision-penalty", type=float, default=0.0)
    parser.add_argument("--lookahead-biased-sequences", type=int, default=48)
    parser.add_argument("--lookahead-bias-kp", type=float, default=2.0)
    parser.add_argument("--lookahead-use-cem", action="store_true")
    parser.add_argument("--lookahead-cem-iters", type=int, default=3)
    parser.add_argument("--lookahead-cem-elite-frac", type=float, default=0.1)
    parser.add_argument("--lookahead-cem-std-init-frac", type=float, default=0.5)
    parser.add_argument("--planner-alpha-subgoal", type=float, default=1.0)
    parser.add_argument("--planner-alpha-final", type=float, default=0.3)
    parser.add_argument("--planner-alpha-task-terminal", type=float, default=0.5)
    parser.add_argument("--planner-use-env-stage-cost", dest="planner_use_env_stage_cost", action="store_true", default=True)
    parser.add_argument("--no-planner-use-env-stage-cost", dest="planner_use_env_stage_cost", action="store_false")
    parser.add_argument("--taskscore-beta-obs", type=float, default=1.0)
    parser.add_argument("--taskscore-beta-comm", type=float, default=1.0)
    parser.add_argument("--taskscore-beta-feas", type=float, default=0.5)
    parser.add_argument("--taskscore-margin-clip", type=float, default=2.0)

    parser.add_argument("--save-visualizations", action="store_true")
    parser.add_argument("--viz-max-successes", type=int, default=10)
    parser.add_argument("--viz-max-failures", type=int, default=10)
    parser.add_argument("--viz-save-gif", action="store_true")
    parser.add_argument("--viz-gif-fps", type=int, default=8)
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(
            f"未找到 checkpoint: {args.checkpoint}. "
            "请确认训练已完成，或通过 --checkpoint 指定正确路径。"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    output_dir = Path(args.output_dir)
    device = auto_device(args.device)

    env = make_comm_inspection_env(args)
    agent, ckpt_step = build_qrl_adapter(args, device, env)

    execution_modes = _parse_execution_modes(args.execution_modes)
    subgoal_actor = None
    subgoal_actor_meta: Dict[str, Any] = {}
    subgoal_ckpt: Optional[str] = None
    top_model = None
    top_model_meta: Dict[str, Any] = {}
    top_model_ckpt: Optional[str] = None
    if "hierarchical" in execution_modes:
        subgoal_ckpt = args.subgoal_actor_checkpoint
        if not subgoal_ckpt:
            candidate = Path(args.checkpoint).with_name("subgoal_actor_checkpoint_final.pth")
            subgoal_ckpt = str(candidate)
        if not os.path.exists(subgoal_ckpt):
            raise FileNotFoundError(
                f"hierarchical 模式需要 subgoal actor checkpoint，但未找到: {subgoal_ckpt}"
            )
        subgoal_actor, subgoal_actor_meta = load_subgoal_actor_checkpoint(subgoal_ckpt, device=device)
        if str(args.subgoal_selector) == "cost_aware":
            top_model_ckpt = args.top_model_checkpoint
            if not top_model_ckpt:
                candidate = Path(args.checkpoint).with_name("cost_aware_subgoal_scorer_checkpoint_final.pth")
                top_model_ckpt = str(candidate)
            if not os.path.exists(top_model_ckpt):
                raise FileNotFoundError(
                    f"subgoal_selector=cost_aware 需要顶层 scorer checkpoint，但未找到: {top_model_ckpt}"
                )
            top_model, top_model_meta = load_cost_aware_subgoal_scorer_checkpoint(top_model_ckpt, device=device)

    lookahead_cfg = DubinsLookaheadConfig(
        horizon=int(args.lookahead_horizon),
        num_sequences=int(args.lookahead_num_sequences),
        step_cost_weight=float(args.lookahead_step_cost_weight),
        collision_penalty=float(args.lookahead_collision_penalty),
        biased_sequences=int(args.lookahead_biased_sequences),
        bias_kp=float(args.lookahead_bias_kp),
        use_cem=bool(args.lookahead_use_cem),
        cem_iters=int(args.lookahead_cem_iters),
        cem_elite_frac=float(args.lookahead_cem_elite_frac),
        cem_std_init_frac=float(args.lookahead_cem_std_init_frac),
        alpha_subgoal=float(args.planner_alpha_subgoal),
        alpha_final=float(args.planner_alpha_final),
        alpha_task_terminal=float(args.planner_alpha_task_terminal),
        use_env_stage_cost=bool(args.planner_use_env_stage_cost),
    )
    viz_cfg = VisualizationConfig(
        save_visualizations=bool(args.save_visualizations),
        max_successes=int(args.viz_max_successes),
        max_failures=int(args.viz_max_failures),
        save_gif=bool(args.viz_save_gif),
        gif_fps=int(args.viz_gif_fps),
    )

    results: Dict[str, Dict[str, float]] = {}
    visualization_index: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for mode in execution_modes:
        cfg = lookahead_cfg if mode == "lookahead" else None
        if mode == "hierarchical":
            cfg = lookahead_cfg
        metrics, vis_index = evaluate_execution_mode(
            agent,
            env,
            mode,
            n_trials=int(args.n_trials),
            seed=int(args.seed),
            lookahead_cfg=cfg,
            output_dir=output_dir,
            viz_cfg=viz_cfg,
            subgoal_actor=subgoal_actor,
            top_model=top_model,
            actor_device=device,
            high_level_period=int(args.high_level_period),
            subgoal_candidates=int(args.subgoal_candidates),
            subgoal_lambda_final=float(args.subgoal_lambda_final),
            subgoal_lambda_task=float(args.subgoal_lambda_task),
            subgoal_selector=str(args.subgoal_selector),
            top_model_rollout_steps=args.top_model_rollout_steps,
        )
        results[mode] = metrics
        visualization_index[mode] = vis_index

    payload = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "ckpt_step": int(ckpt_step) if ckpt_step is not None else None,
        "seed": int(args.seed),
        "n_trials": int(args.n_trials),
        "execution_modes": execution_modes,
        "results": results,
        "visualizations": visualization_index,
        "lookahead_config": asdict(lookahead_cfg),
        "subgoal_actor_checkpoint": os.path.abspath(subgoal_ckpt) if subgoal_ckpt else None,
        "subgoal_actor_metadata": subgoal_actor_meta,
        "subgoal_selector": str(args.subgoal_selector),
        "top_model_checkpoint": os.path.abspath(top_model_ckpt) if top_model_ckpt else None,
        "top_model_metadata": top_model_meta,
        "visualization_config": asdict(viz_cfg),
        "env_config": {
            "bounds": [float(v) for v in args.bounds],
            "omega_max": float(args.omega_max),
            "v": float(args.v),
            "dt": float(args.dt),
            "max_episode_steps": int(args.max_episode_steps),
            "obstacle_config": str(args.obstacle_config),
            "inspection_target": [float(v) for v in args.inspection_target],
            "ground_station": [float(v) for v in args.ground_station],
            "observation_mode": str(args.observation_mode),
            "goal_sampling_mode": str(args.goal_sampling_mode),
            "require_target_los": bool(args.require_target_los),
            "require_ground_station_los": bool(args.require_ground_station_los),
            "comm_threshold": float(args.comm_threshold),
            "taskscore_beta_obs": float(args.taskscore_beta_obs),
            "taskscore_beta_comm": float(args.taskscore_beta_comm),
            "taskscore_beta_feas": float(args.taskscore_beta_feas),
            "taskscore_margin_clip": float(args.taskscore_margin_clip),
        },
    }

    out_json = os.path.join(args.output_dir, "comm_inspection_execution_eval.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[comm_inspection_execution_eval] 已保存评估结果到 {out_json}")
    for mode in execution_modes:
        metrics = results[mode]
        line = (
            f"  {mode}: success_rate={metrics['success_rate']:.3f}, "
            f"avg_steps_success={metrics['avg_steps_success']:.1f}, "
            f"ever_task_feasible_rate={metrics['ever_task_feasible_rate']:.3f}, "
            f"collision_rate={metrics['collision_rate']:.3f}, "
            f"out_of_bounds_rate={metrics['out_of_bounds_rate']:.3f}"
        )
        if mode == "hierarchical":
            line += (
                f", raw_valid={metrics['raw_actor_output_valid_rate']:.3f}, "
                f"repair={metrics['mean_repair_distance']:.3f}"
            )
        print(line)
        if viz_cfg.save_visualizations:
            vis = visualization_index[mode]
            print(
                f"    visualizations: success={len(vis['success'])}, "
                f"failure={len(vis['failure'])}"
            )


if __name__ == "__main__":
    main()
