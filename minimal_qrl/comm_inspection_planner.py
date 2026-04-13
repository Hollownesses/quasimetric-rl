from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional

import numpy as np

from minimal_qrl.envs import CommInspectionDubinsUAV2D
from minimal_qrl.eval.dubins_execution_mode_eval import DubinsLookaheadConfig
from minimal_qrl.gc_agents import GoalConditionedAgentBase


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def hierarchical_low_level_lookahead_cfg(cfg: DubinsLookaheadConfig) -> DubinsLookaheadConfig:
    return replace(
        cfg,
        alpha_final=0.0,
        alpha_task_terminal=0.0,
    )


def evaluate_comm_lookahead_sequences(
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
        terminal_cost = 0.0
        if not success:
            terminal_cost += float(cfg.alpha_final) * float(agent.value(terminal_obs, goal_obs))
            terminal_cost -= float(cfg.alpha_task_terminal) * float(env.compute_task_score(env.state))
            if subgoal_obs is not None and not reached_subgoal:
                terminal_cost += float(cfg.alpha_subgoal) * float(agent.value(terminal_obs, subgoal_obs))

        costs[i] = float(total_cost + terminal_cost)

    env.set_state(base_state)
    return costs, first_actions


def comm_inspection_lookahead_action(
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
            costs, firsts = evaluate_comm_lookahead_sequences(
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

    costs, firsts = evaluate_comm_lookahead_sequences(
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
