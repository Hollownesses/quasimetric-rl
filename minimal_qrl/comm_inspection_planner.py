from __future__ import annotations

from typing import Any, Optional

import numpy as np

from minimal_qrl.envs import CommInspectionDubinsUAV2D
from minimal_qrl.eval.dubins_execution_mode_eval import DubinsLookaheadConfig
from minimal_qrl.gc_agents import GoalConditionedAgentBase


INVALID_ROLLOUT_COST = 1_000_000.0


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _heuristic_mode(cfg: DubinsLookaheadConfig) -> str:
    mode = str(getattr(cfg, "heuristic_mode", "terminal")).strip().lower()
    if mode not in {"terminal", "dense"}:
        raise ValueError(f"未知 lookahead heuristic_mode: {mode}")
    return mode


def _build_candidate_omegas(
    env: CommInspectionDubinsUAV2D,
    cfg: DubinsLookaheadConfig,
    *,
    subgoal_state: Optional[np.ndarray] = None,
) -> np.ndarray:
    horizon = max(1, int(cfg.horizon))
    num_sequences = max(1, int(cfg.num_sequences))
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
        if subgoal_state is not None:
            desired = np.asarray(subgoal_state, dtype=np.float32)
        elif hasattr(env, "sample_task_terminal_state"):
            best = None
            best_cost = float("inf")
            for _ in range(max(1, n_bias)):
                cand = env.sample_task_terminal_state()
                cost = env.compute_goal_reaching_cost_estimate(env.state, cand)
                if cost < best_cost:
                    best = cand
                    best_cost = float(cost)
            desired = np.asarray(best if best is not None else env.state, dtype=np.float32)
        else:
            desired = np.asarray(env.state, dtype=np.float32)
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
    if (n_bias + n_rand) <= 0:
        return np.zeros((1, horizon), dtype=np.float32)
    return np.concatenate([bias, rand], axis=0)


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
    success_mask = np.zeros((n,), dtype=bool)
    invalid_mask = np.zeros((n,), dtype=bool)
    reached_subgoal_mask = np.zeros((n,), dtype=bool)
    terminal_obs_batch = np.zeros((n, int(goal_obs.shape[0])), dtype=np.float32)
    terminal_states = np.zeros((n, 3), dtype=np.float32)
    mode = _heuristic_mode(cfg)
    progress_alpha = float(getattr(cfg, "qrl_progress_alpha", 0.0))
    collect_progress = mode == "dense" and progress_alpha != 0.0
    progress_seq_indices: list[int] = []
    progress_prev_obs: list[np.ndarray] = []
    progress_next_obs: list[np.ndarray] = []

    for i in range(n):
        env.set_state(base_state)
        total_cost = 0.0
        reached_subgoal = False
        success = False
        invalid = False

        for t in range(int(omegas.shape[1])):
            w = float(omegas[i, t])
            action = np.array([w], dtype=np.float32)
            prev_obs = env.state_to_observation(env.state).astype(np.float32) if collect_progress else None
            _obs, _reward, terminated, truncated, info = env.step(action)
            if collect_progress:
                progress_seq_indices.append(i)
                progress_prev_obs.append(prev_obs)
                progress_next_obs.append(env.state_to_observation(env.state).astype(np.float32))

            if bool(cfg.use_env_stage_cost):
                total_cost += _safe_float(info.get("cost_total"))
            else:
                if cfg.step_cost_weight > 0.0:
                    total_cost += float(cfg.step_cost_weight) * abs(w)
                if cfg.collision_penalty > 0.0 and bool(info.get("collision", False)):
                    total_cost += float(cfg.collision_penalty)

            if bool(info.get("collision", False)) or bool(info.get("out_of_bounds", False)):
                invalid = True

            if subgoal_state is not None and env.is_subgoal_reached(
                env.state,
                subgoal_state,
                pos_tolerance=float(cfg.subgoal_reached_pos_tolerance),
                theta_tolerance=float(cfg.subgoal_reached_theta_tolerance),
            ):
                reached_subgoal = True

            if terminated:
                success = bool(info.get("success", False))
                break
            if truncated:
                break

        terminal_obs_batch[i] = env.state_to_observation(env.state).astype(np.float32)
        terminal_states[i] = np.asarray(env.state, dtype=np.float32)
        success_mask[i] = bool(success)
        invalid_mask[i] = bool(invalid)
        reached_subgoal_mask[i] = bool(reached_subgoal)
        costs[i] = float(INVALID_ROLLOUT_COST + total_cost) if invalid else float(total_cost)

    not_success = np.logical_and(~success_mask, ~invalid_mask)
    if np.any(not_success):
        if collect_progress and progress_seq_indices:
            seq_idx = np.asarray(progress_seq_indices, dtype=np.int64)
            active = not_success[seq_idx]
            if np.any(active):
                active_positions = np.nonzero(active)[0]
                prev_batch = np.stack([progress_prev_obs[j] for j in active_positions], axis=0).astype(np.float32)
                next_batch = np.stack([progress_next_obs[j] for j in active_positions], axis=0).astype(np.float32)
                goal_batch = np.repeat(goal_obs[None, :].astype(np.float32), prev_batch.shape[0], axis=0)
                progress = agent.batch_value(next_batch, goal_batch) - agent.batch_value(prev_batch, goal_batch)
                progress_cost = np.bincount(seq_idx[active], weights=progress, minlength=n).astype(np.float32)
                costs += progress_alpha * progress_cost

        active_obs = terminal_obs_batch[not_success]
        if float(cfg.alpha_final) != 0.0:
            goal_batch = np.repeat(goal_obs[None, :].astype(np.float32), active_obs.shape[0], axis=0)
            costs[not_success] += float(cfg.alpha_final) * agent.batch_value(active_obs, goal_batch)
        if float(cfg.alpha_task_terminal) != 0.0:
            active_scores = np.array(
                [float(env.compute_task_score(state)) for state in terminal_states[not_success]],
                dtype=np.float32,
            )
            costs[not_success] -= float(cfg.alpha_task_terminal) * active_scores
        if subgoal_obs is not None and float(cfg.alpha_subgoal) != 0.0:
            need_subgoal = np.logical_and(not_success, ~reached_subgoal_mask)
            if np.any(need_subgoal):
                subgoal_batch = np.repeat(subgoal_obs[None, :].astype(np.float32), int(np.sum(need_subgoal)), axis=0)
                costs[need_subgoal] += float(cfg.alpha_subgoal) * agent.batch_value(
                    terminal_obs_batch[need_subgoal],
                    subgoal_batch,
                )

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
    sequence = comm_inspection_lookahead_plan(
        agent,
        env,
        goal_obs,
        cfg,
        subgoal_state=subgoal_state,
    )
    return np.array([float(sequence[0])], dtype=np.float32)


def comm_inspection_lookahead_plan(
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
    omegas0 = _build_candidate_omegas(env, cfg, subgoal_state=subgoal_state)
    low = float(env.action_space.low[0])
    high = float(env.action_space.high[0])
    n_bias = int(min(max(0, cfg.biased_sequences), int(omegas0.shape[0])))
    bias = omegas0[:n_bias].copy()

    if cfg.use_cem:
        om_range = float(high - low)
        std = np.full((horizon,), float(cfg.cem_std_init_frac) * 0.5 * om_range, dtype=np.float32)
        mean = np.mean(omegas0, axis=0).astype(np.float32) if omegas0.shape[0] > 0 else np.zeros((horizon,), dtype=np.float32)
        n_elite = max(1, int(float(cfg.cem_elite_frac) * float(num_sequences)))
        best_sequence = np.zeros((horizon,), dtype=np.float32)
        best_cost = float("inf")

        for _ in range(max(1, int(cfg.cem_iters))):
            samples = np.random.normal(loc=mean[None, :], scale=std[None, :], size=(num_sequences, horizon)).astype(np.float32)
            samples = np.clip(samples, low, high)
            if n_bias > 0:
                samples[: min(n_bias, samples.shape[0])] = bias[: min(n_bias, samples.shape[0])]
            costs, _firsts = evaluate_comm_lookahead_sequences(
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
                best_sequence = samples[idx].astype(np.float32).copy()
            elite = samples[np.argsort(costs)[:n_elite]]
            mean = np.mean(elite, axis=0).astype(np.float32)
            std = (np.std(elite, axis=0) + 1e-4).astype(np.float32)

        env.set_state(base_state)
        return best_sequence

    costs, _firsts = evaluate_comm_lookahead_sequences(
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
    return omegas0[best_idx].astype(np.float32).copy()
