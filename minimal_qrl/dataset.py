"""
创建数据集用于 QRL 训练，支持多种环境
"""
import numpy as np
import random
import torch
import warnings
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Optional, List, Sequence, Tuple
import gym

from quasimetric_rl.data import EpisodeData
from minimal_qrl.envs.base import BaseNavigationEnv


TASK_AWARE_TEACHER_MAX_ATTEMPTS = 8
TASK_AWARE_TEACHER_KP = 4.0
TASK_AWARE_TEACHER_ACTION_NOISE_STD = 0.0

EXPLORE_OUTCOME_UNKNOWN = 0
EXPLORE_OUTCOME_SUCCESS = 1
EXPLORE_OUTCOME_COLLISION = 2
EXPLORE_OUTCOME_OUT_OF_BOUNDS = 3
EXPLORE_OUTCOME_TIMEOUT = 4
EXPLORE_OUTCOME_BUDGET_CUTOFF = 5

EXPLORE_OUTCOME_NAMES = {
    EXPLORE_OUTCOME_UNKNOWN: "unknown",
    EXPLORE_OUTCOME_SUCCESS: "success",
    EXPLORE_OUTCOME_COLLISION: "collision",
    EXPLORE_OUTCOME_OUT_OF_BOUNDS: "out_of_bounds",
    EXPLORE_OUTCOME_TIMEOUT: "timeout",
    EXPLORE_OUTCOME_BUDGET_CUTOFF: "budget_cutoff",
}


@dataclass(frozen=True)
class QRLExploreConfig:
    """Goal-blind, coverage-oriented data collection for communication QRL."""

    attempted_env_steps: int = 200_000
    start_position_resolution: float = 1.0
    start_heading_bins: int = 12
    action_hold_min_steps: int = 3
    action_hold_max_steps: int = 10
    straight_action_probability: float = 0.5
    exclusion_radius: float = 0.25
    excluded_start_states: Tuple[Tuple[float, float, float], ...] = ()
    start_states: Tuple[Tuple[float, float, float], ...] = ()
    diagnostic_regions: Optional[Mapping[str, Sequence[float]]] = None
    diagnostic_routes: Optional[Mapping[str, Sequence[str]]] = None
    start_strata: Tuple[
        Tuple[str, float, Tuple[float, float, float, float]], ...
    ] = ()
    start_boundary_margin: float = 0.5
    local_safety_lookahead_steps: int = 10


def _actions_to_array(actions: List) -> np.ndarray:
    if len(actions) > 0:
        first_action = actions[0]
        if isinstance(first_action, (int, np.integer)):
            return np.array(actions, dtype=np.int64)
        return np.array(actions, dtype=np.float32)
    return np.array([], dtype=np.int64)


def _normalize_angle(theta: float) -> float:
    return float((float(theta) + np.pi) % (2.0 * np.pi) - np.pi)


def build_qrl_exploration_start_bank(
    env: gym.Env,
    config: QRLExploreConfig,
    *,
    seed: Optional[int],
) -> np.ndarray:
    """Build a deterministic free-space x-y-heading bank without using a task target.

    The ordering covers every retained spatial cell once before revisiting a
    position with another heading. This matters when the interaction budget is
    too small to consume every position-heading combination.
    """

    resolution = float(config.start_position_resolution)
    heading_bins = int(config.start_heading_bins)
    if not np.isfinite(resolution) or resolution <= 0.0:
        raise ValueError("QRL-explore start_position_resolution must be positive")
    if heading_bins <= 0:
        raise ValueError("QRL-explore start_heading_bins must be positive")
    bounds = (
        float(getattr(env, "x_min")),
        float(getattr(env, "y_min")),
        float(getattr(env, "x_max")),
        float(getattr(env, "y_max")),
    )
    x_values = np.arange(bounds[0] + 0.5 * resolution, bounds[2], resolution)
    y_values = np.arange(bounds[1] + 0.5 * resolution, bounds[3], resolution)
    excluded = np.asarray(config.excluded_start_states, dtype=np.float32)
    if excluded.size == 0:
        excluded = np.zeros((0, 3), dtype=np.float32)
    else:
        excluded = excluded.reshape((-1, 3))
    exclusion_radius_sq = max(0.0, float(config.exclusion_radius)) ** 2
    boundary_margin = max(0.0, float(config.start_boundary_margin))

    positions: list[tuple[float, float]] = []
    for y in y_values:
        for x in x_values:
            if not (
                bounds[0] + boundary_margin <= x <= bounds[2] - boundary_margin
                and bounds[1] + boundary_margin <= y <= bounds[3] - boundary_margin
            ):
                continue
            probe = np.asarray([x, y, 0.0], dtype=np.float32)
            if not bool(env.is_valid_state(probe)):
                continue
            if len(excluded) > 0:
                delta = excluded[:, :2] - probe[None, :2]
                if bool(np.any(np.sum(delta * delta, axis=1) <= exclusion_radius_sq)):
                    continue
            positions.append((float(x), float(y)))
    if not positions:
        raise ValueError("QRL-explore start bank contains no valid free-space cells")

    rng = np.random.default_rng(None if seed is None else int(seed) + 73_856_093)
    states: list[tuple[float, float, float]] = []
    position_indices = np.arange(len(positions), dtype=np.int64)
    base_heading_offsets = np.arange(len(positions), dtype=np.int64) % heading_bins
    rng.shuffle(base_heading_offsets)
    for heading_round in range(heading_bins):
        order = position_indices.copy()
        rng.shuffle(order)
        for position_index in order:
            x, y = positions[int(position_index)]
            heading_index = (
                int(base_heading_offsets[int(position_index)]) + heading_round
            ) % heading_bins
            theta = -np.pi + 2.0 * np.pi * float(heading_index) / float(heading_bins)
            states.append((x, y, _normalize_angle(theta)))
    return np.asarray(states, dtype=np.float32)


def _state_heading_key(
    state: np.ndarray,
    env: gym.Env,
    *,
    position_resolution: float,
    heading_bins: int,
) -> tuple[int, int, int]:
    state = np.asarray(state, dtype=np.float32).reshape(3)
    x_bin = int(np.floor((float(state[0]) - float(env.x_min)) / position_resolution))
    y_bin = int(np.floor((float(state[1]) - float(env.y_min)) / position_resolution))
    angle = (_normalize_angle(float(state[2])) + np.pi) % (2.0 * np.pi)
    theta_bin = int(np.floor(angle / (2.0 * np.pi) * heading_bins)) % heading_bins
    return x_bin, y_bin, theta_bin


def _trajectory_has_loop(keys: Sequence[tuple[int, int, int]], min_separation: int = 10) -> bool:
    first_seen: dict[tuple[int, int, int], int] = {}
    for index, key in enumerate(keys):
        previous = first_seen.get(key)
        if previous is not None and index - previous >= int(min_separation):
            return True
        first_seen.setdefault(key, index)
    return False


def _validated_diagnostic_regions(
    regions: Optional[Mapping[str, Sequence[float]]],
) -> dict[str, tuple[float, float, float, float]]:
    result: dict[str, tuple[float, float, float, float]] = {}
    for name, raw_bounds in (regions or {}).items():
        values = tuple(float(value) for value in raw_bounds)
        if len(values) != 4 or values[2] <= values[0] or values[3] <= values[1]:
            raise ValueError(f"invalid QRL-explore diagnostic region {name!r}: {raw_bounds}")
        result[str(name)] = values
    return result


def _validated_diagnostic_routes(
    routes: Optional[Mapping[str, Sequence[str]]],
    regions: Mapping[str, Sequence[float]],
) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for raw_name, raw_regions in (routes or {}).items():
        name = str(raw_name)
        route = tuple(str(region) for region in raw_regions)
        if not name or name in result or len(route) < 2:
            raise ValueError(f"invalid QRL-explore diagnostic route {name!r}")
        missing = [region for region in route if region not in regions]
        if missing:
            raise ValueError(
                f"diagnostic route {name!r} references unknown regions: {missing}"
            )
        result[name] = route
    return result


def _trajectory_matches_region_route(
    region_indices: Mapping[str, np.ndarray],
    route: Sequence[str],
) -> bool:
    """Whether one trajectory visits every named region in the given order."""

    cursor = -1
    for region_name in route:
        indices = np.asarray(region_indices.get(region_name, ()), dtype=np.int64)
        later = indices[indices > cursor]
        if len(later) == 0:
            return False
        cursor = int(later[0])
    return True


def _validated_start_strata(
    strata: Sequence[Tuple[str, float, Sequence[float]]],
) -> tuple[tuple[str, float, tuple[float, float, float, float]], ...]:
    result = []
    names = set()
    for raw_name, raw_weight, raw_bounds in strata:
        name = str(raw_name)
        weight = float(raw_weight)
        bounds = tuple(float(value) for value in raw_bounds)
        if not name or name in names:
            raise ValueError(f"invalid or duplicate QRL-explore start stratum {name!r}")
        if not np.isfinite(weight) or weight <= 0.0:
            raise ValueError(f"QRL-explore start stratum {name!r} must have positive weight")
        if len(bounds) != 4 or bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
            raise ValueError(f"invalid QRL-explore start stratum bounds for {name!r}")
        names.add(name)
        result.append((name, weight, bounds))
    return tuple(result)


def _weighted_stratum_schedule(
    strata: Sequence[tuple[str, float, tuple[float, float, float, float]]],
    *,
    seed: Optional[int],
    slots: int = 100,
) -> tuple[str, ...]:
    if not strata:
        return ()
    weights = np.asarray([item[1] for item in strata], dtype=np.float64)
    weights /= float(weights.sum())
    raw_counts = weights * max(int(slots), len(strata))
    counts = np.floor(raw_counts).astype(np.int64)
    counts[counts == 0] = 1
    target_slots = max(int(slots), len(strata))
    while int(counts.sum()) < target_slots:
        index = int(np.argmax(raw_counts - counts))
        counts[index] += 1
    while int(counts.sum()) > target_slots:
        candidates = np.flatnonzero(counts > 1)
        index = int(candidates[np.argmin(raw_counts[candidates] - counts[candidates])])
        counts[index] -= 1
    schedule = [
        name
        for (name, _weight, _bounds), count in zip(strata, counts)
        for _ in range(int(count))
    ]
    rng = np.random.default_rng(None if seed is None else int(seed) + 91_771)
    rng.shuffle(schedule)
    return tuple(schedule)


def _locally_safe_steps(
    env: gym.Env,
    state: np.ndarray,
    omega: float,
    max_steps: int,
) -> int:
    """Return collision-free rollout length using only local dynamics/geometry."""

    x, y, theta = (float(value) for value in np.asarray(state).reshape(3))
    safe_steps = 0
    for _ in range(max(0, int(max_steps))):
        theta_new = _normalize_angle(theta + float(omega) * float(env.dt))
        x_new = x + float(env.v) * np.cos(theta_new) * float(env.dt)
        y_new = y + float(env.v) * np.sin(theta_new) * float(env.dt)
        if not (
            float(env.x_min) <= x_new <= float(env.x_max)
            and float(env.y_min) <= y_new <= float(env.y_max)
        ):
            break
        if hasattr(env, "_check_collision") and bool(
            env._check_collision(x, y, x_new, y_new)
        ):
            break
        probe = np.asarray([x_new, y_new, theta_new], dtype=np.float32)
        if not bool(env.is_valid_state(probe)):
            break
        safe_steps += 1
        x, y, theta = x_new, y_new, theta_new
    return int(safe_steps)


def _goal_set_transition_infos(
    *,
    n: int,
    task_goal_obs: np.ndarray,
    context_id: int,
    env: Optional[gym.Env] = None,
    global_push_seed: Optional[int] = None,
    include_global_push_pairs: bool = False,
    abstract_goal_edge: bool = False,
    source_terminal_goal_state: bool = False,
    teacher_guided: bool = False,
    exploration: bool = False,
    exploration_start_index: int = -1,
    exploration_episode_id: int = -1,
    exploration_outcome: int = EXPLORE_OUTCOME_UNKNOWN,
    exploration_loop_detected: bool = False,
    task_success_episode: bool = False,
    goal_return_costs: Optional[Sequence[float]] = None,
) -> dict:
    infos = {
        "abstract_goal_edge": np.full((n,), bool(abstract_goal_edge), dtype=np.bool_),
        "source_terminal_goal_state": np.full((n,), bool(source_terminal_goal_state), dtype=np.bool_),
        "task_goal_observations": np.repeat(task_goal_obs[None, :], n, axis=0).astype(np.float32),
        "context_id": np.full((n,), int(context_id), dtype=np.int64),
        "teacher_guided": np.full((n,), bool(teacher_guided), dtype=np.bool_),
        "exploration": np.full((n,), bool(exploration), dtype=np.bool_),
        "exploration_start_index": np.full((n,), int(exploration_start_index), dtype=np.int64),
        "exploration_episode_id": np.full((n,), int(exploration_episode_id), dtype=np.int64),
        "exploration_outcome": np.full((n,), int(exploration_outcome), dtype=np.int64),
        "exploration_loop_detected": np.full(
            (n,), bool(exploration_loop_detected), dtype=np.bool_
        ),
        "task_success_episode": np.full(
            (n,), bool(task_success_episode), dtype=np.bool_
        ),
    }
    if goal_return_costs is None:
        goal_return_array = np.zeros((n,), dtype=np.float32)
    else:
        goal_return_array = np.asarray(goal_return_costs, dtype=np.float32).reshape(-1)
        if len(goal_return_array) != n:
            raise ValueError("goal_return_costs must match transition count")
        if np.any(~np.isfinite(goal_return_array)) or np.any(goal_return_array < 0.0):
            raise ValueError("goal_return_costs must be finite and nonnegative")
    infos["goal_return_cost"] = goal_return_array
    infos["goal_return_mask"] = np.full(
        (n,), bool(task_success_episode), dtype=np.bool_
    )
    device_index = int(getattr(env, "active_device_index", -1)) if env is not None else -1
    infos["device_index"] = np.full((n,), device_index, dtype=np.int64)

    obs_dim = int(task_goal_obs.shape[0])
    source_pairs = np.zeros((n, obs_dim), dtype=np.float32)
    goal_pairs = np.zeros((n, obs_dim), dtype=np.float32)
    pair_mask = np.zeros((n,), dtype=np.bool_)
    if include_global_push_pairs and env is not None and n > 0:
        rng = np.random.default_rng(global_push_seed)
        for i in range(n):
            source_state = env.sample_valid_state(seed=int(rng.integers(0, 1_000_000_000)))
            goal_state = env.sample_valid_state(seed=int(rng.integers(0, 1_000_000_000)))
            source_pairs[i] = env.state_to_observation(source_state).astype(np.float32)
            goal_pairs[i] = env.state_to_observation(goal_state).astype(np.float32)
            pair_mask[i] = True
    infos["global_push_source_observations"] = source_pairs
    infos["global_push_goal_observations"] = goal_pairs
    infos["global_push_pair_mask"] = pair_mask
    return infos


def _remaining_nonnegative_costs(rewards: Sequence[float]) -> np.ndarray:
    """Executed suffix costs; valid as task-goal upper bounds on successful paths."""

    costs = np.maximum(0.0, -np.asarray(rewards, dtype=np.float32).reshape(-1))
    return np.cumsum(costs[::-1], dtype=np.float64)[::-1].astype(np.float32)


def _make_goal_set_abstract_edge(
    env: gym.Env,
    *,
    terminal_obs: np.ndarray,
    task_goal_obs: np.ndarray,
    context_id: int,
    action_template,
    action_dtype,
) -> EpisodeData:
    zero_action = np.zeros_like(np.asarray(action_template, dtype=np.float32))
    if zero_action.shape == ():
        zero_action = np.asarray(0, dtype=action_dtype)
    return EpisodeData.from_simple_trajectory(
        observations=np.asarray(terminal_obs, dtype=np.float32)[None, :],
        actions=np.asarray([zero_action], dtype=action_dtype),
        next_observations=task_goal_obs[None, :].astype(np.float32),
        rewards=np.array([0.0], dtype=np.float32),
        terminals=np.array([True], dtype=np.bool_),
        timeouts=np.array([False], dtype=np.bool_),
        transition_infos=_goal_set_transition_infos(
            n=1,
            task_goal_obs=task_goal_obs,
            context_id=context_id,
            env=env,
            abstract_goal_edge=True,
            source_terminal_goal_state=True,
        ),
    )


def _dubins_teacher_action(
    env: gym.Env,
    terminal_state: np.ndarray,
    *,
    kp: float,
    heading_switch_distance: float,
    rng: np.random.Generator,
    noise_std: float,
) -> np.ndarray:
    state = np.asarray(env.state, dtype=np.float32).reshape(3)
    terminal_state = np.asarray(terminal_state, dtype=np.float32).reshape(3)
    delta = terminal_state[:2] - state[:2]
    dist = float(np.linalg.norm(delta))
    if dist <= float(heading_switch_distance):
        desired_heading = float(terminal_state[2])
    else:
        desired_heading = float(np.arctan2(delta[1], delta[0]))

    heading_error = _normalize_angle(desired_heading - float(state[2]))
    omega = float(kp) * heading_error
    if noise_std > 0.0:
        omega += float(rng.normal(0.0, float(noise_std)))
    omega = float(np.clip(omega, -float(env.omega_max), float(env.omega_max)))
    return np.array([omega], dtype=np.float32)


def collect_random_episode(
    env: gym.Env,
    max_steps: int = 200,
    sample_valid_start: bool = True,
    seed: Optional[int] = None
) -> EpisodeData:
    """
    收集一个随机 episode
    
    Args:
        env: 环境实例（应实现 BaseNavigationEnv 接口）
        max_steps: 每个 episode 的最大步数
        sample_valid_start: 是否使用环境的 sample_valid_state 方法采样起始状态
        seed: 随机种子
    
    Returns:
        EpisodeData
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    # 如果环境实现了 BaseNavigationEnv 接口且需要采样合法起始状态
    if sample_valid_start and isinstance(env, BaseNavigationEnv):
        # 采样合法起始状态
        start_state = env.sample_valid_state(seed=seed)
        # 通过 options 传递起始状态（如果环境支持）
        # ContinuousObstacle2D 和 SimpleGrid2D 都支持通过 options 传递 start
        options = {}
        if hasattr(env, 'start') or hasattr(env, 'start_pos'):
            options['start'] = start_state.tolist() if isinstance(start_state, np.ndarray) else start_state
        obs, _ = env.reset(seed=seed, options=options if options else None)
    else:
        obs, _ = env.reset(seed=seed)
    
    observations = [obs.copy()]
    actions = []
    next_observations = []
    rewards = []
    terminals = []
    timeouts = []
    
    for _ in range(max_steps):
        # 随机动作
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        
        # 确保下一个观察是合法的（对于障碍物环境）
        if isinstance(env, BaseNavigationEnv) and not env.is_valid_state(next_obs):
            # 如果状态不合法，尝试投影到合法状态
            # 这里我们简单地跳过这个转移，或者使用当前状态
            # 为了简单，我们使用当前状态
            next_obs = obs.copy()
        
        observations.append(next_obs.copy())
        
        # 处理动作类型（离散或连续）
        # 将动作转换为 numpy 数组以便统一处理
        if isinstance(action, (int, np.integer)):
            actions.append(int(action))
        elif isinstance(action, np.ndarray):
            actions.append(action.astype(np.float32))
        else:
            # 尝试转换为 numpy 数组
            try:
                actions.append(np.array(action, dtype=np.float32))
            except:
                actions.append(action)
        
        next_observations.append(next_obs.copy())
        rewards.append(float(reward))
        terminals.append(bool(terminated))
        timeouts.append(bool(truncated))
        
        obs = next_obs
        
        if terminated or truncated:
            break
    
    actions_array = _actions_to_array(actions)
    
    return EpisodeData.from_simple_trajectory(
        observations=np.array(observations[:-1], dtype=np.float32),  # 去掉最后一个
        actions=actions_array,
        next_observations=np.array(next_observations, dtype=np.float32),
        rewards=np.array(rewards, dtype=np.float32),
        terminals=np.array(terminals, dtype=np.bool_),
        timeouts=np.array(timeouts, dtype=np.bool_),
    )


def collect_goal_set_comm_episode_pair(
    env: gym.Env,
    max_steps: int = 200,
    seed: Optional[int] = None,
    context_id: int = 0,
    device_id: Optional[str] = None,
) -> Tuple[EpisodeData, Optional[EpisodeData]]:
    """
    Collect one communication-inspection goal-set episode plus one abstract
    zero-cost transition terminal_state -> G_task(xi).

    The abstract edge is part of the augmented MDP for every task context. It
    does not require the random rollout to discover the terminal set.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    reset_options = {"device_id": str(device_id)} if device_id is not None else None
    obs, _ = env.reset(seed=seed, options=reset_options)
    task_goal_obs = env.abstract_goal_observation().astype(np.float32)

    observations = [obs.copy()]
    actions = []
    next_observations = []
    rewards = []
    terminals = []
    timeouts = []
    final_info = {}

    for _ in range(max_steps):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)

        if isinstance(action, (int, np.integer)):
            actions.append(int(action))
        else:
            actions.append(np.asarray(action, dtype=np.float32))
        observations.append(next_obs.copy())
        next_observations.append(next_obs.copy())
        rewards.append(float(reward))
        terminals.append(bool(terminated))
        timeouts.append(bool(truncated))
        final_info = dict(info)
        obs = next_obs
        if terminated or truncated:
            break

    actions_array = _actions_to_array(actions)
    n = len(actions)
    success = bool(final_info.get("success", False)) and n > 0
    rewards_array = np.asarray(rewards, dtype=np.float32)
    episode = EpisodeData.from_simple_trajectory(
        observations=np.array(observations[:-1], dtype=np.float32),
        actions=actions_array,
        next_observations=np.array(next_observations, dtype=np.float32),
        rewards=rewards_array,
        terminals=np.array(terminals, dtype=np.bool_),
        timeouts=np.array(timeouts, dtype=np.bool_),
        transition_infos=_goal_set_transition_infos(
            n=n,
            task_goal_obs=task_goal_obs,
            context_id=context_id,
            env=env,
            global_push_seed=None if seed is None else int(seed + 32452843),
            include_global_push_pairs=True,
            task_success_episode=success,
            goal_return_costs=(
                _remaining_nonnegative_costs(rewards_array) if success else None
            ),
        ),
    )

    abstract_episode = None
    try:
        terminal_obs = np.asarray(obs, dtype=np.float32)
        if not success:
            terminal_state = env.sample_task_terminal_state(
                seed=None if seed is None else int(seed + 104729)
            )
            terminal_obs = env.state_to_observation(terminal_state).astype(np.float32)

        abstract_episode = _make_goal_set_abstract_edge(
            env,
            terminal_obs=terminal_obs,
            task_goal_obs=task_goal_obs,
            context_id=context_id,
            action_template=actions_array[-1] if n > 0 else env.action_space.sample(),
            action_dtype=actions_array.dtype if n > 0 else np.float32,
        )
    except RuntimeError:
        abstract_episode = None
    return episode, abstract_episode


def collect_task_aware_comm_teacher_episode_pair(
    env: gym.Env,
    max_steps: int = 200,
    seed: Optional[int] = None,
    context_id: int = 0,
    heading_switch_distance: Optional[float] = None,
    device_id: Optional[str] = None,
    task_context=None,
    task_goal_obs: Optional[np.ndarray] = None,
    collection_stats: Optional[dict[str, Any]] = None,
) -> Tuple[Optional[EpisodeData], Optional[EpisodeData]]:
    """
    Collect a successful task-aware Dubins teacher trajectory:

        s0 -> s1 -> ... -> g in G_task(xi) -> abstract G_task(xi)

    A provided device_id/task_context is preserved across all attempts. The
    collector samples one terminal state g in G_task(xi), resets to a random
    non-terminal start under the same context, and uses Dubins guidance to g.
    """
    rng = np.random.default_rng(seed)
    for attempt in range(max(1, int(TASK_AWARE_TEACHER_MAX_ATTEMPTS))):
        attempt_seed = None if seed is None else int(seed + 1009 * attempt)
        context_options = {}
        if task_context is not None:
            context_options["task_context"] = task_context
        elif device_id is not None:
            context_options["device_id"] = str(device_id)
        _obs, _ = env.reset(seed=attempt_seed, options=context_options or None)
        current_task_context = env.active_task
        current_task_goal_obs = (
            np.asarray(task_goal_obs, dtype=np.float32).copy()
            if task_goal_obs is not None
            else env.abstract_goal_observation().astype(np.float32)
        )

        try:
            terminal_state = env.sample_task_terminal_state(
                seed=None if seed is None else int(seed + 200003 + 1009 * attempt)
            )
        except RuntimeError:
            continue

        reset_seed = None if seed is None else int(seed + 400009 + 1009 * attempt)
        obs, _ = env.reset(
            seed=reset_seed,
            options={"task_context": current_task_context},
        )

        observations = [obs.copy()]
        actions = []
        next_observations = []
        rewards = []
        terminals = []
        timeouts = []
        final_info = {}
        success = False
        switch_dist = (
            float(heading_switch_distance)
            if heading_switch_distance is not None
            else max(0.35, 0.5 * float(getattr(env, "observation_radius", 1.0)))
        )

        for _ in range(max_steps):
            action = _dubins_teacher_action(
                env,
                terminal_state,
                kp=float(TASK_AWARE_TEACHER_KP),
                heading_switch_distance=switch_dist,
                rng=rng,
                noise_std=float(TASK_AWARE_TEACHER_ACTION_NOISE_STD),
            )
            next_obs, reward, terminated, truncated, info = env.step(action)

            actions.append(action.astype(np.float32))
            observations.append(next_obs.copy())
            next_observations.append(next_obs.copy())
            rewards.append(float(reward))
            terminals.append(bool(terminated))
            timeouts.append(bool(truncated))
            final_info = dict(info)
            success = bool(info.get("success", False))
            if terminated or truncated:
                break

        if collection_stats is not None:
            collection_stats["attempted_teacher_steps"] = int(
                collection_stats.get("attempted_teacher_steps", 0)
            ) + len(actions)

        if not success or len(actions) == 0:
            if collection_stats is not None:
                collection_stats["failed_teacher_attempt_steps"] = int(
                    collection_stats.get("failed_teacher_attempt_steps", 0)
                ) + len(actions)
            continue

        actions_array = _actions_to_array(actions)
        n = len(actions)
        rewards_array = np.asarray(rewards, dtype=np.float32)
        episode = EpisodeData.from_simple_trajectory(
                observations=np.array(observations[:-1], dtype=np.float32),
                actions=actions_array,
                next_observations=np.array(next_observations, dtype=np.float32),
                rewards=rewards_array,
                terminals=np.array(terminals, dtype=np.bool_),
                timeouts=np.array(timeouts, dtype=np.bool_),
                transition_infos=_goal_set_transition_infos(
                    n=n,
                    task_goal_obs=current_task_goal_obs,
                    context_id=context_id,
                    env=env,
                    global_push_seed=None if seed is None else int(seed + 49979687 + attempt),
                    include_global_push_pairs=True,
                    teacher_guided=True,
                    task_success_episode=True,
                    goal_return_costs=_remaining_nonnegative_costs(rewards_array),
                ),
        )
        terminal_obs = np.asarray(observations[-1], dtype=np.float32)
        abstract_episode = _make_goal_set_abstract_edge(
            env,
            terminal_obs=terminal_obs,
            task_goal_obs=current_task_goal_obs,
            context_id=context_id,
            action_template=actions_array[-1],
            action_dtype=actions_array.dtype,
        )
        return episode, abstract_episode

    warnings.warn(
        "Task-aware Dubins teacher failed to collect a successful trajectory "
        f"after {max(1, int(TASK_AWARE_TEACHER_MAX_ATTEMPTS))} attempt(s) for context_id={context_id}.",
        RuntimeWarning,
        stacklevel=2,
    )
    return None, None


def _slice_episode(episode: EpisodeData, length: int) -> EpisodeData:
    """Return the first ``length`` transitions as a well-formed episode."""

    length = int(length)
    if length <= 0 or length > int(episode.num_transitions):
        raise ValueError("episode slice length must be in [1, num_transitions]")
    terminals = episode.terminals[:length].clone()
    timeouts = episode.timeouts[:length].clone()
    if length < int(episode.num_transitions):
        terminals[-1] = False
        timeouts[-1] = True
    return EpisodeData(
        episode_lengths=torch.tensor([length], dtype=torch.int64),
        all_observations=episode.all_observations[: length + 1].clone(),
        actions=episode.actions[:length].clone(),
        rewards=episode.rewards[:length].clone(),
        terminals=terminals,
        timeouts=timeouts,
        observation_infos={key: value[: length + 1].clone() for key, value in episode.observation_infos.items()},
        transition_infos={
            key: value[:length].clone() for key, value in episode.transition_infos.items()
        },
    )


def create_budgeted_comm_dataset(
    env: gym.Env,
    *,
    target_env_transitions: int,
    max_steps_per_episode: int,
    seed: Optional[int],
    task_aware_teacher_ratio: float,
    collection_stats: Optional[dict[str, Any]] = None,
) -> Iterator[EpisodeData]:
    """Collect an exact, task-balanced real-transition budget.

    Random and successful teacher transitions count toward the real budget.
    One-step abstract-goal edges are emitted as synthetic data and accounted
    separately.  Devices are visited in seeded, shuffled round-robin order so
    that each task receives comparable collection opportunities.
    """

    target = int(target_env_transitions)
    if target <= 0:
        raise ValueError("target_env_transitions must be positive")
    device_ids = list(getattr(env, "device_ids", ()))
    if not device_ids:
        raise ValueError("budgeted communication dataset requires device_ids")

    stats = collection_stats if collection_stats is not None else {}
    stats.clear()
    stats.update(
        {
            "target_real_transitions": target,
            "stored_real_transitions": 0,
            "attempted_env_steps": 0,
            "attempted_teacher_steps": 0,
            "failed_teacher_attempt_steps": 0,
            "synthetic_abstract_edges": 0,
            "random_episodes": 0,
            "successful_teacher_episodes": 0,
            "per_device_real_transitions": {device_id: 0 for device_id in device_ids},
            "per_device_collection_contexts": {device_id: 0 for device_id in device_ids},
        }
    )
    per_device_targets = {
        device_id: target // len(device_ids) + int(index < target % len(device_ids))
        for index, device_id in enumerate(device_ids)
    }
    stats["per_device_target_transitions"] = dict(per_device_targets)
    teacher_ratio = max(0.0, float(task_aware_teacher_ratio))
    teacher_floor = int(np.floor(teacher_ratio))
    teacher_fraction = teacher_ratio - teacher_floor
    rng = np.random.default_rng(None if seed is None else int(seed) + 91815541)
    context_id = 0

    while int(stats["stored_real_transitions"]) < target:
        if context_id % len(device_ids) == 0:
            round_ids = list(device_ids)
            rng.shuffle(round_ids)
        device_id = round_ids[context_id % len(device_ids)]
        if (
            int(stats["per_device_real_transitions"][device_id])
            >= int(per_device_targets[device_id])
        ):
            context_id += 1
            continue
        episode_seed = None if seed is None else int(seed + context_id * 104729)
        episode, abstract_episode = collect_goal_set_comm_episode_pair(
            env,
            max_steps=max_steps_per_episode,
            seed=episode_seed,
            context_id=context_id,
            device_id=device_id,
        )
        stats["random_episodes"] = int(stats["random_episodes"]) + 1
        stats["per_device_collection_contexts"][device_id] += 1
        stats["attempted_env_steps"] = int(stats["attempted_env_steps"]) + int(
            episode.num_transitions
        )
        remaining = int(per_device_targets[device_id]) - int(
            stats["per_device_real_transitions"][device_id]
        )
        kept = min(remaining, int(episode.num_transitions))
        yield episode if kept == int(episode.num_transitions) else _slice_episode(episode, kept)
        stats["stored_real_transitions"] = int(stats["stored_real_transitions"]) + kept
        stats["per_device_real_transitions"][device_id] += kept
        if abstract_episode is not None:
            yield abstract_episode
            stats["synthetic_abstract_edges"] = int(stats["synthetic_abstract_edges"]) + 1

        if int(stats["stored_real_transitions"]) >= target:
            break
        if (
            int(stats["per_device_real_transitions"][device_id])
            >= int(per_device_targets[device_id])
        ):
            context_id += 1
            continue

        num_teacher = teacher_floor + int(float(rng.uniform()) < teacher_fraction)
        task_goal_obs = env.abstract_goal_for_task(device_id).astype(np.float32)
        for teacher_index in range(num_teacher):
            teacher_seed = (
                None
                if seed is None
                else int(seed + 1_000_003 + context_id * 9973 + teacher_index)
            )
            attempted_before = int(stats["attempted_teacher_steps"])
            teacher_episode, teacher_abstract = collect_task_aware_comm_teacher_episode_pair(
                env,
                max_steps=max_steps_per_episode,
                seed=teacher_seed,
                context_id=context_id,
                device_id=device_id,
                task_goal_obs=task_goal_obs,
                collection_stats=stats,
            )
            stats["attempted_env_steps"] = int(stats["attempted_env_steps"]) + (
                int(stats["attempted_teacher_steps"]) - attempted_before
            )
            if teacher_episode is None:
                continue
            stats["successful_teacher_episodes"] = int(
                stats["successful_teacher_episodes"]
            ) + 1
            remaining = int(per_device_targets[device_id]) - int(
                stats["per_device_real_transitions"][device_id]
            )
            if remaining <= 0:
                break
            kept = min(remaining, int(teacher_episode.num_transitions))
            yield (
                teacher_episode
                if kept == int(teacher_episode.num_transitions)
                else _slice_episode(teacher_episode, kept)
            )
            stats["stored_real_transitions"] = int(stats["stored_real_transitions"]) + kept
            stats["per_device_real_transitions"][device_id] += kept
            if teacher_abstract is not None:
                yield teacher_abstract
                stats["synthetic_abstract_edges"] = int(stats["synthetic_abstract_edges"]) + 1
            if int(stats["stored_real_transitions"]) >= target:
                break
        context_id += 1


def collect_qrl_explore_episode_pair(
    env: gym.Env,
    *,
    start_state: np.ndarray,
    start_index: int,
    max_steps: int,
    seed: Optional[int],
    context_id: int,
    device_id: str,
    config: QRLExploreConfig,
    budget_cutoff: bool,
) -> tuple[EpisodeData, Optional[EpisodeData], dict[str, Any]]:
    """Collect one persistent-random, target-blind exploration trajectory."""

    if int(max_steps) <= 0:
        raise ValueError("QRL-explore episode max_steps must be positive")
    rng = np.random.default_rng(seed)
    obs, reset_info = env.reset(
        seed=seed,
        options={"device_id": str(device_id), "start": np.asarray(start_state).tolist()},
    )
    task_goal_obs = env.abstract_goal_observation().astype(np.float32)
    observations = [obs.copy()]
    raw_states = [np.asarray(env.state, dtype=np.float32).copy()]
    actions: list[np.ndarray] = []
    next_observations: list[np.ndarray] = []
    rewards: list[float] = []
    terminals: list[bool] = []
    timeouts: list[bool] = []
    final_info = dict(reset_info)
    action_segments = 0
    nonzero_action_steps = 0
    safety_resampled_segments = 0

    hold_min = int(config.action_hold_min_steps)
    hold_max = int(config.action_hold_max_steps)
    if hold_min <= 0 or hold_max < hold_min:
        raise ValueError("QRL-explore action hold range must satisfy 1 <= min <= max")
    straight_probability = float(config.straight_action_probability)
    if not 0.0 <= straight_probability <= 1.0:
        raise ValueError("QRL-explore straight_action_probability must be in [0, 1]")
    action_scales = np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    action_probabilities = np.asarray(
        [
            0.25 * (1.0 - straight_probability),
            0.25 * (1.0 - straight_probability),
            straight_probability,
            0.25 * (1.0 - straight_probability),
            0.25 * (1.0 - straight_probability),
        ],
        dtype=np.float64,
    )
    safety_lookahead = max(0, int(config.local_safety_lookahead_steps))

    while len(actions) < int(max_steps):
        hold_steps = int(rng.integers(hold_min, hold_max + 1))
        scale = float(rng.choice(action_scales, p=action_probabilities))
        omega = scale * float(env.omega_max)
        if safety_lookahead > 0:
            requested_safety = min(hold_steps, safety_lookahead)
            proposed_safe_steps = _locally_safe_steps(
                env, env.state, omega, requested_safety
            )
            if proposed_safe_steps < requested_safety:
                safety_by_scale = np.asarray(
                    [
                        _locally_safe_steps(
                            env,
                            env.state,
                            float(candidate_scale) * float(env.omega_max),
                            requested_safety,
                        )
                        for candidate_scale in action_scales
                    ],
                    dtype=np.int64,
                )
                best_safety = int(safety_by_scale.max())
                best_indices = np.flatnonzero(safety_by_scale == best_safety)
                chosen_index = int(rng.choice(best_indices))
                omega = float(action_scales[chosen_index]) * float(env.omega_max)
                safety_resampled_segments += 1
                if best_safety > 0:
                    hold_steps = min(hold_steps, best_safety)
        action_segments += 1
        for _ in range(min(hold_steps, int(max_steps) - len(actions))):
            action = np.asarray([omega], dtype=np.float32)
            next_obs, reward, terminated, truncated, info = env.step(action)
            actions.append(action)
            nonzero_action_steps += int(abs(omega) > 1e-12)
            observations.append(next_obs.copy())
            raw_states.append(np.asarray(env.state, dtype=np.float32).copy())
            next_observations.append(next_obs.copy())
            rewards.append(float(reward))
            terminals.append(bool(terminated))
            timeouts.append(bool(truncated))
            final_info = dict(info)
            if terminated or truncated:
                break
        if bool(terminals[-1]) or bool(timeouts[-1]):
            break

    if not actions:
        raise RuntimeError("QRL-explore produced an empty trajectory")
    if not terminals[-1] and not timeouts[-1]:
        timeouts[-1] = True

    if bool(final_info.get("success", False)):
        outcome = EXPLORE_OUTCOME_SUCCESS
    elif bool(final_info.get("collision", False)):
        outcome = EXPLORE_OUTCOME_COLLISION
    elif bool(final_info.get("out_of_bounds", False)):
        outcome = EXPLORE_OUTCOME_OUT_OF_BOUNDS
    elif bool(budget_cutoff):
        outcome = EXPLORE_OUTCOME_BUDGET_CUTOFF
    else:
        outcome = EXPLORE_OUTCOME_TIMEOUT

    keys = [
        _state_heading_key(
            state,
            env,
            position_resolution=float(config.start_position_resolution),
            heading_bins=int(config.start_heading_bins),
        )
        for state in raw_states
    ]
    loop_detected = _trajectory_has_loop(keys)
    actions_array = _actions_to_array(actions)
    n = len(actions)
    rewards_array = np.asarray(rewards, dtype=np.float32)
    success = outcome == EXPLORE_OUTCOME_SUCCESS
    episode = EpisodeData.from_simple_trajectory(
        observations=np.asarray(observations[:-1], dtype=np.float32),
        actions=actions_array,
        next_observations=np.asarray(next_observations, dtype=np.float32),
        rewards=rewards_array,
        terminals=np.asarray(terminals, dtype=np.bool_),
        timeouts=np.asarray(timeouts, dtype=np.bool_),
        transition_infos=_goal_set_transition_infos(
            n=n,
            task_goal_obs=task_goal_obs,
            context_id=context_id,
            env=env,
            global_push_seed=None if seed is None else int(seed + 49_979_687),
            include_global_push_pairs=True,
            exploration=True,
            exploration_start_index=int(start_index),
            exploration_episode_id=int(context_id),
            exploration_outcome=int(outcome),
            exploration_loop_detected=bool(loop_detected),
            task_success_episode=success,
            goal_return_costs=(
                _remaining_nonnegative_costs(rewards_array) if success else None
            ),
        ),
    )

    abstract_episode = None
    try:
        terminal_obs = np.asarray(observations[-1], dtype=np.float32)
        if outcome != EXPLORE_OUTCOME_SUCCESS:
            terminal_state = env.sample_task_terminal_state(
                seed=None if seed is None else int(seed + 104_729)
            )
            terminal_obs = env.state_to_observation(terminal_state).astype(np.float32)
        abstract_episode = _make_goal_set_abstract_edge(
            env,
            terminal_obs=terminal_obs,
            task_goal_obs=task_goal_obs,
            context_id=context_id,
            action_template=actions_array[-1],
            action_dtype=actions_array.dtype,
        )
    except RuntimeError:
        abstract_episode = None

    return episode, abstract_episode, {
        "outcome": EXPLORE_OUTCOME_NAMES[int(outcome)],
        "loop_detected": bool(loop_detected),
        "action_segments": int(action_segments),
        "nonzero_action_steps": int(nonzero_action_steps),
        "safety_resampled_segments": int(safety_resampled_segments),
        "state_heading_keys": keys,
        "raw_states": raw_states,
    }


def create_qrl_explore_comm_dataset(
    env: gym.Env,
    *,
    config: QRLExploreConfig,
    max_steps_per_episode: int,
    seed: Optional[int],
    collection_stats: Optional[dict[str, Any]] = None,
) -> Iterator[EpisodeData]:
    """Collect an exact attempted-step budget with no goal-directed controller.

    Every executed transition is retained. Synthetic terminal-set edges are
    emitted separately and never count against the environment interaction
    budget.
    """

    budget = int(config.attempted_env_steps)
    if budget <= 0:
        raise ValueError("QRL-explore attempted_env_steps must be positive")
    if int(max_steps_per_episode) <= 0:
        raise ValueError("QRL-explore max_steps_per_episode must be positive")
    device_ids = list(getattr(env, "device_ids", ()))
    if not device_ids:
        raise ValueError("QRL-explore communication dataset requires device_ids")
    if config.start_states:
        start_bank = np.asarray(config.start_states, dtype=np.float32).reshape((-1, 3))
    else:
        start_bank = build_qrl_exploration_start_bank(env, config, seed=seed)
    if len(start_bank) == 0:
        raise ValueError("QRL-explore start bank is empty")
    regions = _validated_diagnostic_regions(config.diagnostic_regions)
    routes = _validated_diagnostic_routes(config.diagnostic_routes, regions)
    start_strata = _validated_start_strata(config.start_strata)
    stratum_schedule = _weighted_stratum_schedule(start_strata, seed=seed)
    start_indices_by_stratum: dict[str, np.ndarray] = {}
    for stratum_name, _weight, (x_min, y_min, x_max, y_max) in start_strata:
        mask = (
            (start_bank[:, 0] >= x_min)
            & (start_bank[:, 0] <= x_max)
            & (start_bank[:, 1] >= y_min)
            & (start_bank[:, 1] <= y_max)
        )
        indices = np.flatnonzero(mask).astype(np.int64)
        if len(indices) == 0:
            raise ValueError(
                f"QRL-explore start stratum {stratum_name!r} contains no start-bank states"
            )
        start_indices_by_stratum[stratum_name] = indices

    per_device_targets = {
        device_id: budget // len(device_ids) + int(index < budget % len(device_ids))
        for index, device_id in enumerate(device_ids)
    }
    stats = collection_stats if collection_stats is not None else {}
    stats.clear()
    stats.update(
        {
            "collection_mode": "qrl_explore",
            "attempted_env_step_budget": budget,
            "attempted_env_steps": 0,
            "stored_real_transitions": 0,
            "successful_real_transitions": 0,
            "synthetic_abstract_edges": 0,
            "episodes": 0,
            "loop_episodes": 0,
            "action_segments": 0,
            "nonzero_action_steps": 0,
            "safety_resampled_segments": 0,
            "outcomes": {name: 0 for name in EXPLORE_OUTCOME_NAMES.values() if name != "unknown"},
            "start_bank_size": int(len(start_bank)),
            "unique_start_indices": 0,
            "unique_state_heading_bins": 0,
            "directed_state_heading_edges": 0,
            "per_device_real_transitions": {device_id: 0 for device_id in device_ids},
            "per_device_successful_real_transitions": {
                device_id: 0 for device_id in device_ids
            },
            "per_device_target_transitions": dict(per_device_targets),
            "per_device_episodes": {device_id: 0 for device_id in device_ids},
            "per_device_outcomes": {
                device_id: {
                    name: 0 for name in EXPLORE_OUTCOME_NAMES.values() if name != "unknown"
                }
                for device_id in device_ids
            },
            "start_stratum_episodes": {
                name: 0 for name, _weight, _bounds in start_strata
            },
            "per_device_start_stratum_episodes": {
                device_id: {
                    name: 0 for name, _weight, _bounds in start_strata
                }
                for device_id in device_ids
            },
            "start_stratum_weights": {
                name: float(weight) for name, weight, _bounds in start_strata
            },
            "diagnostic_region_episode_visits": {name: 0 for name in regions},
            "diagnostic_directed_region_crossings": {
                f"{source}->{target}": 0
                for source in regions
                for target in regions
                if source != target
            },
            "per_device_diagnostic_region_episode_visits": {
                device_id: {name: 0 for name in regions}
                for device_id in device_ids
            },
            "per_device_diagnostic_directed_region_crossings": {
                device_id: {
                    f"{source}->{target}": 0
                    for source in regions
                    for target in regions
                    if source != target
                }
                for device_id in device_ids
            },
            "diagnostic_route_traversals": {name: 0 for name in routes},
            "per_device_diagnostic_route_traversals": {
                device_id: {name: 0 for name in routes}
                for device_id in device_ids
            },
        }
    )
    used_start_indices: set[int] = set()
    visited_keys: set[tuple[int, int, int]] = set()
    directed_edges: set[tuple[tuple[int, int, int], tuple[int, int, int]]] = set()
    start_cursors: dict[tuple[str, str], int] = {}
    for device_index, device_id in enumerate(device_ids):
        start_cursors[(device_id, "__uniform__")] = (
            device_index * 104_729
        ) % len(start_bank)
        for stratum_index, (stratum_name, _weight, _bounds) in enumerate(start_strata):
            candidates = start_indices_by_stratum[stratum_name]
            start_cursors[(device_id, stratum_name)] = (
                device_index * 104_729 + stratum_index * 7_919
            ) % len(candidates)
    stratum_schedule_cursors = {
        device_id: device_index * 17
        for device_index, device_id in enumerate(device_ids)
    }
    context_id = 0
    device_cursor = 0

    while int(stats["attempted_env_steps"]) < budget:
        for _ in range(len(device_ids)):
            device_id = device_ids[device_cursor % len(device_ids)]
            device_cursor += 1
            if int(stats["per_device_real_transitions"][device_id]) < int(
                per_device_targets[device_id]
            ):
                break
        else:  # pragma: no cover - guarded by the exact total budget
            raise RuntimeError("QRL-explore exhausted all per-device budgets early")

        env.set_task_by_device_id(device_id)
        requested_start_stratum = ""
        if stratum_schedule:
            schedule_cursor = int(stratum_schedule_cursors[device_id])
            requested_start_stratum = stratum_schedule[
                schedule_cursor % len(stratum_schedule)
            ]
            stratum_schedule_cursors[device_id] = schedule_cursor + 1
            candidate_indices = start_indices_by_stratum[requested_start_stratum]
            cursor_key = (device_id, requested_start_stratum)
        else:
            candidate_indices = np.arange(len(start_bank), dtype=np.int64)
            cursor_key = (device_id, "__uniform__")
        start_index = -1
        start_state = None
        for _ in range(len(candidate_indices)):
            cursor = int(start_cursors[cursor_key])
            candidate_index = int(candidate_indices[cursor % len(candidate_indices)])
            start_cursors[cursor_key] = cursor + 1
            candidate = start_bank[candidate_index]
            if env.is_valid_state(candidate) and not env.is_terminal_goal_state(candidate):
                start_index = candidate_index
                start_state = candidate
                break
        if start_state is None:
            raise RuntimeError(f"QRL-explore found no non-terminal start for {device_id}")

        remaining_total = budget - int(stats["attempted_env_steps"])
        remaining_device = int(per_device_targets[device_id]) - int(
            stats["per_device_real_transitions"][device_id]
        )
        episode_limit = min(int(max_steps_per_episode), remaining_total, remaining_device)
        budget_cutoff = episode_limit < int(max_steps_per_episode)
        episode_seed = None if seed is None else int(seed + 1_000_003 + context_id * 104_729)
        episode, abstract_episode, diagnostics = collect_qrl_explore_episode_pair(
            env,
            start_state=start_state,
            start_index=start_index,
            max_steps=episode_limit,
            seed=episode_seed,
            context_id=context_id,
            device_id=device_id,
            config=config,
            budget_cutoff=budget_cutoff,
        )
        real_steps = int(episode.num_transitions)
        yield episode
        if abstract_episode is not None:
            yield abstract_episode
            stats["synthetic_abstract_edges"] = int(stats["synthetic_abstract_edges"]) + 1

        stats["attempted_env_steps"] = int(stats["attempted_env_steps"]) + real_steps
        stats["stored_real_transitions"] = int(stats["stored_real_transitions"]) + real_steps
        stats["episodes"] = int(stats["episodes"]) + 1
        stats["per_device_real_transitions"][device_id] += real_steps
        stats["per_device_episodes"][device_id] += 1
        outcome = str(diagnostics["outcome"])
        stats["outcomes"][outcome] += 1
        stats["per_device_outcomes"][device_id][outcome] += 1
        if outcome == "success":
            stats["successful_real_transitions"] = int(
                stats["successful_real_transitions"]
            ) + real_steps
            stats["per_device_successful_real_transitions"][device_id] += real_steps
        stats["loop_episodes"] = int(stats["loop_episodes"]) + int(
            diagnostics["loop_detected"]
        )
        stats["action_segments"] = int(stats["action_segments"]) + int(
            diagnostics["action_segments"]
        )
        stats["nonzero_action_steps"] = int(stats["nonzero_action_steps"]) + int(
            diagnostics["nonzero_action_steps"]
        )
        stats["safety_resampled_segments"] = int(
            stats["safety_resampled_segments"]
        ) + int(diagnostics["safety_resampled_segments"])
        if requested_start_stratum:
            stats["start_stratum_episodes"][requested_start_stratum] += 1
            stats["per_device_start_stratum_episodes"][device_id][
                requested_start_stratum
            ] += 1
        used_start_indices.add(int(start_index))
        keys = diagnostics["state_heading_keys"]
        visited_keys.update(keys)
        directed_edges.update((left, right) for left, right in zip(keys[:-1], keys[1:]) if left != right)

        raw_states = np.asarray(diagnostics["raw_states"], dtype=np.float32)
        first_region_index: dict[str, int] = {}
        last_region_index: dict[str, int] = {}
        region_indices: dict[str, np.ndarray] = {}
        for region_name, (x_min, y_min, x_max, y_max) in regions.items():
            mask = (
                (raw_states[:, 0] >= x_min)
                & (raw_states[:, 0] <= x_max)
                & (raw_states[:, 1] >= y_min)
                & (raw_states[:, 1] <= y_max)
            )
            indices = np.flatnonzero(mask)
            if len(indices) > 0:
                region_indices[region_name] = indices
                stats["diagnostic_region_episode_visits"][region_name] += 1
                stats["per_device_diagnostic_region_episode_visits"][device_id][
                    region_name
                ] += 1
                first_region_index[region_name] = int(indices[0])
                last_region_index[region_name] = int(indices[-1])
        for source in regions:
            for target in regions:
                if source == target or source not in first_region_index or target not in last_region_index:
                    continue
                if last_region_index[target] > first_region_index[source]:
                    crossing_name = f"{source}->{target}"
                    stats["diagnostic_directed_region_crossings"][crossing_name] += 1
                    stats["per_device_diagnostic_directed_region_crossings"][device_id][
                        crossing_name
                    ] += 1
        for route_name, route in routes.items():
            if _trajectory_matches_region_route(region_indices, route):
                stats["diagnostic_route_traversals"][route_name] += 1
                stats["per_device_diagnostic_route_traversals"][device_id][
                    route_name
                ] += 1

        context_id += 1

    stats["unique_start_indices"] = int(len(used_start_indices))
    stats["unique_state_heading_bins"] = int(len(visited_keys))
    stats["directed_state_heading_edges"] = int(len(directed_edges))
    stats["mean_action_hold_steps"] = float(
        stats["attempted_env_steps"] / max(int(stats["action_segments"]), 1)
    )
    stats["nonzero_action_step_ratio"] = float(
        stats["nonzero_action_steps"] / max(int(stats["attempted_env_steps"]), 1)
    )
    stats["safety_resampled_segment_ratio"] = float(
        stats["safety_resampled_segments"] / max(int(stats["action_segments"]), 1)
    )
    stats["failed_episodes"] = int(stats["episodes"]) - int(
        stats["outcomes"]["success"]
    )
    stats["natural_exit_episodes"] = int(stats["outcomes"]["timeout"])


def create_dataset(
    env: gym.Env,
    num_episodes: int = 100,
    max_steps_per_episode: int = 200,
    sample_valid_states: bool = True,
    seed: Optional[int] = None,
    task_aware_teacher_ratio: float = 0.0,
    target_env_transitions: Optional[int] = None,
    collection_stats: Optional[dict[str, Any]] = None,
    qrl_explore_config: Optional[QRLExploreConfig] = None,
) -> Iterator[EpisodeData]:
    """
    创建数据集，支持多种环境
    
    Args:
        env: 环境实例（应实现 BaseNavigationEnv 接口）
        num_episodes: episode 数量
        max_steps_per_episode: 每个 episode 的最大步数
        sample_valid_states: 是否使用环境的 sample_valid_state 方法采样合法状态
        seed: 随机种子
        task_aware_teacher_ratio: 通信巡检 goal-set 数据中额外追加的 teacher 成功轨迹比例。
            1.0 表示每个 random rollout 额外收集 1 条 guided 成功轨迹。
    
    Yields:
        EpisodeData
    """
    if qrl_explore_config is not None:
        if not (
            hasattr(env, "abstract_goal_observation")
            and hasattr(env, "device_ids")
        ):
            raise ValueError("QRL-explore is only supported for communication goal-set environments")
        yield from create_qrl_explore_comm_dataset(
            env,
            config=qrl_explore_config,
            max_steps_per_episode=max_steps_per_episode,
            seed=seed,
            collection_stats=collection_stats,
        )
        return

    if (
        target_env_transitions is not None
        and hasattr(env, "abstract_goal_observation")
        and hasattr(env, "device_ids")
    ):
        yield from create_budgeted_comm_dataset(
            env,
            target_env_transitions=int(target_env_transitions),
            max_steps_per_episode=max_steps_per_episode,
            seed=seed,
            task_aware_teacher_ratio=task_aware_teacher_ratio,
            collection_stats=collection_stats,
        )
        return

    if getattr(env, 'dataset_mode', None) == 'discrete_graph' and hasattr(env, 'iter_discrete_transitions'):
        for state, action, next_state, reward, done in env.iter_discrete_transitions():
            yield EpisodeData.from_simple_trajectory(
                observations=np.array([state], dtype=np.float32),
                actions=np.array([action], dtype=np.int64),
                next_observations=np.array([next_state], dtype=np.float32),
                rewards=np.array([reward], dtype=np.float32),
                terminals=np.array([done], dtype=np.bool_),
                timeouts=np.array([False], dtype=np.bool_),
            )
        return

    if hasattr(env, "abstract_goal_observation") and hasattr(env, "is_terminal_goal_state"):
        teacher_ratio = max(0.0, float(task_aware_teacher_ratio))
        teacher_floor = int(np.floor(teacher_ratio))
        teacher_fraction = teacher_ratio - float(teacher_floor)
        teacher_rng = np.random.default_rng(None if seed is None else int(seed) + 91815541)
        for i in range(num_episodes):
            episode_seed = (seed + i) if seed is not None else None
            episode, abstract_episode = collect_goal_set_comm_episode_pair(
                env,
                max_steps=max_steps_per_episode,
                seed=episode_seed,
                context_id=i,
            )
            shared_device_id = str(env.active_device_id)
            shared_task_goal_obs = env.abstract_goal_observation().astype(np.float32)
            yield episode
            if abstract_episode is not None:
                yield abstract_episode

            num_teacher = teacher_floor + int(float(teacher_rng.uniform(0.0, 1.0)) < teacher_fraction)
            for teacher_idx in range(num_teacher):
                teacher_seed = None if seed is None else int(seed + 1_000_003 + i * 9973 + teacher_idx)
                teacher_episode, teacher_abstract_episode = collect_task_aware_comm_teacher_episode_pair(
                    env,
                    max_steps=max_steps_per_episode,
                    seed=teacher_seed,
                    context_id=i,
                    device_id=shared_device_id,
                    task_goal_obs=shared_task_goal_obs,
                )
                if teacher_episode is not None:
                    yield teacher_episode
                if teacher_abstract_episode is not None:
                    yield teacher_abstract_episode
        return

    for i in range(num_episodes):
        episode_seed = (seed + i) if seed is not None else None
        yield collect_random_episode(
            env,
            max_steps=max_steps_per_episode,
            sample_valid_start=sample_valid_states,
            seed=episode_seed
        )

    if getattr(env, 'dataset_mode', None) == 'random_policy_paper' and hasattr(env, 'iter_added_goal_transitions'):
        for state, action, next_state, reward, done in env.iter_added_goal_transitions():
            yield EpisodeData.from_simple_trajectory(
                observations=np.array([state], dtype=np.float32),
                actions=np.array([action], dtype=np.int64),
                next_observations=np.array([next_state], dtype=np.float32),
                rewards=np.array([reward], dtype=np.float32),
                terminals=np.array([done], dtype=np.bool_),
                timeouts=np.array([False], dtype=np.bool_),
            )


# 为了向后兼容，保留旧函数名
def create_simple_dataset(
    env: gym.Env,
    num_episodes: int = 100,
    max_steps_per_episode: int = 200,
) -> Iterator[EpisodeData]:
    """
    创建简单的数据集（向后兼容函数）
    
    Args:
        env: 环境实例
        num_episodes: episode 数量
        max_steps_per_episode: 每个 episode 的最大步数
    
    Yields:
        EpisodeData
    """
    return create_dataset(
        env=env,
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps_per_episode,
        sample_valid_states=True,
    )
