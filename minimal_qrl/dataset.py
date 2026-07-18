"""
创建数据集用于 QRL 训练，支持多种环境
"""
import numpy as np
import random
import torch
import warnings
from typing import Any, Iterator, Optional, List, Tuple
import gym

from quasimetric_rl.data import EpisodeData
from minimal_qrl.envs.base import BaseNavigationEnv


TASK_AWARE_TEACHER_MAX_ATTEMPTS = 8
TASK_AWARE_TEACHER_KP = 4.0
TASK_AWARE_TEACHER_ACTION_NOISE_STD = 0.0


def _actions_to_array(actions: List) -> np.ndarray:
    if len(actions) > 0:
        first_action = actions[0]
        if isinstance(first_action, (int, np.integer)):
            return np.array(actions, dtype=np.int64)
        return np.array(actions, dtype=np.float32)
    return np.array([], dtype=np.int64)


def _normalize_angle(theta: float) -> float:
    return float((float(theta) + np.pi) % (2.0 * np.pi) - np.pi)


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
) -> dict:
    infos = {
        "abstract_goal_edge": np.full((n,), bool(abstract_goal_edge), dtype=np.bool_),
        "source_terminal_goal_state": np.full((n,), bool(source_terminal_goal_state), dtype=np.bool_),
        "task_goal_observations": np.repeat(task_goal_obs[None, :], n, axis=0).astype(np.float32),
        "context_id": np.full((n,), int(context_id), dtype=np.int64),
        "teacher_guided": np.full((n,), bool(teacher_guided), dtype=np.bool_),
    }
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
    episode = EpisodeData.from_simple_trajectory(
        observations=np.array(observations[:-1], dtype=np.float32),
        actions=actions_array,
        next_observations=np.array(next_observations, dtype=np.float32),
        rewards=np.array(rewards, dtype=np.float32),
        terminals=np.array(terminals, dtype=np.bool_),
        timeouts=np.array(timeouts, dtype=np.bool_),
        transition_infos=_goal_set_transition_infos(
            n=n,
            task_goal_obs=task_goal_obs,
            context_id=context_id,
            env=env,
            global_push_seed=None if seed is None else int(seed + 32452843),
            include_global_push_pairs=True,
        ),
    )

    abstract_episode = None
    try:
        terminal_obs = np.asarray(obs, dtype=np.float32)
        if not (bool(final_info.get("success", False)) and n > 0):
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
        episode = EpisodeData.from_simple_trajectory(
                observations=np.array(observations[:-1], dtype=np.float32),
                actions=actions_array,
                next_observations=np.array(next_observations, dtype=np.float32),
                rewards=np.array(rewards, dtype=np.float32),
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


def create_dataset(
    env: gym.Env,
    num_episodes: int = 100,
    max_steps_per_episode: int = 200,
    sample_valid_states: bool = True,
    seed: Optional[int] = None,
    task_aware_teacher_ratio: float = 0.0,
    target_env_transitions: Optional[int] = None,
    collection_stats: Optional[dict[str, Any]] = None,
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
