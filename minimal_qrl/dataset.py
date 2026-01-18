"""
创建数据集用于 QRL 训练，支持多种环境
"""
import numpy as np
import random
from typing import Iterator, Optional
import gym

from quasimetric_rl.data import EpisodeData
from minimal_qrl.envs.base import BaseNavigationEnv


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
    
    # 转换为 EpisodeData
    # 处理动作数组
    # 检查动作类型：如果所有动作都是标量，使用 int64；否则使用 float32
    if len(actions) > 0:
        first_action = actions[0]
        if isinstance(first_action, (int, np.integer)):
            # 离散动作
            actions_array = np.array(actions, dtype=np.int64)
        else:
            # 连续动作
            actions_array = np.array(actions, dtype=np.float32)
    else:
        # 空动作列表（不应该发生）
        actions_array = np.array([], dtype=np.int64)
    
    return EpisodeData.from_simple_trajectory(
        observations=np.array(observations[:-1], dtype=np.float32),  # 去掉最后一个
        actions=actions_array,
        next_observations=np.array(next_observations, dtype=np.float32),
        rewards=np.array(rewards, dtype=np.float32),
        terminals=np.array(terminals, dtype=np.bool_),
        timeouts=np.array(timeouts, dtype=np.bool_),
    )


def create_dataset(
    env: gym.Env,
    num_episodes: int = 100,
    max_steps_per_episode: int = 200,
    sample_valid_states: bool = True,
    seed: Optional[int] = None,
) -> Iterator[EpisodeData]:
    """
    创建数据集，支持多种环境
    
    Args:
        env: 环境实例（应实现 BaseNavigationEnv 接口）
        num_episodes: episode 数量
        max_steps_per_episode: 每个 episode 的最大步数
        sample_valid_states: 是否使用环境的 sample_valid_state 方法采样合法状态
        seed: 随机种子
    
    Yields:
        EpisodeData
    """
    for i in range(num_episodes):
        episode_seed = (seed + i) if seed is not None else None
        yield collect_random_episode(
            env,
            max_steps=max_steps_per_episode,
            sample_valid_start=sample_valid_states,
            seed=episode_seed
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

