"""
创建简单的数据集用于 QRL 训练
"""
import numpy as np
import random
from typing import Iterator

from quasimetric_rl.data import EpisodeData
from minimal_qrl.simple_env import SimpleGrid2D


def collect_random_episode(env: SimpleGrid2D, max_steps: int = 200) -> EpisodeData:
    """收集一个随机 episode"""
    obs, _ = env.reset()
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
        
        observations.append(next_obs.copy())
        actions.append(int(action))
        next_observations.append(next_obs.copy())
        rewards.append(float(reward))
        terminals.append(bool(terminated))
        timeouts.append(bool(truncated))
        
        if terminated or truncated:
            break
    
    # 转换为 EpisodeData
    return EpisodeData.from_simple_trajectory(
        observations=np.array(observations[:-1], dtype=np.float32),  # 去掉最后一个
        actions=np.array(actions, dtype=np.int64),
        next_observations=np.array(next_observations, dtype=np.float32),
        rewards=np.array(rewards, dtype=np.float32),
        terminals=np.array(terminals, dtype=np.bool_),
        timeouts=np.array(timeouts, dtype=np.bool_),
    )


def create_simple_dataset(
    env: SimpleGrid2D,
    num_episodes: int = 100,
    max_steps_per_episode: int = 200,
) -> Iterator[EpisodeData]:
    """创建简单的数据集"""
    for _ in range(num_episodes):
        yield collect_random_episode(env, max_steps=max_steps_per_episode)

