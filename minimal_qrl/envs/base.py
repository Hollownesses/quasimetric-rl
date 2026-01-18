"""
环境基类接口，定义所有环境需要实现的方法
"""
import numpy as np
from typing import Tuple, Optional
import gym


class BaseNavigationEnv(gym.Env):
    """
    导航环境基类，定义统一接口
    
    所有导航环境应该实现以下方法：
    - is_valid_state: 检查状态是否合法
    - sample_valid_state: 采样合法状态
    - compute_shortest_path_distance: 计算真实最短路径距离（可选）
    """
    
    def is_valid_state(self, state: np.ndarray) -> bool:
        """
        检查状态是否合法（不在障碍物内等）
        
        Args:
            state: 状态数组，形状为 (obs_dim,)
        
        Returns:
            是否为合法状态
        """
        raise NotImplementedError
    
    def sample_valid_state(self, seed: Optional[int] = None) -> np.ndarray:
        """
        采样一个合法状态
        
        Args:
            seed: 随机种子
        
        Returns:
            合法状态数组，形状为 (obs_dim,)
        """
        raise NotImplementedError
    
    def compute_shortest_path_distance(
        self,
        start: Optional[np.ndarray] = None,
        goal: Optional[np.ndarray] = None
    ) -> float:
        """
        计算从起点到终点的真实最短路径距离
        
        如果环境没有实现此方法，可以返回欧几里得距离作为默认值
        
        Args:
            start: 起始状态，如果为 None 则使用当前环境的起点
            goal: 目标状态，如果为 None 则使用当前环境的目标
        
        Returns:
            最短路径距离（标量）
        """
        # 默认实现：使用欧几里得距离
        if start is None or goal is None:
            # 如果环境有 start 和 goal 属性，使用它们
            if hasattr(self, 'start') and hasattr(self, 'goal'):
                start = np.array(self.start) if start is None else start
                goal = np.array(self.goal) if goal is None else goal
            elif hasattr(self, 'start_pos') and hasattr(self, 'goal_pos'):
                # SimpleGrid2D 使用 start_pos 和 goal_pos
                h, w = self.grid_size
                if start is None:
                    sx, sy = self.start_pos
                    start = np.array([sx / (h - 1), sy / (w - 1)], dtype=np.float32)
                if goal is None:
                    gx, gy = self.goal_pos
                    goal = np.array([gx / (h - 1), gy / (w - 1)], dtype=np.float32)
            else:
                raise ValueError("无法确定起点或终点")
        
        return float(np.linalg.norm(start - goal))
