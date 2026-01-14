"""
简单的 2D 网格环境，不依赖 mujoco/d4rl
"""
import numpy as np
import gym
import gym.spaces
from typing import Tuple, Optional


class SimpleGrid2D(gym.Env):
    """
    简单的 2D 网格环境，用于 QRL 训练
    
    - 状态: 归一化的 (x, y) 坐标，范围 [0, 1]^2
    - 动作: 0=上, 1=右, 2=下, 3=左 (4邻域)
    - 奖励: 到达目标时 +1，否则 -0.01 (步数惩罚)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(
        self,
        grid_size: Tuple[int, int] = (10, 10),
        start_pos: Optional[Tuple[int, int]] = None,
        goal_pos: Optional[Tuple[int, int]] = None,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 200,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.height, self.width = grid_size
        self.start_pos = start_pos if start_pos else (1, 1)
        self.goal_pos = goal_pos if goal_pos else (self.height - 2, self.width - 2)
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        
        # 动作空间: 0=上, 1=右, 2=下, 3=左
        self.action_space = gym.spaces.Discrete(4)
        # 观察空间: 归一化的 (x, y) 坐标
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(2,), dtype=np.float32
        )
        
        self.agent_pos: Tuple[int, int] = self.start_pos
        self._t = 0
    
    def _get_obs(self) -> np.ndarray:
        """获取当前观察（归一化的坐标）"""
        x, y = self.agent_pos
        return np.array([
            x / (self.height - 1),
            y / (self.width - 1)
        ], dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        self.agent_pos = self.start_pos
        self._t = 0
        return self._get_obs(), {}
    
    def step(self, action: int):
        """执行一步"""
        x, y = self.agent_pos
        # 动作映射: 0=上(-1,0), 1=右(0,1), 2=下(1,0), 3=左(0,-1)
        dx, dy = [(-1, 0), (0, 1), (1, 0), (0, -1)][int(action)]
        nx = int(np.clip(x + dx, 0, self.height - 1))
        ny = int(np.clip(y + dy, 0, self.width - 1))
        self.agent_pos = (nx, ny)
        
        self._t += 1
        terminated = (self.agent_pos == self.goal_pos)
        truncated = (self._t >= self.max_episode_steps)
        
        # 奖励: 到达目标 +1，否则 -0.01 (步数惩罚)
        reward = 1.0 if terminated else -0.01
        
        return self._get_obs(), float(reward), terminated, truncated, {"is_success": terminated}
    
    def render(self):
        """渲染环境（文本模式）"""
        if self.render_mode != "human":
            return
        
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        sx, sy = self.start_pos
        gx, gy = self.goal_pos
        grid[sx][sy] = 'S'
        grid[gx][gy] = 'G'
        ax, ay = self.agent_pos
        if grid[ax][ay] not in ['S', 'G']:
            grid[ax][ay] = 'A'
        
        print("\n".join(["".join(row) for row in grid]))
        print()
