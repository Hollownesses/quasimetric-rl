"""
2D 连续空间导航环境，包含矩形障碍物
"""
import numpy as np
import gym
from gym import spaces
from typing import Tuple, Optional, List
from dataclasses import dataclass

from .base import BaseNavigationEnv


@dataclass
class Obstacle:
    """轴对齐矩形障碍物"""
    x_min: float  # x 下界 [0, 1]
    x_max: float  # x 上界 [0, 1]
    y_min: float  # y 下界 [0, 1]
    y_max: float  # y 上界 [0, 1]
    
    def contains(self, x: float, y: float) -> bool:
        """检查点是否在障碍物内（包含边界）"""
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max
    
    def intersects_segment(self, x1: float, y1: float, x2: float, y2: float) -> bool:
        """检查线段是否与障碍物相交"""
        # 使用分离轴定理（SAT）检查线段与轴对齐矩形的相交
        # 对于轴对齐矩形，简化版本：检查线段端点是否在矩形内，或线段是否穿过矩形
        
        # 如果任一端点在矩形内，则相交
        if self.contains(x1, y1) or self.contains(x2, y2):
            return True
        
        # 检查线段是否与矩形的四条边相交
        # 矩形边：x=x_min, x=x_max, y=y_min, y=y_max
        
        # 检查是否与 x=x_min 或 x=x_max 相交
        if min(x1, x2) <= self.x_min <= max(x1, x2):
            # 计算在 x=x_min 处的 y 值
            if x2 != x1:
                t = (self.x_min - x1) / (x2 - x1)
                y_inter = y1 + t * (y2 - y1)
                if self.y_min <= y_inter <= self.y_max:
                    return True
        
        if min(x1, x2) <= self.x_max <= max(x1, x2):
            if x2 != x1:
                t = (self.x_max - x1) / (x2 - x1)
                y_inter = y1 + t * (y2 - y1)
                if self.y_min <= y_inter <= self.y_max:
                    return True
        
        # 检查是否与 y=y_min 或 y=y_max 相交
        if min(y1, y2) <= self.y_min <= max(y1, y2):
            if y2 != y1:
                t = (self.y_min - y1) / (y2 - y1)
                x_inter = x1 + t * (x2 - x1)
                if self.x_min <= x_inter <= self.x_max:
                    return True
        
        if min(y1, y2) <= self.y_max <= max(y1, y2):
            if y2 != y1:
                t = (self.y_max - y1) / (y2 - y1)
                x_inter = x1 + t * (x2 - x1)
                if self.x_min <= x_inter <= self.x_max:
                    return True
        
        return False


class ContinuousObstacle2D(BaseNavigationEnv):
    """
    2D 连续空间导航环境，包含矩形障碍物
    
    - 状态: 连续二维坐标 (x, y)，范围 [0, 1]^2
    - 动作: 连续位移 (dx, dy)，带最大步长限制
    - 障碍物: 轴对齐矩形，agent 不允许穿越或进入
    - 奖励: 到达目标时 +1，否则 -0.01 (步数惩罚)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(
        self,
        obstacles: Optional[List[Obstacle]] = None,
        start: Optional[Tuple[float, float]] = None,
        goal: Optional[Tuple[float, float]] = None,
        max_step_size: float = 0.1,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 200,
        grid_resolution: int = 100,  # 用于 A* 搜索的分辨率
    ):
        super().__init__()
        
        # 默认障碍物配置（如果没有提供）
        if obstacles is None:
            obstacles = [
                Obstacle(x_min=0.3, x_max=0.5, y_min=0.2, y_max=0.4),
                Obstacle(x_min=0.6, x_max=0.8, y_min=0.6, y_max=0.8),
            ]
        
        self.obstacles = obstacles
        self.max_step_size = max_step_size
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.grid_resolution = grid_resolution
        
        # 动作空间: 连续位移 (dx, dy)，范围 [-max_step_size, max_step_size]
        self.action_space = spaces.Box(
            low=-max_step_size,
            high=max_step_size,
            shape=(2,),
            dtype=np.float32
        )
        
        # 观察空间: 连续二维坐标 [0, 1]^2
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )
        
        # 起终点（可在 reset 时设置）
        self.start = start if start else (0.1, 0.1)
        self.goal = goal if goal else (0.9, 0.9)
        
        # 当前状态
        self.agent_pos: Tuple[float, float] = self.start
        self._t = 0
        
        # 用于 A* 的缓存
        self._distance_cache: Optional[dict] = None
    
    def _is_valid_position(self, x: float, y: float) -> bool:
        """检查位置是否合法（不在障碍物内且在边界内）"""
        # 检查边界
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            return False
        
        # 检查是否在障碍物内
        for obs in self.obstacles:
            if obs.contains(x, y):
                return False
        
        return True
    
    def is_valid_state(self, state: np.ndarray) -> bool:
        """
        检查状态是否合法（不在障碍物内且在边界内）
        
        Args:
            state: 状态数组，形状为 (2,)，归一化坐标 [0, 1]^2
        
        Returns:
            是否为合法状态
        """
        return self._is_valid_position(float(state[0]), float(state[1]))
    
    def sample_valid_state(self, seed: Optional[int] = None) -> np.ndarray:
        """
        采样一个合法状态（不在障碍物内）
        
        Args:
            seed: 随机种子
        
        Returns:
            合法状态数组，形状为 (2,)，归一化坐标 [0, 1]^2
        """
        if seed is not None:
            np.random.seed(seed)
        
        # 尝试采样，最多尝试 1000 次
        max_attempts = 1000
        for _ in range(max_attempts):
            x = np.random.uniform(0.0, 1.0)
            y = np.random.uniform(0.0, 1.0)
            state = np.array([x, y], dtype=np.float32)
            if self.is_valid_state(state):
                return state
        
        # 如果采样失败，返回起点（应该是合法的）
        return np.array(self.start, dtype=np.float32)
    
    def _project_to_valid(self, x: float, y: float) -> Tuple[float, float]:
        """将位置投影到最近的合法区域"""
        # 首先限制在边界内
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)
        
        # 如果已经在合法位置，直接返回
        if self._is_valid_position(x, y):
            return (x, y)
        
        # 否则，尝试找到最近的合法位置
        # 简单策略：向障碍物外移动
        best_x, best_y = x, y
        best_dist = float('inf')
        
        # 尝试多个方向
        for dx in [-0.01, 0.0, 0.01]:
            for dy in [-0.01, 0.0, 0.01]:
                test_x = np.clip(x + dx, 0.0, 1.0)
                test_y = np.clip(y + dy, 0.0, 1.0)
                if self._is_valid_position(test_x, test_y):
                    dist = (test_x - x) ** 2 + (test_y - y) ** 2
                    if dist < best_dist:
                        best_dist = dist
                        best_x, best_y = test_x, test_y
        
        # 如果还是找不到，返回边界上的点
        if best_dist == float('inf'):
            return (np.clip(x, 0.0, 1.0), np.clip(y, 0.0, 1.0))
        
        return (best_x, best_y)
    
    def _check_collision(self, x1: float, y1: float, x2: float, y2: float) -> bool:
        """检查从 (x1, y1) 到 (x2, y2) 的移动是否穿越障碍物"""
        # 检查线段是否与任何障碍物相交
        for obs in self.obstacles:
            if obs.intersects_segment(x1, y1, x2, y2):
                return True
        return False
    
    def _get_obs(self) -> np.ndarray:
        """获取当前观察（归一化的坐标）"""
        x, y = self.agent_pos
        return np.array([x, y], dtype=np.float32)
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        重置环境
        
        Args:
            seed: 随机种子
            options: 可选参数，可包含 'start' 和 'goal' 键
        
        Returns:
            observation, info
        """
        super().reset(seed=seed)
        
        # 如果提供了 start 和 goal，使用它们
        if options is not None:
            if 'start' in options:
                self.start = tuple(options['start'])
            if 'goal' in options:
                self.goal = tuple(options['goal'])
        
        # 确保起终点合法
        self.start = self._project_to_valid(*self.start)
        self.goal = self._project_to_valid(*self.goal)
        
        self.agent_pos = self.start
        self._t = 0
        
        # 清除距离缓存
        self._distance_cache = None
        
        return self._get_obs(), {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        执行一步
        
        Args:
            action: 动作 (dx, dy)
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        action = np.array(action, dtype=np.float32).flatten()
        if len(action) != 2:
            raise ValueError(f"动作维度应为 2，得到 {len(action)}")
        
        dx, dy = action[0], action[1]
        
        # 限制动作幅度
        action_norm = np.sqrt(dx ** 2 + dy ** 2)
        if action_norm > self.max_step_size:
            scale = self.max_step_size / action_norm
            dx *= scale
            dy *= scale
        
        # 计算新位置
        x, y = self.agent_pos
        new_x = x + dx
        new_y = y + dy
        
        # 检查是否穿越障碍物
        collision_occurred = False
        if self._check_collision(x, y, new_x, new_y):
            # 如果穿越障碍物，保持当前位置不变
            new_x, new_y = x, y
            collision_occurred = True
        else:
            # 检查新位置是否合法
            if not self._is_valid_position(new_x, new_y):
                # 投影到合法区域
                new_x, new_y = self._project_to_valid(new_x, new_y)
                collision_occurred = True  # 进入障碍物也算碰撞
        
        self.agent_pos = (new_x, new_y)
        self._t += 1
        
        # 检查是否到达目标（使用小的容差）
        dist_to_goal = np.sqrt((new_x - self.goal[0]) ** 2 + (new_y - self.goal[1]) ** 2)
        terminated = dist_to_goal < 0.05  # 容差：0.05
        truncated = self._t >= self.max_episode_steps
        
        # 改进的奖励函数：增加碰撞惩罚
        # 到达目标 +1，碰撞 -0.1（比步数惩罚大10倍），否则 -0.01 (步数惩罚)
        if terminated:
            reward = 1.0
        elif collision_occurred:
            reward = -0.1  # 碰撞惩罚：比步数惩罚大10倍，让模型学习避免碰撞
        else:
            reward = -0.01  # 步数惩罚
        
        return self._get_obs(), float(reward), terminated, truncated, {"is_success": terminated}
    
    def compute_shortest_path_distance(
        self,
        start: Optional[np.ndarray] = None,
        goal: Optional[np.ndarray] = None
    ) -> float:
        """
        计算从起点到终点的真实最短路径距离（使用 grid discretization + A*）
        仅用于评估，不参与训练
        
        Args:
            start: 起始位置（归一化坐标），如果为 None 则使用 self.start
            goal: 目标位置（归一化坐标），如果为 None 则使用 self.goal
        
        Returns:
            最短路径距离（欧几里得距离）
        """
        if start is None:
            start = np.array(self.start, dtype=np.float32)
        else:
            start = np.array(start, dtype=np.float32)
        
        if goal is None:
            goal = np.array(self.goal, dtype=np.float32)
        else:
            goal = np.array(goal, dtype=np.float32)
        
        # 转换为元组格式（用于内部实现）
        start_tuple = (float(start[0]), float(start[1]))
        goal_tuple = (float(goal[0]), float(goal[1]))
        
        # 使用缓存
        cache_key = (start_tuple, goal_tuple)
        if self._distance_cache is not None and cache_key in self._distance_cache:
            return self._distance_cache[cache_key]
        
        # 网格离散化
        grid_size = self.grid_resolution
        grid = np.zeros((grid_size, grid_size), dtype=bool)  # True 表示可通行
        
        # 标记障碍物
        for obs in self.obstacles:
            x_min_idx = int(obs.x_min * grid_size)
            x_max_idx = int(np.ceil(obs.x_max * grid_size))
            y_min_idx = int(obs.y_min * grid_size)
            y_max_idx = int(np.ceil(obs.y_max * grid_size))
            
            x_min_idx = max(0, min(x_min_idx, grid_size - 1))
            x_max_idx = max(0, min(x_max_idx, grid_size - 1))
            y_min_idx = max(0, min(y_min_idx, grid_size - 1))
            y_max_idx = max(0, min(y_max_idx, grid_size - 1))
            
            grid[x_min_idx:x_max_idx+1, y_min_idx:y_max_idx+1] = True  # True 表示障碍物
        
        # 转换为可通行性（False=障碍物，True=可通行）
        grid = ~grid
        
        # 起点和终点在网格中的位置
        sx = int(np.clip(start[0] * grid_size, 0, grid_size - 1))
        sy = int(np.clip(start[1] * grid_size, 0, grid_size - 1))
        gx = int(np.clip(goal[0] * grid_size, 0, grid_size - 1))
        gy = int(np.clip(goal[1] * grid_size, 0, grid_size - 1))
        
        # 如果起点或终点在障碍物内，返回大值
        if not grid[sx, sy] or not grid[gx, gy]:
            distance = float('inf')
            if self._distance_cache is None:
                self._distance_cache = {}
            self._distance_cache[cache_key] = distance
            return distance
        
        # A* 搜索
        from heapq import heappush, heappop
        
        def heuristic(x1, y1, x2, y2):
            """欧几里得距离启发式"""
            return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        
        # (f, g, x, y, parent)
        open_set = [(heuristic(sx, sy, gx, gy), 0.0, sx, sy, None)]
        closed_set = set()
        g_score = {(sx, sy): 0.0}
        parent = {}
        
        while open_set:
            f, g, x, y, p = heappop(open_set)
            
            if (x, y) in closed_set:
                continue
            
            closed_set.add((x, y))
            parent[(x, y)] = p
            
            if x == gx and y == gy:
                # 找到路径，计算总距离
                path = []
                curr = (x, y)
                while curr is not None:
                    path.append(curr)
                    curr = parent.get(curr)
                path.reverse()
                
                # 计算路径总长度（欧几里得距离）
                total_dist = 0.0
                for i in range(len(path) - 1):
                    x1, y1 = path[i]
                    x2, y2 = path[i + 1]
                    # 转换为连续坐标
                    cx1 = x1 / grid_size
                    cy1 = y1 / grid_size
                    cx2 = x2 / grid_size
                    cy2 = y2 / grid_size
                    total_dist += np.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2)
                
                if self._distance_cache is None:
                    self._distance_cache = {}
                self._distance_cache[cache_key] = total_dist
                return total_dist
            
            # 检查邻居（8邻域）
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    nx, ny = x + dx, y + dy
                    
                    if nx < 0 or nx >= grid_size or ny < 0 or ny >= grid_size:
                        continue
                    
                    if not grid[nx, ny]:
                        continue
                    
                    if (nx, ny) in closed_set:
                        continue
                    
                    # 移动成本（欧几里得距离）
                    move_cost = np.sqrt(dx ** 2 + dy ** 2) / grid_size
                    tentative_g = g + move_cost
                    
                    if (nx, ny) not in g_score or tentative_g < g_score[(nx, ny)]:
                        g_score[(nx, ny)] = tentative_g
                        h = heuristic(nx, ny, gx, gy) / grid_size
                        f = tentative_g + h
                        heappush(open_set, (f, tentative_g, nx, ny, (x, y)))
        
        # 未找到路径
        distance = float('inf')
        if self._distance_cache is None:
            self._distance_cache = {}
        self._distance_cache[cache_key] = distance
        return distance
    
    def render(self):
        """渲染环境（文本模式）"""
        if self.render_mode != "human":
            return
        
        # 创建简单的文本表示
        resolution = 20
        grid = [[' ' for _ in range(resolution)] for _ in range(resolution)]
        
        # 绘制障碍物
        for obs in self.obstacles:
            x_min_idx = int(obs.x_min * resolution)
            x_max_idx = int(np.ceil(obs.x_max * resolution))
            y_min_idx = int(obs.y_min * resolution)
            y_max_idx = int(np.ceil(obs.y_max * resolution))
            
            for i in range(x_min_idx, min(x_max_idx + 1, resolution)):
                for j in range(y_min_idx, min(y_max_idx + 1, resolution)):
                    grid[i][j] = '#'
        
        # 绘制起点
        sx, sy = int(self.start[0] * resolution), int(self.start[1] * resolution)
        sx, sy = np.clip(sx, 0, resolution - 1), np.clip(sy, 0, resolution - 1)
        if grid[sx][sy] == ' ':
            grid[sx][sy] = 'S'
        
        # 绘制终点
        gx, gy = int(self.goal[0] * resolution), int(self.goal[1] * resolution)
        gx, gy = np.clip(gx, 0, resolution - 1), np.clip(gy, 0, resolution - 1)
        if grid[gx][gy] == ' ':
            grid[gx][gy] = 'G'
        
        # 绘制 agent
        ax, ay = int(self.agent_pos[0] * resolution), int(self.agent_pos[1] * resolution)
        ax, ay = np.clip(ax, 0, resolution - 1), np.clip(ay, 0, resolution - 1)
        if grid[ax][ay] not in ['S', 'G', '#']:
            grid[ax][ay] = 'A'
        
        print("\n".join(["".join(row) for row in grid]))
        print(f"Agent: ({self.agent_pos[0]:.2f}, {self.agent_pos[1]:.2f})")
        print()
