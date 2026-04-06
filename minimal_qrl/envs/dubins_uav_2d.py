"""
2D Dubins UAV 导航环境

实现一个简洁、清晰的 Gym-like 2D Dubins 飞行器导航环境。
状态空间: (x, y, theta) - 位置和朝向角
动作空间: 角速度 u ∈ [-omega_max, omega_max]
动力学: Dubins 模型，固定前进速度 v
"""
import numpy as np
import gym
from gym import spaces
from typing import Tuple, Optional, List, Union
from dataclasses import dataclass

from .base import BaseNavigationEnv


@dataclass
class CircleObstacle:
    """圆形障碍物：(x, y, radius)"""
    x: float
    y: float
    radius: float

    def contains(self, x: float, y: float) -> bool:
        """检查点是否在圆内（含边界）"""
        return (x - self.x) ** 2 + (y - self.y) ** 2 <= (self.radius ** 2)

    def intersects_segment(self, x1: float, y1: float, x2: float, y2: float) -> bool:
        """检查线段是否与圆相交"""
        if self.contains(x1, y1) or self.contains(x2, y2):
            return True
        vx, vy = x2 - x1, y2 - y1
        wx, wy = self.x - x1, self.y - y1
        c2 = vx * vx + vy * vy
        if c2 <= 1e-12:
            return (wx * wx + wy * wy) <= (self.radius ** 2)
        t = np.clip((vx * wx + vy * wy) / c2, 0.0, 1.0)
        px = x1 + t * vx
        py = y1 + t * vy
        return (self.x - px) ** 2 + (self.y - py) ** 2 <= (self.radius ** 2)


@dataclass
class Obstacle:
    """轴对齐矩形障碍物"""
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def contains(self, x: float, y: float) -> bool:
        """检查点是否在障碍物内（包含边界）"""
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max
    
    def intersects_segment(self, x1: float, y1: float, x2: float, y2: float) -> bool:
        """检查线段是否与障碍物相交"""
        # 如果任一端点在矩形内，则相交
        if self.contains(x1, y1) or self.contains(x2, y2):
            return True
        
        # 检查线段是否与矩形的四条边相交
        # 检查是否与 x=x_min 或 x=x_max 相交
        if min(x1, x2) <= self.x_min <= max(x1, x2):
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


class DubinsUAV2D(BaseNavigationEnv):
    """
    2D Dubins UAV 导航环境
    
    状态空间: (x, y, theta)
        - x, y: 二维连续位置，范围由 bounds 指定
        - theta: 朝向角（弧度），范围 [-pi, pi]
    
    动作空间: 角速度 u ∈ [-omega_max, omega_max]
    
    动力学（离散时间）:
        theta_{t+1} = theta_t + u * dt
        x_{t+1} = x_t + v * cos(theta_{t+1}) * dt
        y_{t+1} = y_t + v * sin(theta_{t+1}) * dt
    
    其中:
        - v: 固定前进速度（常数）
        - dt: 时间步长
        - u: 角速度控制输入
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}
    
    def __init__(
        self,
        bounds: Tuple[float, float, float, float] = (0.0, 0.0, 10.0, 10.0),
        omega_max: float = 1.0,
        v: float = 1.0,
        dt: float = 0.1,
        max_episode_steps: int = 200,
        epsilon_pos: float = 0.1,
        epsilon_theta: float = 0.2,  # 弧度，约 11.5 度
        obstacles: Optional[List[Union[Obstacle, CircleObstacle]]] = None,  # noqa: E501
        collision_penalty: float = -10.0,
        start: Optional[Tuple[float, float, float]] = None,
        goal: Optional[Tuple[float, float, float]] = None,
        render_mode: Optional[str] = None,
        use_cos_sin_obs: bool = False,
    ):
        """
        初始化环境
        
        Args:
            bounds: (x_min, y_min, x_max, y_max) 地图边界
            omega_max: 最大角速度（弧度/秒）
            v: 固定前进速度（单位/秒）
            dt: 时间步长（秒）
            max_episode_steps: 每个 episode 的最大步数
            epsilon_pos: 位置到达目标的容差（欧几里得距离）
            epsilon_theta: 朝向到达目标的容差（弧度）
            obstacles: 障碍物列表，如果为 None 则无障碍物
            collision_penalty: 碰撞时的负奖励
            start: 初始状态 (x, y, theta)，如果为 None 则随机采样
            goal: 目标状态 (x, y, theta)，如果为 None 则随机采样
            render_mode: 渲染模式
            use_cos_sin_obs: 若 True，观察为 (x, y, cos(theta), sin(theta))，便于网络学习
        """
        super().__init__()
        
        self.bounds = bounds
        x_min, y_min, x_max, y_max = bounds
        self.x_min, self.y_min, self.x_max, self.y_max = x_min, y_min, x_max, y_max
        
        self.omega_max = omega_max
        self.v = v
        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.epsilon_pos = epsilon_pos
        self.epsilon_theta = epsilon_theta
        self.obstacles = obstacles if obstacles is not None else []
        self.collision_penalty = collision_penalty
        self.render_mode = render_mode
        self.use_cos_sin_obs = use_cos_sin_obs
        
        # 动作空间: 角速度 u ∈ [-omega_max, omega_max]
        self.action_space = spaces.Box(
            low=-omega_max,
            high=omega_max,
            shape=(1,),
            dtype=np.float32
        )
        
        # 观察空间: (x, y, theta) 或 (x, y, cos(theta), sin(theta))
        if use_cos_sin_obs:
            self.observation_space = spaces.Box(
                low=np.array([x_min, y_min, -1.0, -1.0], dtype=np.float32),
                high=np.array([x_max, y_max, 1.0, 1.0], dtype=np.float32),
                shape=(4,),
                dtype=np.float32
            )
        else:
            self.observation_space = spaces.Box(
                low=np.array([x_min, y_min, -np.pi], dtype=np.float32),
                high=np.array([x_max, y_max, np.pi], dtype=np.float32),
                shape=(3,),
                dtype=np.float32
            )
        
        # 固定起终点配置（若为 None，则 reset 时重新采样）
        self._fixed_start = tuple(start) if start is not None else None
        self._fixed_goal = tuple(goal) if goal is not None else None

        # 当前 episode 的起终点
        self.start = self._fixed_start
        self.goal = self._fixed_goal
        
        # 当前状态
        self.state: np.ndarray = np.zeros(3, dtype=np.float32)
        self._t = 0
    
    def _normalize_angle(self, theta: float) -> float:
        """将角度归一化到 [-pi, pi]"""
        while theta > np.pi:
            theta -= 2 * np.pi
        while theta < -np.pi:
            theta += 2 * np.pi
        return theta
    
    def _is_valid_position(self, x: float, y: float) -> bool:
        """检查位置是否合法（不在障碍物内且在边界内）"""
        # 检查边界
        if not (self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max):
            return False
        
        # 检查是否在障碍物内
        for obs in self.obstacles:
            if obs.contains(x, y):
                return False
        
        return True
    
    def is_valid_state(self, state: np.ndarray) -> bool:
        """
        检查状态是否合法
        
        Args:
            state: 状态数组，形状为 (3,)，包含 (x, y, theta)
        
        Returns:
            是否为合法状态
        """
        x, y = float(state[0]), float(state[1])
        return self._is_valid_position(x, y)
    
    def sample_valid_state(self, seed: Optional[int] = None) -> np.ndarray:
        """
        采样一个合法状态
        
        Args:
            seed: 随机种子
        
        Returns:
            合法状态数组，形状为 (3,)，包含 (x, y, theta)
        """
        if seed is not None:
            np.random.seed(seed)
        
        # 尝试采样，最多尝试 1000 次
        max_attempts = 1000
        for _ in range(max_attempts):
            x = np.random.uniform(self.x_min, self.x_max)
            y = np.random.uniform(self.y_min, self.y_max)
            theta = np.random.uniform(-np.pi, np.pi)
            state = np.array([x, y, theta], dtype=np.float32)
            if self.is_valid_state(state):
                return state
        
        # 如果采样失败，返回中心位置
        center_x = (self.x_min + self.x_max) / 2
        center_y = (self.y_min + self.y_max) / 2
        return np.array([center_x, center_y, 0.0], dtype=np.float32)
    
    def _check_collision(self, x1: float, y1: float, x2: float, y2: float) -> bool:
        """检查从 (x1, y1) 到 (x2, y2) 的移动是否穿越障碍物"""
        if not self.obstacles:
            return False
        
        # 检查线段是否与任何障碍物相交
        for obs in self.obstacles:
            if obs.intersects_segment(x1, y1, x2, y2):
                return True
        return False
    
    def _get_obs(self) -> np.ndarray:
        """获取当前观察：内部状态恒为 (x, y, theta)，观察可为 (x, y, cos θ, sin θ)"""
        return self.state_to_observation(self.state)
    
    def state_to_observation(self, state: np.ndarray) -> np.ndarray:
        """
        将内部状态 (x, y, theta) 转为网络输入观察。
        若 use_cos_sin_obs=True 返回 (x, y, cos θ, sin θ)，否则返回 (x, y, theta)。
        """
        state = np.asarray(state, dtype=np.float32).reshape(3)
        x, y, theta = state[0], state[1], state[2]
        if self.use_cos_sin_obs:
            return np.array([x, y, np.cos(theta), np.sin(theta)], dtype=np.float32)
        return state.copy()
    
    def observation_to_state(self, obs: np.ndarray) -> np.ndarray:
        """
        将观察转回内部状态 (x, y, theta)。
        若 use_cos_sin_obs=True，obs 为 (x, y, cos θ, sin θ)，用 atan2 恢复 theta。
        """
        obs = np.asarray(obs, dtype=np.float32).flatten()
        if self.use_cos_sin_obs and len(obs) >= 4:
            x, y, c, s = obs[0], obs[1], obs[2], obs[3]
            theta = np.arctan2(s, c)
            return np.array([x, y, theta], dtype=np.float32)
        if len(obs) >= 3:
            return np.array([obs[0], obs[1], obs[2]], dtype=np.float32)
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    
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

        options = options or {}
        start_override = options.get("start", self._fixed_start)
        goal_override = options.get("goal", self._fixed_goal)

        # 确定起始状态
        if start_override is not None:
            start_state = np.array(start_override, dtype=np.float32)
            start_state[2] = self._normalize_angle(start_state[2])  # 归一化角度
        else:
            start_state = self.sample_valid_state(seed=seed)

        # 确定目标状态
        if goal_override is not None:
            goal_state = np.array(goal_override, dtype=np.float32)
            goal_state[2] = self._normalize_angle(goal_state[2])  # 归一化角度
        else:
            goal_state = self.sample_valid_state(seed=(seed + 1000) if seed is not None else None)
        
        self.start = tuple(start_state)
        self.goal = tuple(goal_state)
        self.state = start_state.copy()
        self._t = 0
        
        return self._get_obs(), {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        执行一步
        
        Args:
            action: 动作（角速度），形状为 (1,) 或标量
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        # 处理动作格式
        if isinstance(action, (int, float, np.number)):
            omega = float(action)
        else:
            action = np.array(action, dtype=np.float32).flatten()
            if len(action) != 1:
                raise ValueError(f"动作维度应为 1，得到 {len(action)}")
            omega = float(action[0])
        
        # 限制角速度
        omega = np.clip(omega, -self.omega_max, self.omega_max)
        
        # 当前状态
        x, y, theta = self.state[0], self.state[1], self.state[2]
        
        # Dubins 动力学（离散时间）
        # 先更新角度
        theta_new = self._normalize_angle(theta + omega * self.dt)
        
        # 然后根据新角度更新位置
        x_new = x + self.v * np.cos(theta_new) * self.dt
        y_new = y + self.v * np.sin(theta_new) * self.dt
        
        # 检查碰撞
        collision_occurred = False
        if self.obstacles and self._check_collision(x, y, x_new, y_new):
            # 如果穿越障碍物，保持当前位置不变
            x_new, y_new = x, y
            collision_occurred = True
        elif not self._is_valid_position(x_new, y_new):
            # 如果新位置不合法（在障碍物内或越界），保持当前位置不变
            x_new, y_new = x, y
            collision_occurred = True
        
        # 更新状态
        self.state = np.array([x_new, y_new, theta_new], dtype=np.float32)
        self._t += 1
        
        # 检查是否到达目标
        goal_x, goal_y, goal_theta = self.goal
        pos_dist = np.sqrt((x_new - goal_x) ** 2 + (y_new - goal_y) ** 2)
        theta_diff = abs(self._normalize_angle(theta_new - goal_theta))
        
        terminated = (pos_dist < self.epsilon_pos) and (theta_diff < self.epsilon_theta)
        truncated = self._t >= self.max_episode_steps
        
        # 奖励设计：每步 -dt（time-optimal），碰撞时额外惩罚
        reward = -self.dt
        if collision_occurred:
            reward += self.collision_penalty
        
        info = {
            "is_success": terminated,
            "pos_dist": pos_dist,
            "theta_diff": theta_diff,
            "collision": collision_occurred,
        }
        
        return self._get_obs(), float(reward), terminated, truncated, info
    
    def compute_shortest_path_distance(
        self,
        start: Optional[np.ndarray] = None,
        goal: Optional[np.ndarray] = None
    ) -> float:
        """
        计算从起点到终点的最短路径距离（位置欧几里得近似）。
        若需与 time-to-go 一致的度量，请使用 compute_min_time_to_go。
        """
        if start is None:
            start = np.array(self.start, dtype=np.float32)
        else:
            start = np.array(start, dtype=np.float32)
        
        if goal is None:
            goal = np.array(self.goal, dtype=np.float32)
        else:
            goal = np.array(goal, dtype=np.float32)
        
        pos_dist = np.sqrt((start[0] - goal[0]) ** 2 + (start[1] - goal[1]) ** 2)
        return float(pos_dist)
    
    def compute_min_time_to_go(
        self,
        start: Optional[np.ndarray] = None,
        goal: Optional[np.ndarray] = None
    ) -> float:
        """
        近似最小 time-to-go（秒）：在 Dubins 动力学下从 start 到 goal 的估计时间。
        使用下界估计: max(直线距离/v, 角度差/omega_max) 的凸组合近似。
        用于 QRL 学习 d(s,g) ≈ minimum time-to-go 时的 ground truth。
        """
        if start is None:
            start = np.array(self.start, dtype=np.float32)
        else:
            start = np.array(start, dtype=np.float32).reshape(3)
        
        if goal is None:
            goal = np.array(self.goal, dtype=np.float32)
        else:
            goal = np.array(goal, dtype=np.float32).reshape(3)
        
        pos_dist = np.sqrt((start[0] - goal[0]) ** 2 + (start[1] - goal[1]) ** 2)
        theta_diff = abs(self._normalize_angle(goal[2] - start[2]))
        time_pos = pos_dist / (self.v + 1e-8)
        time_angle = theta_diff / (self.omega_max + 1e-8)
        # 保守上界：先转向再直线，或同时进行的下界
        return float(time_pos + time_angle)
    
    def get_distance_scale(self) -> float:
        """
        将 QRL 学到的「步长单位」距离转为时间的缩放因子。
        当 local constraint 使用 step_cost=1.0（一步=1）时，预测距离 * dt = 时间（秒）。
        """
        return float(self.dt)
    
    def get_state(self) -> dict:
        """
        获取环境内部状态快照（用于 planning / lookahead 仿真）
        """
        return {
            "state": self.state.copy().tolist(),
            "start": list(self.start) if self.start else None,
            "goal": list(self.goal) if self.goal else None,
            "t": int(self._t),
        }
    
    def set_state(self, state: dict) -> None:
        """
        恢复 get_state() 返回的内部状态快照
        """
        self.state = np.array(state["state"], dtype=np.float32)
        if state["start"] is not None:
            self.start = tuple(state["start"])
        if state["goal"] is not None:
            self.goal = tuple(state["goal"])
        self._t = int(state["t"])
    
    def render(self):
        """渲染环境（文本模式）"""
        if self.render_mode != "human":
            return
        
        # 简单的文本表示
        print(f"Step {self._t}:")
        print(f"  State: x={self.state[0]:.2f}, y={self.state[1]:.2f}, theta={self.state[2]:.3f}")
        if self.goal:
            goal_x, goal_y, goal_theta = self.goal
            pos_dist = np.sqrt((self.state[0] - goal_x) ** 2 + (self.state[1] - goal_y) ** 2)
            theta_diff = abs(self._normalize_angle(self.state[2] - goal_theta))
            print(f"  Goal: x={goal_x:.2f}, y={goal_y:.2f}, theta={goal_theta:.3f}")
            print(f"  Distance: pos={pos_dist:.2f}, theta={theta_diff:.3f}")
        print()
