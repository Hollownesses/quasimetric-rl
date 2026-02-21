"""
Planning / Reachability 评估模块
专门面向 obstacle navigation 场景的评估功能：
1. Greedy Navigation Success Rate
2. Path Efficiency (与最短路径的竞争比)
3. Failure Mode 可视化
"""
import os
from typing import Tuple, Dict, Optional, List, Literal
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import gym
from minimal_qrl.envs.base import BaseNavigationEnv


ExecutionMode = Literal["greedy", "lookahead"]
DistanceType = Literal["qrl", "euclidean"]


def _unwrap_env(env: gym.Env) -> gym.Env:
    return env.unwrapped if hasattr(env, "unwrapped") else env


def _as_nav_env(env: gym.Env) -> Optional[BaseNavigationEnv]:
    u = _unwrap_env(env)
    return u if isinstance(u, BaseNavigationEnv) else None


def _capture_env_state(env: gym.Env) -> dict:
    """
    捕获环境状态，用于 lookahead 多分支仿真后恢复。
    优先使用 BaseNavigationEnv.get_state / set_state。
    """
    u = _unwrap_env(env)
    state: dict = {}

    if hasattr(u, "get_state") and callable(getattr(u, "get_state")):
        try:
            state["unwrapped"] = u.get_state()
        except NotImplementedError:
            state["unwrapped"] = None
    else:
        state["unwrapped"] = None

    # 兼容可能存在的 TimeLimit / wrapper 计步器
    if hasattr(env, "_elapsed_steps"):
        state["_elapsed_steps"] = int(getattr(env, "_elapsed_steps"))

    return state


def _restore_env_state(env: gym.Env, state: dict) -> None:
    u = _unwrap_env(env)

    if state.get("unwrapped") is not None and hasattr(u, "set_state") and callable(getattr(u, "set_state")):
        u.set_state(state["unwrapped"])

    if "_elapsed_steps" in state and hasattr(env, "_elapsed_steps"):
        setattr(env, "_elapsed_steps", int(state["_elapsed_steps"]))


def _get_action_bounds(env: gym.Env) -> Tuple[np.ndarray, np.ndarray]:
    # 获取动作空间范围
    if hasattr(env, "action_space"):
        action_space = env.action_space
        if hasattr(action_space, "low") and hasattr(action_space, "high"):
            low = np.array(action_space.low, dtype=np.float32).reshape(-1)
            high = np.array(action_space.high, dtype=np.float32).reshape(-1)
            return low, high

    # 默认范围（与原 greedy 保持一致）
    low = np.array([-0.1, -0.1], dtype=np.float32)
    high = np.array([0.1, 0.1], dtype=np.float32)
    return low, high


def compute_qrl_distance_batch(
    agent: nn.Module,
    states: np.ndarray,
    goal: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """
    批量计算 states -> goal 的 QRL distance。
    """
    states = np.asarray(states, dtype=np.float32)
    goal = np.asarray(goal, dtype=np.float32).reshape(2)

    critic = agent.critics[0]
    device_obj = torch.device(device)

    states_t = torch.tensor(states, device=device_obj, dtype=torch.float32)
    goal_t = torch.tensor(goal[None].repeat(len(states), 0), device=device_obj, dtype=torch.float32)

    with torch.no_grad():
        zx = critic.encoder(states_t)
        zy = critic.encoder(goal_t)
        dists = critic.quasimetric_model(zx, zy).cpu().numpy()

    return dists.astype(np.float32).reshape(-1)


@dataclass
class LookaheadConfig:
    """
    QRL-guided lookahead / local planning 的超参数（仅用于 evaluation / rollout）。
    """
    horizon: int = 5
    num_sequences: int = 64
    step_cost_weight: float = 0.0
    collision_penalty: float = 0.0
    biased_sequences: int = 12  # 添加一部分朝向 goal 的序列以增强稳定性
    # 终端代价使用的 distance 类型：
    # - "qrl": 使用 QRL learned distance（默认，保持向后兼容）
    # - "euclidean": 使用欧几里得距离 ||s_T - goal|| 作为启发式
    distance_type: DistanceType = "qrl"


def compute_ground_truth_distance(
    env: gym.Env,
    start: np.ndarray,
    goal: np.ndarray
) -> float:
    """
    计算两个状态之间的真实最短路径距离
    
    如果环境实现了 BaseNavigationEnv 接口，使用其 compute_shortest_path_distance 方法
    否则使用欧几里得距离作为默认值
    
    Args:
        env: 环境实例
        start: 起始状态，归一化坐标
        goal: 目标状态，归一化坐标
    
    Returns:
        真实最短路径距离
    """
    nav_env = _as_nav_env(env)
    if nav_env is not None:
        return nav_env.compute_shortest_path_distance(start=start, goal=goal)
    else:
        # 默认使用欧几里得距离
        return float(np.linalg.norm(start - goal))


def sample_state_goal_pairs(
    env: gym.Env,
    n_pairs: int = 2000,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    随机采样状态-目标对，确保状态合法
    
    Args:
        env: 环境实例（应实现 BaseNavigationEnv 接口）
        n_pairs: 采样对数
        seed: 随机种子
    
    Returns:
        states: (n_pairs, obs_dim) 合法状态
        goals: (n_pairs, obs_dim) 合法目标
    """
    if seed is not None:
        np.random.seed(seed)
    
    states = []
    goals = []
    
    # 如果环境实现了 BaseNavigationEnv 接口，使用其 sample_valid_state 方法
    nav_env = _as_nav_env(env)
    if nav_env is not None:
        for i in range(n_pairs):
            state = nav_env.sample_valid_state(seed=seed + i if seed is not None else None)
            goal = nav_env.sample_valid_state(seed=seed + i + n_pairs if seed is not None else None)
            states.append(state)
            goals.append(goal)
    else:
        # 否则，从观察空间采样
        for i in range(n_pairs):
            state = env.observation_space.sample()
            goal = env.observation_space.sample()
            states.append(state)
            goals.append(goal)
    
    return np.array(states, dtype=np.float32), np.array(goals, dtype=np.float32)


def compute_qrl_distance(
    agent: nn.Module,
    state: np.ndarray,
    goal: np.ndarray,
    device: str = 'cpu'
) -> float:
    """
    计算从 state 到 goal 的 QRL distance
    
    Args:
        agent: QRL Agent（包含 critics）
        state: 当前状态，形状为 (2,)
        goal: 目标状态，形状为 (2,)
        device: 设备
    
    Returns:
        QRL distance（标量）
    """
    critic = agent.critics[0]
    device_obj = torch.device(device)
    
    state_t = torch.tensor(state[None], device=device_obj, dtype=torch.float32)
    goal_t = torch.tensor(goal[None], device=device_obj, dtype=torch.float32)
    
    with torch.no_grad():
        zx = critic.encoder(state_t)
        zy = critic.encoder(goal_t)
        dist = critic.quasimetric_model(zx, zy).cpu().item()
    
    return float(dist)


def greedy_action_selection(
    agent: nn.Module,
    env: gym.Env,
    current_state: np.ndarray,
    goal: np.ndarray,
    device: str = 'cpu',
    num_candidates: int = 32,
    obstacle_penalty_weight: float = 0.0  # 默认禁用障碍物惩罚，让QRL模型自己学习
) -> np.ndarray:
    """
    使用 QRL distance 进行 greedy action selection
    
    在连续动作空间中，采样多个候选动作，选择能最小化下一状态到 goal 的 QRL distance 的动作
    
    Args:
        agent: QRL Agent
        env: 环境实例
        current_state: 当前状态，形状为 (2,)
        goal: 目标状态，形状为 (2,)
        device: 设备
        num_candidates: 候选动作数量
        obstacle_penalty_weight: 障碍物惩罚权重（越大，越避免接近障碍物）
    
    Returns:
        选择的动作，形状为 (2,)
    """
    low, high = _get_action_bounds(env)
    
    # 采样候选动作（包括朝向 goal 的方向）
    candidates = []
    
    # 1. 随机采样
    random_candidates = np.random.uniform(
        low=low,
        high=high,
        size=(num_candidates - 4, 2)
    ).astype(np.float32)
    candidates.extend(random_candidates)
    
    # 2. 添加朝向 goal 的方向（归一化并限制在动作空间内）
    direction = goal - current_state
    direction_norm = np.linalg.norm(direction)
    if direction_norm > 1e-6:
        direction_normalized = direction / direction_norm
        # 限制步长
        max_step = min(abs(high[0]), abs(high[1])) if len(high) >= 2 else 0.1
        candidates.append(direction_normalized * max_step)
        candidates.append(direction_normalized * max_step * 0.5)
        candidates.append(direction_normalized * max_step * 0.25)
    else:
        candidates.append(np.array([0.0, 0.0], dtype=np.float32))
    
    # 3. 添加零动作
    candidates.append(np.array([0.0, 0.0], dtype=np.float32))
    
    candidates = np.array(candidates, dtype=np.float32)
    
    # 对每个候选动作，计算执行后的下一状态到 goal 的 QRL distance
    best_action = None
    best_score = float('inf')
    
    # 批量计算距离以提高效率
    next_states = []
    valid_actions = []
    
    # 计算障碍物惩罚函数（默认禁用，让QRL模型自己学习）
    def compute_obstacle_penalty(state: np.ndarray) -> float:
        """
        计算状态到最近障碍物的惩罚（距离越近，惩罚越大）
        
        注意：默认禁用（obstacle_penalty_weight=0.0），让QRL模型自己学习障碍物信息
        如果QRL模型没有很好地学习到障碍物信息，可以启用此惩罚（设置obstacle_penalty_weight > 0）
        """
        if obstacle_penalty_weight <= 0.0:
            return 0.0  # 禁用障碍物惩罚
        
        nav_env = _as_nav_env(env)
        if nav_env is None or not hasattr(nav_env, 'obstacles'):
            return 0.0
        
        x, y = state[0], state[1]
        min_dist_to_obstacle = float('inf')
        
        for obs in nav_env.obstacles:
            # 计算点到矩形障碍物的最短距离
            # 如果点在矩形内，距离为0（应该被is_valid_state过滤）
            # 否则计算到矩形边界的最短距离
            
            # 计算到矩形各边的距离
            if x < obs.x_min:
                dx = obs.x_min - x
            elif x > obs.x_max:
                dx = x - obs.x_max
            else:
                dx = 0.0
            
            if y < obs.y_min:
                dy = obs.y_min - y
            elif y > obs.y_max:
                dy = y - obs.y_max
            else:
                dy = 0.0
            
            # 如果dx和dy都为0，点在障碍物内（这种情况应该被过滤）
            if dx == 0.0 and dy == 0.0:
                return float('inf')  # 极大惩罚
            
            dist = np.sqrt(dx ** 2 + dy ** 2)
            min_dist_to_obstacle = min(min_dist_to_obstacle, dist)
        
        # 只在非常接近障碍物时才给予惩罚（距离 < 0.02）
        if min_dist_to_obstacle < 0.02:
            # 使用线性衰减
            normalized_dist = min_dist_to_obstacle / 0.02
            penalty = obstacle_penalty_weight * 0.1 * (1.0 - normalized_dist)
        else:
            penalty = 0.0
        
        return penalty
    
    for action in candidates:
        # 计算执行动作后的下一状态（简化：直接相加，环境会在 step 时处理碰撞）
        next_state = current_state + action
        
        # 限制在边界内
        next_state = np.clip(next_state, 0.0, 1.0)
        
        # 检查是否合法（对于障碍物环境）
        nav_env = _as_nav_env(env)
        if nav_env is not None:
            if not nav_env.is_valid_state(next_state):
                # 如果状态不合法，跳过
                continue
        
        next_states.append(next_state)
        valid_actions.append(action)
    
    if not valid_actions:
        # 如果没有找到合法动作，返回朝向 goal 的方向
        direction = goal - current_state
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 1e-6:
            direction = direction / direction_norm
            max_step = min(abs(high[0]), abs(high[1])) if len(high) >= 2 else 0.1
            return direction * max_step
        else:
            return np.array([0.0, 0.0], dtype=np.float32)
    
    # 批量计算 QRL distance
    next_states = np.array(next_states, dtype=np.float32)
    valid_actions = np.array(valid_actions, dtype=np.float32)
    dists = compute_qrl_distance_batch(agent, next_states, goal, device=device)
    
    # 计算每个动作的综合得分：QRL距离 + 障碍物惩罚
    scores = []
    for i, next_state in enumerate(next_states):
        qrl_dist = dists[i]
        obstacle_penalty = compute_obstacle_penalty(next_state)
        total_score = qrl_dist + obstacle_penalty
        scores.append(total_score)
    
    scores = np.array(scores)
    
    # 选择最小得分对应的动作
    best_idx = np.argmin(scores)
    best_action = valid_actions[best_idx]
    
    return best_action


def lookahead_action_selection(
    agent: nn.Module,
    env: gym.Env,
    current_state: np.ndarray,
    goal: np.ndarray,
    *,
    device: str = "cpu",
    config: Optional[LookaheadConfig] = None,
) -> np.ndarray:
    """
    QRL-guided lookahead / local planning：
    - 使用环境 step 进行短视野多步仿真（shooting）
    - 用 QRL distance 作为终端代价（可叠加步长/碰撞惩罚）
    - 返回最优动作序列的第一个动作（receding horizon）
    """
    if config is None:
        config = LookaheadConfig()

    horizon = int(max(1, config.horizon))
    num_sequences = int(max(1, config.num_sequences))
    low, high = _get_action_bounds(env)

    current_state = np.asarray(current_state, dtype=np.float32).reshape(2)
    goal = np.asarray(goal, dtype=np.float32).reshape(2)

    # 采样 action sequences
    seqs = np.random.uniform(low=low, high=high, size=(num_sequences, horizon, 2)).astype(np.float32)

    # 加入少量“朝向 goal”的偏置序列，提升稳定性
    direction = goal - current_state
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm > 1e-6:
        unit = direction / direction_norm
        max_step = float(min(abs(high[0]), abs(high[1]))) if len(high) >= 2 else 0.1
        base = (unit * max_step).astype(np.float32)
        n_bias = int(min(max(0, config.biased_sequences), num_sequences))
        scales = [1.0, 0.75, 0.5, 0.35, 0.25, 0.15]
        for i in range(n_bias):
            s = scales[i % len(scales)]
            seqs[i, 0] = base * float(s)
            if horizon > 1:
                # 后续动作给较小随机扰动，避免过于僵硬
                seqs[i, 1:] = np.random.uniform(low=low, high=high, size=(horizon - 1, 2)).astype(np.float32) * 0.25
    else:
        # 极少数情况下 start==goal，加入零动作序列
        seqs[0, :, :] = 0.0

    # 仿真并打分（先累积非终端部分，终端 QRL distance 批量算）
    base_env_state = _capture_env_state(env)
    success_mask = np.zeros((num_sequences,), dtype=bool)
    collision_counts = np.zeros((num_sequences,), dtype=np.int32)
    step_costs = np.zeros((num_sequences,), dtype=np.float32)
    terminal_states = np.zeros((num_sequences, 2), dtype=np.float32)

    nav_env = _as_nav_env(env)

    for i in range(num_sequences):
        _restore_env_state(env, base_env_state)

        obs = None
        terminated = False
        truncated = False

        for t in range(horizon):
            a = seqs[i, t]
            obs, reward, terminated, truncated, _info = env.step(a)

            # 代价：可选步长惩罚（抑制抖动/绕圈）
            if config.step_cost_weight > 0.0:
                step_costs[i] += float(config.step_cost_weight) * float(np.linalg.norm(a))

            # 代价：可选碰撞惩罚（ContinuousObstacle2D 碰撞奖励为 -0.1）
            if config.collision_penalty > 0.0:
                if float(reward) <= -0.05:
                    collision_counts[i] += 1

            if terminated:
                success_mask[i] = True
                break

            if truncated:
                break

        # 记录终端状态（如果 horizon 内未赋值 obs，退化为当前状态）
        if obs is None:
            terminal_states[i] = current_state
        else:
            terminal_states[i] = np.asarray(obs, dtype=np.float32).reshape(2)

        # 基于环境可行性做一次兜底（防止异常）
        if nav_env is not None and not nav_env.is_valid_state(terminal_states[i]):
            collision_counts[i] += 5

    # 恢复环境状态（外部 rollout 继续使用原始状态）
    _restore_env_state(env, base_env_state)

    # 终端代价：根据 distance_type 选择 QRL 或欧几里得距离
    distance_type: DistanceType = getattr(config, "distance_type", "qrl")  # 向后兼容
    if distance_type == "qrl":
        terminal_costs = compute_qrl_distance_batch(agent, terminal_states, goal, device=device)
        terminal_costs = terminal_costs.astype(np.float32)
    elif distance_type == "euclidean":
        diffs = terminal_states - goal[None, :].astype(np.float32)
        terminal_costs = np.linalg.norm(diffs, axis=1).astype(np.float32)
    else:
        raise ValueError(f"未知 distance_type: {distance_type}")

    # 成功序列的终端代价视为 0
    terminal_costs[success_mask] = 0.0

    costs = terminal_costs + step_costs + (collision_counts.astype(np.float32) * float(config.collision_penalty))
    best_idx = int(np.argmin(costs))
    best_action = seqs[best_idx, 0].astype(np.float32)
    return best_action


def navigation_rollout(
    agent: nn.Module,
    env: gym.Env,
    start: np.ndarray,
    goal: np.ndarray,
    device: str = 'cpu',
    max_steps: int = 200,
    num_action_candidates: int = 32,
    use_improved_termination: bool = True,
    execution_mode: ExecutionMode = "greedy",
    lookahead_config: Optional[LookaheadConfig] = None,
) -> Dict:
    """
    执行一次 navigation rollout（支持 greedy / lookahead 两种执行机制）
    
    Args:
        agent: QRL Agent
        env: 环境实例
        start: 起始状态，形状为 (2,)
        goal: 目标状态，形状为 (2,)
        device: 设备
        max_steps: 最大步数
        num_action_candidates: 每步候选动作数量
        use_improved_termination: 是否使用改进的终止条件（结合QRL距离和欧几里得距离）
    
    Returns:
        包含以下键的字典：
        - success: 是否成功到达目标
        - num_steps: 实际步数
        - path_length: 轨迹总长度
        - trajectory: 轨迹列表，每个元素是 (state, action)
        - final_state: 最终状态
    """
    # 重置环境
    obs, _ = env.reset(seed=None, options={'start': tuple(start), 'goal': tuple(goal)})
    current_state = obs.copy()
    
    trajectory = []
    path_length = 0.0
    success = False
    
    for step in range(max_steps):
        # 改进的终止条件：结合欧几里得距离和QRL距离
        euclidean_dist = np.linalg.norm(current_state - goal)
        
        if use_improved_termination:
            # 计算QRL距离
            qrl_dist = compute_qrl_distance(agent, current_state, goal, device)
            
            # 改进的终止条件（放宽条件，避免因QRL距离高估导致无法终止）：
            # 1. 欧几里得距离必须足够小（基本要求）
            # 2. QRL距离也应该较小，但阈值放宽（因为QRL距离可能被高估）
            # 3. 两者应该相对一致（但允许QRL距离有一定的高估）
            euclidean_threshold = 0.05  # 与环境的容差一致
            qrl_threshold = 1.0  # QRL距离阈值放宽（从0.1提高到1.0，允许QRL距离被高估）
            
            # 改进的终止条件：主要依赖欧几里得距离，QRL距离作为辅助验证
            # 如果QRL距离远大于欧几里得距离（比如20倍以上），说明QRL估计严重不准确，不应该终止
            # 但如果欧几里得距离足够小，即使QRL距离被高估，也应该允许终止（因为这是真实距离）
            if euclidean_dist < euclidean_threshold:
                # 如果QRL距离不是异常大（小于阈值），或者QRL距离虽然大但不超过欧几里得距离的合理倍数
                if qrl_dist < qrl_threshold or qrl_dist < euclidean_dist * 50.0:
                    success = True
                    break
        else:
            # 原始终止条件（仅使用欧几里得距离）
            if euclidean_dist < 0.05:
                success = True
                break
        
        # 选择动作（execution mechanism）
        if execution_mode == "greedy":
            action = greedy_action_selection(
                agent, env, current_state, goal, device, num_action_candidates
            )
        elif execution_mode == "lookahead":
            action = lookahead_action_selection(
                agent, env, current_state, goal, device=device, config=lookahead_config
            )
        else:
            raise ValueError(f"未知 execution_mode: {execution_mode}")
        
        # 执行动作
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_state = next_obs.copy()
        
        # 记录轨迹
        trajectory.append((current_state.copy(), action.copy()))
        
        # 计算路径长度
        step_length = np.linalg.norm(next_state - current_state)
        path_length += step_length
        
        # 更新状态
        current_state = next_state
        
        if terminated or truncated:
            if terminated:
                success = True
            break
    
    return {
        'success': success,
        'num_steps': len(trajectory),
        'path_length': path_length,
        'trajectory': trajectory,
        'final_state': current_state.copy()
    }


def greedy_navigation_rollout(
    agent: nn.Module,
    env: gym.Env,
    start: np.ndarray,
    goal: np.ndarray,
    device: str = 'cpu',
    max_steps: int = 200,
    num_action_candidates: int = 32,
    use_improved_termination: bool = True
) -> Dict:
    """
    兼容旧接口：greedy navigation rollout（内部调用 navigation_rollout）。
    """
    return navigation_rollout(
        agent=agent,
        env=env,
        start=start,
        goal=goal,
        device=device,
        max_steps=max_steps,
        num_action_candidates=num_action_candidates,
        use_improved_termination=use_improved_termination,
        execution_mode="greedy",
        lookahead_config=None,
    )


def evaluate_navigation_success_rate(
    agent: nn.Module,
    env: gym.Env,
    n_trials: int = 100,
    device: str = 'cpu',
    seed: Optional[int] = None,
    num_action_candidates: int = 32,
    execution_mode: ExecutionMode = "greedy",
    lookahead_config: Optional[LookaheadConfig] = None,
    starts: Optional[np.ndarray] = None,
    goals: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    评估 Navigation Success Rate（greedy / lookahead）
    
    Args:
        agent: QRL Agent
        env: 环境实例
        n_trials: 测试次数
        device: 设备
        seed: 随机种子
        num_action_candidates: 每步候选动作数量
    
    Returns:
        评估指标字典，包含：
        - success_rate: 成功率
        - avg_steps: 平均步数（仅成功案例）
        - avg_path_length: 平均路径长度（仅成功案例）
        - all_steps: 所有案例的平均步数
        - all_path_length: 所有案例的平均路径长度
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 采样起点和目标对（允许外部传入，保证不同 mode 公平对比）
    if starts is None or goals is None:
        starts, goals = sample_state_goal_pairs(env, n_pairs=n_trials, seed=seed)
    
    success_count = 0
    success_steps = []
    success_path_lengths = []
    all_steps = []
    all_path_lengths = []
    
    for i in range(n_trials):
        rollout_result = navigation_rollout(
            agent, env, starts[i], goals[i], device,
            max_steps=env.max_episode_steps if hasattr(env, 'max_episode_steps') else 200,
            num_action_candidates=num_action_candidates,
            use_improved_termination=True,
            execution_mode=execution_mode,
            lookahead_config=lookahead_config,
        )
        
        all_steps.append(rollout_result['num_steps'])
        all_path_lengths.append(rollout_result['path_length'])
        
        if rollout_result['success']:
            success_count += 1
            success_steps.append(rollout_result['num_steps'])
            success_path_lengths.append(rollout_result['path_length'])
    
    success_rate = success_count / n_trials if n_trials > 0 else 0.0
    
    return {
        'success_rate': success_rate,
        'avg_steps': np.mean(success_steps) if success_steps else 0.0,
        'avg_path_length': np.mean(success_path_lengths) if success_path_lengths else 0.0,
        'std_steps': np.std(success_steps) if success_steps else 0.0,
        'std_path_length': np.std(success_path_lengths) if success_path_lengths else 0.0,
        'all_avg_steps': np.mean(all_steps) if all_steps else 0.0,
        'all_avg_path_length': np.mean(all_path_lengths) if all_path_lengths else 0.0,
        'num_success': success_count,
        'num_trials': n_trials,
    }


def evaluate_greedy_navigation_success_rate(
    agent: nn.Module,
    env: gym.Env,
    n_trials: int = 100,
    device: str = 'cpu',
    seed: Optional[int] = None,
    num_action_candidates: int = 32
) -> Dict[str, float]:
    """
    兼容旧接口：评估 Greedy Navigation Success Rate
    """
    return evaluate_navigation_success_rate(
        agent=agent,
        env=env,
        n_trials=n_trials,
        device=device,
        seed=seed,
        num_action_candidates=num_action_candidates,
        execution_mode="greedy",
        lookahead_config=None,
    )


def evaluate_path_efficiency(
    agent: nn.Module,
    env: gym.Env,
    n_trials: int = 100,
    device: str = 'cpu',
    seed: Optional[int] = None,
    num_action_candidates: int = 32,
    execution_mode: ExecutionMode = "greedy",
    lookahead_config: Optional[LookaheadConfig] = None,
    starts: Optional[np.ndarray] = None,
    goals: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    评估 Path Efficiency（与最短路径的竞争比）
    
    对成功到达目标的 rollout，计算实际轨迹长度与最短路径长度之比
    
    Args:
        agent: QRL Agent
        env: 环境实例
        n_trials: 测试次数
        device: 设备
        seed: 随机种子
        num_action_candidates: 每步候选动作数量
    
    Returns:
        评估指标字典，包含：
        - avg_efficiency_ratio: 平均效率比（实际路径长度 / 最短路径长度）
        - median_efficiency_ratio: 中位数效率比
        - std_efficiency_ratio: 效率比标准差
        - min_efficiency_ratio: 最小效率比
        - max_efficiency_ratio: 最大效率比
        - num_success: 成功案例数
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 采样起点和目标对（允许外部传入，保证不同 mode 公平对比）
    if starts is None or goals is None:
        starts, goals = sample_state_goal_pairs(env, n_pairs=n_trials, seed=seed)
    
    efficiency_ratios = []
    
    for i in range(n_trials):
        rollout_result = navigation_rollout(
            agent, env, starts[i], goals[i], device,
            max_steps=env.max_episode_steps if hasattr(env, 'max_episode_steps') else 200,
            num_action_candidates=num_action_candidates,
            use_improved_termination=True,
            execution_mode=execution_mode,
            lookahead_config=lookahead_config,
        )
        
        if rollout_result['success']:
            # 计算最短路径距离
            shortest_dist = compute_ground_truth_distance(env, starts[i], goals[i])
            
            if shortest_dist > 1e-6:  # 避免除零
                path_length = rollout_result['path_length']
                
                # 验证是否真正到达目标：检查最终状态到目标的距离
                final_state = rollout_result['final_state']
                final_dist_to_goal = np.linalg.norm(final_state - goals[i])
                
                # 只有真正到达目标（距离 < 0.05）的路径才计算效率
                # 这样可以避免误判为成功但实际未到达的情况
                if final_dist_to_goal < 0.05:
                    efficiency_ratio = path_length / shortest_dist
                    
                    # 确保efficiency_ratio >= 1（实际路径长度应该 >= 最短路径长度）
                    # 如果出现 < 1，可能是路径长度计算有问题（比如路径很短但被误判为成功）
                    if efficiency_ratio < 1.0:
                        print(f"Warning: Efficiency ratio < 1.0 for trial {i}: path_length={path_length:.4f}, "
                              f"shortest_dist={shortest_dist:.4f}, final_dist_to_goal={final_dist_to_goal:.4f}")
                        # 如果路径长度异常小，可能是计算错误，跳过这个样本
                        if path_length < shortest_dist * 0.5:  # 如果路径长度小于最短路径的一半，可能是错误
                            continue
                        # 否则，使用最短路径长度作为最小值
                        efficiency_ratio = max(1.0, efficiency_ratio)
                    
                    efficiency_ratios.append(efficiency_ratio)
                else:
                    # 虽然标记为success，但实际未到达目标，跳过
                    print(f"Warning: Trial {i} marked as success but final_dist_to_goal={final_dist_to_goal:.4f} > 0.05, skipping")
    
    if not efficiency_ratios:
        return {
            'avg_efficiency_ratio': 0.0,
            'median_efficiency_ratio': 0.0,
            'std_efficiency_ratio': 0.0,
            'min_efficiency_ratio': 0.0,
            'max_efficiency_ratio': 0.0,
            'num_success': 0,
            'num_trials': n_trials,
        }
    
    efficiency_ratios = np.array(efficiency_ratios)
    
    return {
        'avg_efficiency_ratio': float(np.mean(efficiency_ratios)),
        'median_efficiency_ratio': float(np.median(efficiency_ratios)),
        'std_efficiency_ratio': float(np.std(efficiency_ratios)),
        'min_efficiency_ratio': float(np.min(efficiency_ratios)),
        'max_efficiency_ratio': float(np.max(efficiency_ratios)),
        'num_success': len(efficiency_ratios),
        'num_trials': n_trials,
    }


def visualize_failure_modes(
    agent: nn.Module,
    env: gym.Env,
    n_failures: int = 10,
    device: str = 'cpu',
    seed: Optional[int] = None,
    output_dir: str = './results',
    step: int = 0,
    num_action_candidates: int = 32,
    resolution: Tuple[int, int] = (50, 50),
    execution_mode: ExecutionMode = "greedy",
    lookahead_config: Optional[LookaheadConfig] = None,
) -> List[str]:
    """
    可视化 Failure Mode
    
    自动收集 rollout 失败的起点，对这些 failure case 可视化：
    - QRL learned distance heatmap
    - shortest-path distance heatmap
    - 实际 rollout 轨迹（叠加在环境上）
    
    Args:
        agent: QRL Agent
        env: 环境实例
        n_failures: 需要收集的失败案例数
        device: 设备
        seed: 随机种子
        output_dir: 输出目录
        step: 训练步数（用于文件名）
        num_action_candidates: 每步候选动作数量
        resolution: 热力图分辨率 (height, width)
    
    Returns:
        保存的图像文件路径列表
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if seed is not None:
        np.random.seed(seed)
    
    # 收集失败案例
    failure_cases = []
    max_attempts = n_failures * 10  # 最多尝试次数
    
    starts, goals = sample_state_goal_pairs(env, n_pairs=max_attempts, seed=seed)
    
    for i in range(max_attempts):
        if len(failure_cases) >= n_failures:
            break
        
        rollout_result = navigation_rollout(
            agent, env, starts[i], goals[i], device,
            max_steps=env.max_episode_steps if hasattr(env, 'max_episode_steps') else 200,
            num_action_candidates=num_action_candidates,
            use_improved_termination=True,
            execution_mode=execution_mode,
            lookahead_config=lookahead_config,
        )
        
        if not rollout_result['success']:
            failure_cases.append({
                'start': starts[i],
                'goal': goals[i],
                'rollout_result': rollout_result
            })
    
    if not failure_cases:
        print(f"警告：未找到失败案例，无法生成可视化")
        return []
    
    # 可视化每个失败案例
    saved_paths = []
    critic = agent.critics[0]
    device_obj = torch.device(device)
    
    for idx, case in enumerate(failure_cases):
        start = case['start']
        goal = case['goal']
        trajectory = case['rollout_result']['trajectory']
        
        # 创建网格用于热力图
        h, w = resolution
        x_coords = np.linspace(0.0, 1.0, h, dtype=np.float32)
        y_coords = np.linspace(0.0, 1.0, w, dtype=np.float32)
        Y, X = np.meshgrid(y_coords, x_coords)
        
        states = np.stack([X.flatten(), Y.flatten()], axis=1).astype(np.float32)
        
        # 计算 QRL distance heatmap
        states_t = torch.tensor(states, device=device_obj, dtype=torch.float32)
        goal_t = torch.tensor(goal[None].repeat(len(states), 0), device=device_obj, dtype=torch.float32)
        
        with torch.no_grad():
            zx = critic.encoder(states_t)
            zy = critic.encoder(goal_t)
            qrl_dists = critic.quasimetric_model(zx, zy).cpu().numpy()
        
        qrl_dists = qrl_dists.reshape(h, w)
        
        # 计算 shortest-path distance heatmap
        gt_dists = np.zeros((h, w), dtype=np.float32)
        for i in range(h):
            for j in range(w):
                s = np.array([x_coords[i], y_coords[j]], dtype=np.float32)
                gt_dists[i, j] = compute_ground_truth_distance(env, s, goal)
        
        # 绘制
        fig = plt.figure(figsize=(18, 6))
        
        # 左：QRL distance heatmap + 轨迹
        ax1 = fig.add_subplot(131)
        im1 = ax1.imshow(qrl_dists, origin='lower', extent=[0, 1, 0, 1],
                        cmap='viridis', aspect='auto', interpolation='nearest', alpha=0.7)
        ax1.set_xlabel('Y position (normalized)')
        ax1.set_ylabel('X position (normalized)')
        ax1.set_title(f'QRL Distance Heatmap\n(Failure Case {idx+1})')
        plt.colorbar(im1, ax=ax1, label='QRL Distance')
        
        # 绘制障碍物（如果是 ContinuousObstacle2D）
        if hasattr(env, 'obstacles'):
            for obs in env.obstacles:
                rect = patches.Rectangle(
                    (obs.y_min, obs.x_min),
                    obs.y_max - obs.y_min,
                    obs.x_max - obs.x_min,
                    linewidth=2, edgecolor='red', facecolor='red', alpha=0.5
                )
                ax1.add_patch(rect)
        
        # 绘制轨迹
        if trajectory:
            traj_states = np.array([s for s, a in trajectory])
            ax1.plot(traj_states[:, 1], traj_states[:, 0], 'r-', linewidth=2, label='Trajectory', alpha=0.8)
            ax1.plot(traj_states[0, 1], traj_states[0, 0], 'go', markersize=10, label='Start')
            ax1.plot(traj_states[-1, 1], traj_states[-1, 0], 'rs', markersize=10, label='End')
        
        ax1.plot(goal[1], goal[0], 'r*', markersize=20, label='Goal')
        ax1.legend()
        
        # 中：Shortest-path distance heatmap + 轨迹
        ax2 = fig.add_subplot(132)
        im2 = ax2.imshow(gt_dists, origin='lower', extent=[0, 1, 0, 1],
                        cmap='viridis', aspect='auto', interpolation='nearest', alpha=0.7)
        ax2.set_xlabel('Y position (normalized)')
        ax2.set_ylabel('X position (normalized)')
        ax2.set_title('Shortest-Path Distance Heatmap')
        plt.colorbar(im2, ax=ax2, label='Shortest-Path Distance')
        
        # 绘制障碍物
        if hasattr(env, 'obstacles'):
            for obs in env.obstacles:
                rect = patches.Rectangle(
                    (obs.y_min, obs.x_min),
                    obs.y_max - obs.y_min,
                    obs.x_max - obs.x_min,
                    linewidth=2, edgecolor='red', facecolor='red', alpha=0.5
                )
                ax2.add_patch(rect)
        
        # 绘制轨迹
        if trajectory:
            traj_states = np.array([s for s, a in trajectory])
            ax2.plot(traj_states[:, 1], traj_states[:, 0], 'r-', linewidth=2, label='Trajectory', alpha=0.8)
            ax2.plot(traj_states[0, 1], traj_states[0, 0], 'go', markersize=10, label='Start')
            ax2.plot(traj_states[-1, 1], traj_states[-1, 0], 'rs', markersize=10, label='End')
        
        ax2.plot(goal[1], goal[0], 'r*', markersize=20, label='Goal')
        ax2.legend()
        
        # 右：误差热力图
        ax3 = fig.add_subplot(133)
        error = np.abs(qrl_dists - gt_dists)
        im3 = ax3.imshow(error, origin='lower', extent=[0, 1, 0, 1],
                        cmap='hot', aspect='auto', interpolation='nearest', alpha=0.7)
        ax3.set_xlabel('Y position (normalized)')
        ax3.set_ylabel('X position (normalized)')
        ax3.set_title('Distance Error (|QRL - Shortest|)')
        plt.colorbar(im3, ax=ax3, label='Absolute Error')
        
        # 绘制障碍物
        if hasattr(env, 'obstacles'):
            for obs in env.obstacles:
                rect = patches.Rectangle(
                    (obs.y_min, obs.x_min),
                    obs.y_max - obs.y_min,
                    obs.x_max - obs.x_min,
                    linewidth=2, edgecolor='red', facecolor='red', alpha=0.5
                )
                ax3.add_patch(rect)
        
        # 绘制轨迹
        if trajectory:
            traj_states = np.array([s for s, a in trajectory])
            ax3.plot(traj_states[:, 1], traj_states[:, 0], 'r-', linewidth=2, label='Trajectory', alpha=0.8)
            ax3.plot(traj_states[0, 1], traj_states[0, 0], 'go', markersize=10, label='Start')
            ax3.plot(traj_states[-1, 1], traj_states[-1, 0], 'rs', markersize=10, label='End')
        
        ax3.plot(goal[1], goal[0], 'r*', markersize=20, label='Goal')
        ax3.legend()
        
        plt.tight_layout()
        
        # 保存
        fname = os.path.join(output_dir, f'failure_mode_{execution_mode}_{idx+1}_step{step:05d}.png')
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        
        saved_paths.append(fname)
    
    return saved_paths


def visualize_failure_start_distribution(
    env: gym.Env,
    failures_by_mode: Dict[str, List[dict]],
    *,
    output_dir: str,
    step: int,
) -> str:
    """
    聚合可视化：不同 execution mode 的 failure start 空间分布对比。
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Failure start distribution (by execution mode)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    nav_env = _as_nav_env(env)
    if nav_env is not None and hasattr(nav_env, "obstacles"):
        for obs in nav_env.obstacles:
            rect = patches.Rectangle(
                (obs.x_min, obs.y_min),
                obs.x_max - obs.x_min,
                obs.y_max - obs.y_min,
                linewidth=1.5,
                edgecolor="black",
                facecolor="gray",
                alpha=0.45,
            )
            ax.add_patch(rect)

    colors = {
        "greedy": "tab:red",
        "lookahead": "tab:blue",
    }

    for mode, cases in failures_by_mode.items():
        if not cases:
            continue
        starts = np.stack([np.asarray(c["start"], dtype=np.float32).reshape(2) for c in cases], axis=0)
        ax.scatter(
            starts[:, 0],
            starts[:, 1],
            s=26,
            alpha=0.75,
            c=colors.get(mode, None),
            label=f"{mode} (n={len(cases)})",
        )

    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()

    fname = os.path.join(output_dir, f"failure_start_distribution_step{step:05d}.png")
    plt.savefig(fname, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return fname


def evaluate_planning_reachability(
    agent: nn.Module,
    env: gym.Env,
    n_trials: int = 100,
    device: str = 'cpu',
    seed: Optional[int] = None,
    num_action_candidates: int = 32,
    visualize_failures: bool = False,
    output_dir: str = './results',
    step: int = 0,
    execution_modes: Optional[List[ExecutionMode]] = None,
    lookahead_config: Optional[LookaheadConfig] = None,
    starts: Optional[np.ndarray] = None,
    goals: Optional[np.ndarray] = None,
) -> Dict[str, any]:
    """
    综合评估 Planning / Reachability 功能
    
    整合所有三个评估功能：
    1. Greedy Navigation Success Rate
    2. Path Efficiency
    3. Failure Mode 可视化（可选）
    
    Args:
        agent: QRL Agent
        env: 环境实例
        n_trials: 测试次数
        device: 设备
        seed: 随机种子
        num_action_candidates: 每步候选动作数量
        visualize_failures: 是否可视化失败案例
        output_dir: 输出目录
        step: 训练步数
    
    Returns:
        包含所有评估指标的字典
    """
    results = {}
    
    if execution_modes is None:
        execution_modes = ["greedy"]

    # 为了公平对比：所有 mode（以及可选的不同 distance_type）可以共享同一批 start-goal 对
    if starts is None or goals is None:
        starts, goals = sample_state_goal_pairs(env, n_pairs=n_trials, seed=seed)

    path_eff_by_mode: Dict[str, dict] = {}
    failure_viz_by_mode: Dict[str, List[str]] = {}
    failure_cases_by_mode: Dict[str, List[dict]] = {}

    for mode in execution_modes:
        print(f"评估 Navigation Success Rate ({mode})...")
        success_metrics = evaluate_navigation_success_rate(
            agent, env, n_trials=n_trials, device=device, seed=seed,
            num_action_candidates=num_action_candidates,
            execution_mode=mode,
            lookahead_config=lookahead_config,
            starts=starts, goals=goals,
        )

        if mode == "greedy":
            # 保持与旧版 train.py / tensorboard tag 兼容
            results["greedy_navigation"] = success_metrics
        else:
            results[f"{mode}_navigation"] = success_metrics

        print(f"评估 Path Efficiency ({mode})...")
        efficiency_metrics = evaluate_path_efficiency(
            agent, env, n_trials=n_trials, device=device, seed=seed,
            num_action_candidates=num_action_candidates,
            execution_mode=mode,
            lookahead_config=lookahead_config,
            starts=starts, goals=goals,
        )
        path_eff_by_mode[mode] = efficiency_metrics

        if mode == "greedy" and "path_efficiency" not in results:
            # 旧 key：默认放 greedy
            results["path_efficiency"] = efficiency_metrics

        if visualize_failures:
            print(f"生成 Failure Mode 可视化 ({mode})...")

            # 先收集失败案例（用于分布对比）
            failure_cases: List[dict] = []
            max_attempts = max(50, min(500, n_trials * 10))
            starts_v, goals_v = sample_state_goal_pairs(env, n_pairs=max_attempts, seed=seed)
            for i in range(max_attempts):
                rr = navigation_rollout(
                    agent, env, starts_v[i], goals_v[i], device,
                    max_steps=env.max_episode_steps if hasattr(env, 'max_episode_steps') else 200,
                    num_action_candidates=num_action_candidates,
                    use_improved_termination=True,
                    execution_mode=mode,
                    lookahead_config=lookahead_config,
                )
                if not rr["success"]:
                    failure_cases.append({"start": starts_v[i], "goal": goals_v[i], "rollout_result": rr})
                if len(failure_cases) >= min(10, n_trials // 10):
                    break
            failure_cases_by_mode[mode] = failure_cases

            # 生成每个失败案例的热力图/轨迹可视化
            failure_viz_paths = visualize_failure_modes(
                agent, env, n_failures=min(10, n_trials // 10), device=device,
                seed=seed, output_dir=output_dir, step=step,
                num_action_candidates=num_action_candidates,
                execution_mode=mode,
                lookahead_config=lookahead_config,
            )
            failure_viz_by_mode[mode] = failure_viz_paths

    results["path_efficiency_by_mode"] = path_eff_by_mode

    if visualize_failures:
        results["failure_visualizations_by_mode"] = failure_viz_by_mode
        if len(execution_modes) >= 2:
            dist_path = visualize_failure_start_distribution(
                env,
                failure_cases_by_mode,
                output_dir=output_dir,
                step=step,
            )
            results["failure_start_distribution"] = dist_path
    
    return results
