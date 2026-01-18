"""
Planning / Reachability 评估模块
专门面向 obstacle navigation 场景的评估功能：
1. Greedy Navigation Success Rate
2. Path Efficiency (与最短路径的竞争比)
3. Failure Mode 可视化
"""
import os
from typing import Tuple, Dict, Optional, List
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import gym
from minimal_qrl.envs.base import BaseNavigationEnv


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
    if isinstance(env, BaseNavigationEnv):
        return env.compute_shortest_path_distance(start=start, goal=goal)
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
    if isinstance(env, BaseNavigationEnv):
        for i in range(n_pairs):
            state = env.sample_valid_state(seed=seed + i if seed is not None else None)
            goal = env.sample_valid_state(seed=seed + i + n_pairs if seed is not None else None)
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
    num_candidates: int = 32
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
    
    Returns:
        选择的动作，形状为 (2,)
    """
    # 获取动作空间范围
    if hasattr(env, 'action_space'):
        action_space = env.action_space
        if hasattr(action_space, 'low') and hasattr(action_space, 'high'):
            low = action_space.low
            high = action_space.high
        else:
            # 默认范围
            low = np.array([-0.1, -0.1], dtype=np.float32)
            high = np.array([0.1, 0.1], dtype=np.float32)
    else:
        low = np.array([-0.1, -0.1], dtype=np.float32)
        high = np.array([0.1, 0.1], dtype=np.float32)
    
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
    best_dist = float('inf')
    
    # 批量计算距离以提高效率
    next_states = []
    valid_actions = []
    
    for action in candidates:
        # 计算执行动作后的下一状态（简化：直接相加，环境会在 step 时处理碰撞）
        next_state = current_state + action
        
        # 限制在边界内
        next_state = np.clip(next_state, 0.0, 1.0)
        
        # 检查是否合法（对于障碍物环境）
        if isinstance(env, BaseNavigationEnv):
            if not env.is_valid_state(next_state):
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
    
    critic = agent.critics[0]
    device_obj = torch.device(device)
    
    states_t = torch.tensor(next_states, device=device_obj, dtype=torch.float32)
    goal_t = torch.tensor(goal[None].repeat(len(next_states), 0), device=device_obj, dtype=torch.float32)
    
    with torch.no_grad():
        zx = critic.encoder(states_t)
        zy = critic.encoder(goal_t)
        dists = critic.quasimetric_model(zx, zy).cpu().numpy()
    
    # 选择最小距离对应的动作
    best_idx = np.argmin(dists)
    best_action = valid_actions[best_idx]
    
    return best_action


def greedy_navigation_rollout(
    agent: nn.Module,
    env: gym.Env,
    start: np.ndarray,
    goal: np.ndarray,
    device: str = 'cpu',
    max_steps: int = 200,
    num_action_candidates: int = 32
) -> Dict:
    """
    执行一次 greedy navigation rollout
    
    Args:
        agent: QRL Agent
        env: 环境实例
        start: 起始状态，形状为 (2,)
        goal: 目标状态，形状为 (2,)
        device: 设备
        max_steps: 最大步数
        num_action_candidates: 每步候选动作数量
    
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
        # 检查是否已到达目标
        dist_to_goal = np.linalg.norm(current_state - goal)
        if dist_to_goal < 0.05:  # 与环境的容差一致
            success = True
            break
        
        # 选择 greedy action
        action = greedy_action_selection(
            agent, env, current_state, goal, device, num_action_candidates
        )
        
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


def evaluate_greedy_navigation_success_rate(
    agent: nn.Module,
    env: gym.Env,
    n_trials: int = 100,
    device: str = 'cpu',
    seed: Optional[int] = None,
    num_action_candidates: int = 32
) -> Dict[str, float]:
    """
    评估 Greedy Navigation Success Rate
    
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
    
    # 采样起点和目标对
    starts, goals = sample_state_goal_pairs(env, n_pairs=n_trials, seed=seed)
    
    success_count = 0
    success_steps = []
    success_path_lengths = []
    all_steps = []
    all_path_lengths = []
    
    for i in range(n_trials):
        rollout_result = greedy_navigation_rollout(
            agent, env, starts[i], goals[i], device,
            max_steps=env.max_episode_steps if hasattr(env, 'max_episode_steps') else 200,
            num_action_candidates=num_action_candidates
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


def evaluate_path_efficiency(
    agent: nn.Module,
    env: gym.Env,
    n_trials: int = 100,
    device: str = 'cpu',
    seed: Optional[int] = None,
    num_action_candidates: int = 32
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
    
    # 采样起点和目标对
    starts, goals = sample_state_goal_pairs(env, n_pairs=n_trials, seed=seed)
    
    efficiency_ratios = []
    
    for i in range(n_trials):
        rollout_result = greedy_navigation_rollout(
            agent, env, starts[i], goals[i], device,
            max_steps=env.max_episode_steps if hasattr(env, 'max_episode_steps') else 200,
            num_action_candidates=num_action_candidates
        )
        
        if rollout_result['success']:
            # 计算最短路径距离
            shortest_dist = compute_ground_truth_distance(env, starts[i], goals[i])
            
            if shortest_dist > 1e-6:  # 避免除零
                efficiency_ratio = rollout_result['path_length'] / shortest_dist
                efficiency_ratios.append(efficiency_ratio)
    
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
    resolution: Tuple[int, int] = (50, 50)
) -> List[str]:
    """
    可视化 Failure Mode
    
    自动收集 greedy rollout 失败的起点，对这些 failure case 可视化：
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
        
        rollout_result = greedy_navigation_rollout(
            agent, env, starts[i], goals[i], device,
            max_steps=env.max_episode_steps if hasattr(env, 'max_episode_steps') else 200,
            num_action_candidates=num_action_candidates
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
        fname = os.path.join(output_dir, f'failure_mode_{idx+1}_step{step:05d}.png')
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        
        saved_paths.append(fname)
    
    return saved_paths


def evaluate_planning_reachability(
    agent: nn.Module,
    env: gym.Env,
    n_trials: int = 100,
    device: str = 'cpu',
    seed: Optional[int] = None,
    num_action_candidates: int = 32,
    visualize_failures: bool = False,
    output_dir: str = './results',
    step: int = 0
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
    
    # 1. Greedy Navigation Success Rate
    print("评估 Greedy Navigation Success Rate...")
    success_metrics = evaluate_greedy_navigation_success_rate(
        agent, env, n_trials=n_trials, device=device, seed=seed,
        num_action_candidates=num_action_candidates
    )
    results['greedy_navigation'] = success_metrics
    
    # 2. Path Efficiency
    print("评估 Path Efficiency...")
    efficiency_metrics = evaluate_path_efficiency(
        agent, env, n_trials=n_trials, device=device, seed=seed,
        num_action_candidates=num_action_candidates
    )
    results['path_efficiency'] = efficiency_metrics
    
    # 3. Failure Mode 可视化
    if visualize_failures:
        print("生成 Failure Mode 可视化...")
        failure_viz_paths = visualize_failure_modes(
            agent, env, n_failures=min(10, n_trials // 10), device=device,
            seed=seed, output_dir=output_dir, step=step,
            num_action_candidates=num_action_candidates
        )
        results['failure_visualizations'] = failure_viz_paths
    
    return results
