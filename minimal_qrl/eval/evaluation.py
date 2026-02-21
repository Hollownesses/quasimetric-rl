"""
QRL 评估模块：计算真实距离、评估指标和可视化
支持多种环境，使用环境的真实最短路径距离
"""
import os
from typing import Tuple, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import gym
from scipy.stats import spearmanr
from minimal_qrl.envs.base import BaseNavigationEnv
from .planning_evaluation import evaluate_planning_reachability


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


def evaluate_quasimetric(
    agent: nn.Module,
    env: gym.Env,
    n_pairs: int = 2000,
    device: str = 'cpu',
    seed: Optional[int] = None
) -> Dict[str, float]:
    """
    评估 QRL 学习到的 quasimetric
    
    使用环境的真实最短路径距离作为 ground truth
    
    Args:
        agent: QRL Agent（包含 critics）
        env: 环境实例（应实现 BaseNavigationEnv 接口）
        n_pairs: 采样对数
        device: 设备
        seed: 随机种子
    
    Returns:
        评估指标字典
    """
    # 采样状态-目标对（确保状态合法）
    states, goals = sample_state_goal_pairs(env, n_pairs=n_pairs, seed=seed)
    
    # 转换为 tensor
    states_t = torch.tensor(states, device=device, dtype=torch.float32)
    goals_t = torch.tensor(goals, device=device, dtype=torch.float32)
    
    # 计算预测距离（使用第一个 critic）
    critic = agent.critics[0]
    with torch.no_grad():
        # 编码状态和目标
        zx = critic.encoder(states_t)
        zy = critic.encoder(goals_t)
        # 计算 quasimetric 距离
        pred_dists = critic.quasimetric_model(zx, zy).cpu().numpy()
    
    # 计算真实距离（使用环境的真实最短路径距离）
    gt_dists = np.array([
        compute_ground_truth_distance(env, s, g)
        for s, g in zip(states, goals)
    ])
    
    # 计算评估指标
    pred_dists_flat = pred_dists.flatten()
    gt_dists_flat = gt_dists.flatten()
    
    # MSE
    mse = float(np.mean((pred_dists_flat - gt_dists_flat) ** 2))
    
    # MAE
    mae = float(np.mean(np.abs(pred_dists_flat - gt_dists_flat)))
    
    # Spearman 相关系数
    spearman_corr, spearman_p = spearmanr(pred_dists_flat, gt_dists_flat)
    spearman_corr = float(spearman_corr)
    
    # Pearson 相关系数
    pred_centered = pred_dists_flat - pred_dists_flat.mean()
    gt_centered = gt_dists_flat - gt_dists_flat.mean()
    pearson_num = np.sum(pred_centered * gt_centered)
    pearson_den = np.sqrt(np.sum(pred_centered ** 2) * np.sum(gt_centered ** 2)) + 1e-12
    pearson_corr = float(pearson_num / pearson_den)
    
    # 相对误差
    relative_error = float(np.mean(np.abs(pred_dists_flat - gt_dists_flat) / (gt_dists_flat + 1e-6)))
    
    # 统计信息
    pred_mean = float(pred_dists_flat.mean())
    pred_std = float(pred_dists_flat.std())
    gt_mean = float(gt_dists_flat.mean())
    gt_std = float(gt_dists_flat.std())
    
    return {
        'mse': mse,
        'mae': mae,
        'spearman_corr': spearman_corr,
        'spearman_p': float(spearman_p),
        'pearson_corr': pearson_corr,
        'relative_error': relative_error,
        'pred_mean': pred_mean,
        'pred_std': pred_std,
        'gt_mean': gt_mean,
        'gt_std': gt_std,
    }


def visualize_distance_field_heatmap(
    agent: nn.Module,
    env: gym.Env,
    goal: Optional[np.ndarray] = None,
    step: int = 0,
    output_dir: str = './results',
    device: str = 'cpu',
    resolution: Optional[Tuple[int, int]] = None
):
    """
    可视化距离场热力图
    
    支持多种环境，使用环境的真实最短路径距离
    
    Args:
        agent: QRL Agent
        env: 环境实例（应实现 BaseNavigationEnv 接口）
        goal: 目标状态（归一化坐标），如果为 None 则从环境获取
        step: 训练步数
        output_dir: 输出目录
        device: 设备
        resolution: 可视化分辨率 (height, width)，如果为 None 则根据环境自动确定
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 确定目标
    if goal is None:
        if isinstance(env, BaseNavigationEnv):
            if hasattr(env, 'goal_pos'):
                # SimpleGrid2D
                gx, gy = env.goal_pos
                h, w = env.grid_size
                goal = np.array([
                    gx / (h - 1),
                    gy / (w - 1)
                ], dtype=np.float32)
            elif hasattr(env, 'goal'):
                # ContinuousObstacle2D
                goal = np.array(env.goal, dtype=np.float32)
            else:
                # 采样一个合法目标
                goal = env.sample_valid_state()
        else:
            # 从观察空间采样
            goal = env.observation_space.sample()
    
    # 确定分辨率（降低分辨率可显著加速可视化）
    if resolution is None:
        if hasattr(env, 'grid_size'):
            # SimpleGrid2D - 使用网格大小
            h, w = env.grid_size
        else:
            # 障碍物环境 - 使用较低分辨率以加速
            h, w = 25, 25
    else:
        h, w = resolution
    
    device_obj = torch.device(device)
    
    # 创建网格
    x_coords = np.linspace(0.0, 1.0, h, dtype=np.float32)
    y_coords = np.linspace(0.0, 1.0, w, dtype=np.float32)
    Y, X = np.meshgrid(y_coords, x_coords)
    
    # 计算所有网格点的距离
    states = np.stack([X.flatten(), Y.flatten()], axis=1).astype(np.float32)
    
    # 过滤合法状态（对于障碍物环境）
    if isinstance(env, BaseNavigationEnv):
        valid_mask = np.array([env.is_valid_state(s) for s in states])
    else:
        valid_mask = np.ones(len(states), dtype=bool)
    
    states_t = torch.tensor(states, device=device_obj, dtype=torch.float32)
    goal_t = torch.tensor(goal[None].repeat(len(states), 0), device=device_obj, dtype=torch.float32)
    
    # 使用第一个 critic 计算预测距离
    critic = agent.critics[0]
    with torch.no_grad():
        zx = critic.encoder(states_t)
        zy = critic.encoder(goal_t)
        pred_dists = critic.quasimetric_model(zx, zy).cpu().numpy()
    
    pred_dists = pred_dists.reshape(h, w)
    
    # 计算真实距离（用于对比）
    gt_dists = np.zeros((h, w), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            s = np.array([x_coords[i], y_coords[j]], dtype=np.float32)
            gt_dists[i, j] = compute_ground_truth_distance(env, s, goal)
    
    # 绘制
    fig = plt.figure(figsize=(16, 6))
    
    # 左：预测距离
    ax1 = fig.add_subplot(131)
    im1 = ax1.imshow(pred_dists, origin='lower', extent=[0, 1, 0, 1],
                     cmap='viridis', aspect='auto', interpolation='nearest')
    ax1.set_xlabel('Y position (normalized)')
    ax1.set_ylabel('X position (normalized)')
    ax1.set_title(f'Predicted Distance to Goal (Step {step})')
    plt.colorbar(im1, ax=ax1, label='Distance')
    
    # 标记目标
    ax1.plot(goal[1], goal[0], 'r*', markersize=20, label='Goal')
    ax1.legend()
    
    # 中：真实距离
    ax2 = fig.add_subplot(132)
    im2 = ax2.imshow(gt_dists, origin='lower', extent=[0, 1, 0, 1],
                     cmap='viridis', aspect='auto', interpolation='nearest')
    ax2.set_xlabel('Y position (normalized)')
    ax2.set_ylabel('X position (normalized)')
    ax2.set_title('Ground Truth Distance (Shortest Path)')
    plt.colorbar(im2, ax=ax2, label='Distance')
    ax2.plot(goal[1], goal[0], 'r*', markersize=20, label='Goal')
    ax2.legend()
    
    # 右：误差
    ax3 = fig.add_subplot(133)
    error = np.abs(pred_dists - gt_dists)
    im3 = ax3.imshow(error, origin='lower', extent=[0, 1, 0, 1],
                      cmap='hot', aspect='auto', interpolation='nearest')
    ax3.set_xlabel('Y position (normalized)')
    ax3.set_ylabel('X position (normalized)')
    ax3.set_title('Absolute Error')
    plt.colorbar(im3, ax=ax3, label='|Pred - GT|')
    ax3.plot(goal[1], goal[0], 'r*', markersize=20, label='Goal')
    ax3.legend()
    
    plt.tight_layout()
    
    # 保存
    fname = os.path.join(output_dir, f'distance_heatmap_step{step:05d}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    
    return fname


def evaluate_planning(
    agent: nn.Module,
    env: gym.Env,
    n_trials: int = 100,
    device: str = 'cpu',
    seed: Optional[int] = None,
    num_action_candidates: int = 32,
    visualize_failures: bool = False,
    output_dir: str = './results',
    step: int = 0,
    execution_modes: Optional[list] = None,
    lookahead_config: Optional[object] = None,
    starts: Optional[np.ndarray] = None,
    goals: Optional[np.ndarray] = None,
) -> Dict[str, any]:
    """
    评估 Planning / Reachability 功能（便捷函数）
    
    这是 evaluate_planning_reachability 的便捷包装，用于与现有评估流程集成
    
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
    return evaluate_planning_reachability(
        agent=agent,
        env=env,
        n_trials=n_trials,
        device=device,
        seed=seed,
        num_action_candidates=num_action_candidates,
        visualize_failures=visualize_failures,
        output_dir=output_dir,
        step=step,
        execution_modes=execution_modes,
        lookahead_config=lookahead_config,
        starts=starts,
        goals=goals,
    )

