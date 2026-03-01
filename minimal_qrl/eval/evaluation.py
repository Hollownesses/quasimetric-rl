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
from .planning_evaluation import (
    evaluate_planning_reachability,
    compute_ground_truth_distance,
    sample_state_goal_pairs,
)


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
    # 采样状态-目标对（确保状态合法）；对 Dubins 等为内部状态 (x,y,theta)
    states_raw, goals_raw = sample_state_goal_pairs(env, n_pairs=n_pairs, seed=seed)
    
    # 若环境提供 state_to_observation（如 Dubins use_cos_sin_obs），转为网络输入
    if hasattr(env, 'state_to_observation'):
        u = env.unwrapped if hasattr(env, 'unwrapped') else env
        if hasattr(u, 'state_to_observation'):
            states = np.array([u.state_to_observation(s) for s in states_raw], dtype=np.float32)
            goals = np.array([u.state_to_observation(g) for g in goals_raw], dtype=np.float32)
        else:
            states, goals = states_raw, goals_raw
    else:
        states, goals = states_raw, goals_raw
    
    # 转换为 tensor
    states_t = torch.tensor(states, device=device, dtype=torch.float32)
    goals_t = torch.tensor(goals, device=device, dtype=torch.float32)
    
    # 计算预测距离（使用第一个 critic）
    critic = agent.critics[0]
    with torch.no_grad():
        zx = critic.encoder(states_t)
        zy = critic.encoder(goals_t)
        pred_dists = critic.quasimetric_model(zx, zy).cpu().numpy()
    
    # 若环境提供 get_distance_scale（如 Dubins 用 step_cost=1，预测为「步数」，乘 dt 得时间）
    scale = None
    u = getattr(env, 'unwrapped', env)
    if hasattr(u, 'get_distance_scale'):
        scale = u.get_distance_scale()
    if scale is not None:
        pred_dists = pred_dists * scale
    
    # 真实距离用原始状态对计算（compute_ground_truth_distance 接受内部状态）
    gt_dists = np.array([
        compute_ground_truth_distance(env, s, g)
        for s, g in zip(states_raw, goals_raw)
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


def _is_dubins_like(env: gym.Env) -> bool:
    """是否 Dubins 类环境（有 bounds 且支持 state_to_observation）。"""
    u = getattr(env, 'unwrapped', env)
    return hasattr(u, 'bounds') and hasattr(u, 'state_to_observation')


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
    可视化距离场热力图（时间代价场）。
    支持 2D 环境与 Dubins 3D 状态（固定 theta=0 的 2D 切片）。
    """
    heatmap_dir = os.path.join(output_dir, 'distance_heatmap')
    os.makedirs(heatmap_dir, exist_ok=True)
    device_obj = torch.device(device)
    u = getattr(env, 'unwrapped', env)

    if _is_dubins_like(env):
        # Dubins: 地图为 bounds，状态 (x, y, theta)；固定 theta=0 画 2D 时间代价场
        x_min, y_min, x_max, y_max = u.bounds
        if goal is None:
            goal_3d = np.array(u.goal, dtype=np.float32) if hasattr(u, 'goal') and u.goal is not None else u.sample_valid_state()
        else:
            goal_3d = np.asarray(goal, dtype=np.float32).reshape(3)
        if resolution is None:
            h, w = 25, 25
        else:
            h, w = resolution
        x_coords = np.linspace(x_min, x_max, h, dtype=np.float32)
        y_coords = np.linspace(y_min, y_max, w, dtype=np.float32)
        Y, X = np.meshgrid(y_coords, x_coords)
        theta_fixed = 0.0
        states_3d = np.stack([
            X.flatten(), Y.flatten(), np.full(h * w, theta_fixed, dtype=np.float32)
        ], axis=1)
        valid_mask = np.array([u.is_valid_state(s) for s in states_3d])
        states_obs = np.array([u.state_to_observation(s) for s in states_3d], dtype=np.float32)
        goal_obs = u.state_to_observation(goal_3d)
        states_t = torch.tensor(states_obs, device=device_obj, dtype=torch.float32)
        goal_t = torch.tensor(goal_obs, device=device_obj, dtype=torch.float32).unsqueeze(0).expand(len(states_obs), -1)
        critic = agent.critics[0]
        with torch.no_grad():
            zx = critic.encoder(states_t)
            zy = critic.encoder(goal_t)
            pred_dists = critic.quasimetric_model(zx, zy).cpu().numpy()
        pred_dists = pred_dists.reshape(h, w).astype(np.float32)
        if hasattr(u, 'get_distance_scale'):
            pred_dists = pred_dists * u.get_distance_scale()
        invalid = ~valid_mask.reshape(h, w)
        pred_dists[invalid] = np.nan
        gt_dists = np.zeros((h, w), dtype=np.float32)
        for i in range(h):
            for j in range(w):
                s = np.array([x_coords[i], y_coords[j], theta_fixed], dtype=np.float32)
                gt_dists[i, j] = compute_ground_truth_distance(env, s, goal_3d)
        gt_dists[invalid] = np.nan
        extent = [y_min, y_max, x_min, x_max]
        xlabel, ylabel = 'y', 'x'
        goal_plot_xy = (goal_3d[1], goal_3d[0])
        title_suffix = ' (Time-to-Go, theta=0)'
    else:
        # 2D 环境（SimpleGrid / ContinuousObstacle）
        if goal is None:
            if isinstance(env, BaseNavigationEnv):
                if hasattr(env, 'goal_pos'):
                    gx, gy = env.goal_pos
                    hg, wg = env.grid_size
                    goal = np.array([gx / (hg - 1), gy / (wg - 1)], dtype=np.float32)
                elif hasattr(env, 'goal'):
                    goal = np.array(env.goal, dtype=np.float32)
                else:
                    goal = env.sample_valid_state()
            else:
                goal = env.observation_space.sample()
        if resolution is None:
            h, w = (env.grid_size if hasattr(env, 'grid_size') else (25, 25))
        else:
            h, w = resolution
        x_coords = np.linspace(0.0, 1.0, h, dtype=np.float32)
        y_coords = np.linspace(0.0, 1.0, w, dtype=np.float32)
        Y, X = np.meshgrid(y_coords, x_coords)
        states = np.stack([X.flatten(), Y.flatten()], axis=1).astype(np.float32)
        if isinstance(env, BaseNavigationEnv):
            valid_mask = np.array([env.is_valid_state(s) for s in states])
        else:
            valid_mask = np.ones(len(states), dtype=bool)
        states_t = torch.tensor(states, device=device_obj, dtype=torch.float32)
        goal_t = torch.tensor(goal[None].repeat(len(states), 0), device=device_obj, dtype=torch.float32)
        critic = agent.critics[0]
        with torch.no_grad():
            zx = critic.encoder(states_t)
            zy = critic.encoder(goal_t)
            pred_dists = critic.quasimetric_model(zx, zy).cpu().numpy()
        pred_dists = pred_dists.reshape(h, w)
        gt_dists = np.zeros((h, w), dtype=np.float32)
        for i in range(h):
            for j in range(w):
                s = np.array([x_coords[i], y_coords[j]], dtype=np.float32)
                gt_dists[i, j] = compute_ground_truth_distance(env, s, goal)
        extent = [0, 1, 0, 1]
        xlabel, ylabel = 'Y position (normalized)', 'X position (normalized)'
        goal_plot_xy = (goal[1], goal[0])
        title_suffix = ''

    # 绘制
    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(131)
    im1 = ax1.imshow(pred_dists, origin='lower', extent=extent, cmap='viridis', aspect='auto', interpolation='nearest')
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title(f'Predicted Distance to Goal (Step {step}){title_suffix}')
    plt.colorbar(im1, ax=ax1, label='Distance')
    ax1.plot(goal_plot_xy[0], goal_plot_xy[1], 'r*', markersize=20, label='Goal')
    ax1.legend()

    ax2 = fig.add_subplot(132)
    im2 = ax2.imshow(gt_dists, origin='lower', extent=extent, cmap='viridis', aspect='auto', interpolation='nearest')
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)
    ax2.set_title('Ground Truth (Shortest Path / Time-to-Go)' + title_suffix)
    plt.colorbar(im2, ax=ax2, label='Distance')
    ax2.plot(goal_plot_xy[0], goal_plot_xy[1], 'r*', markersize=20, label='Goal')
    ax2.legend()

    ax3 = fig.add_subplot(133)
    error = np.abs(np.nan_to_num(pred_dists, nan=0) - np.nan_to_num(gt_dists, nan=0))
    im3 = ax3.imshow(error, origin='lower', extent=extent, cmap='hot', aspect='auto', interpolation='nearest')
    ax3.set_xlabel(xlabel)
    ax3.set_ylabel(ylabel)
    ax3.set_title('Absolute Error')
    plt.colorbar(im3, ax=ax3, label='|Pred - GT|')
    ax3.plot(goal_plot_xy[0], goal_plot_xy[1], 'r*', markersize=20, label='Goal')
    ax3.legend()

    plt.tight_layout()
    fname = os.path.join(heatmap_dir, f'distance_heatmap_step{step:05d}.png')
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

