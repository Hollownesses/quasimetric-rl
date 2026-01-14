"""
QRL 评估模块：计算真实距离、评估指标和可视化
"""
import os
from typing import Tuple, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 尝试导入 scipy，如果不可用则使用简化实现
try:
    from scipy.stats import spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    def spearmanr(x, y):
        """简化版 Spearman 相关系数（使用排序）"""
        # 使用 numpy 实现排序
        x_rank = np.argsort(np.argsort(x)) + 1.0
        y_rank = np.argsort(np.argsort(y)) + 1.0
        
        # Pearson 相关系数在排序后的数据上
        x_rank_centered = x_rank - x_rank.mean()
        y_rank_centered = y_rank - y_rank.mean()
        num = np.sum(x_rank_centered * y_rank_centered)
        den = np.sqrt(np.sum(x_rank_centered ** 2) * np.sum(y_rank_centered ** 2)) + 1e-12
        corr = num / den
        return corr, 0.0  # p-value 设为 0（无法计算）

from minimal_qrl.envs.simple_grid_2d import SimpleGrid2D


def compute_manhattan_distance(
    s: np.ndarray, 
    g: np.ndarray, 
    grid_size: Tuple[int, int]
) -> float:
    """
    计算两个状态之间的 Manhattan 距离（真实最短路）
    
    Args:
        s: 起始状态，归一化坐标 [0, 1]^2
        g: 目标状态，归一化坐标 [0, 1]^2
        grid_size: (height, width)
    
    Returns:
        Manhattan 距离（步数）
    """
    h, w = grid_size
    
    # 归一化坐标 -> 离散网格坐标
    sx = int(np.round(s[0] * (h - 1)))
    sy = int(np.round(s[1] * (w - 1)))
    gx = int(np.round(g[0] * (h - 1)))
    gy = int(np.round(g[1] * (w - 1)))
    
    # 限制在网格范围内
    sx = np.clip(sx, 0, h - 1)
    sy = np.clip(sy, 0, w - 1)
    gx = np.clip(gx, 0, h - 1)
    gy = np.clip(gy, 0, w - 1)
    
    # Manhattan 距离
    return float(np.abs(sx - gx) + np.abs(sy - gy))


def sample_state_goal_pairs(
    env: SimpleGrid2D,
    n_pairs: int = 2000,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    随机采样状态-目标对
    
    Args:
        env: 环境实例
        n_pairs: 采样对数
        seed: 随机种子
    
    Returns:
        states: (n_pairs, 2) 归一化状态
        goals: (n_pairs, 2) 归一化目标
    """
    if seed is not None:
        np.random.seed(seed)
    
    h, w = env.grid_size
    
    # 随机采样离散网格坐标
    s_x = np.random.randint(0, h, size=n_pairs)
    s_y = np.random.randint(0, w, size=n_pairs)
    g_x = np.random.randint(0, h, size=n_pairs)
    g_y = np.random.randint(0, w, size=n_pairs)
    
    # 转换为归一化坐标
    states = np.stack([
        s_x.astype(np.float32) / (h - 1),
        s_y.astype(np.float32) / (w - 1)
    ], axis=1)
    
    goals = np.stack([
        g_x.astype(np.float32) / (h - 1),
        g_y.astype(np.float32) / (w - 1)
    ], axis=1)
    
    return states, goals


def evaluate_quasimetric(
    agent: nn.Module,
    env: SimpleGrid2D,
    n_pairs: int = 2000,
    device: str = 'cpu',
    seed: Optional[int] = None
) -> Dict[str, float]:
    """
    评估 QRL 学习到的 quasimetric
    
    Args:
        agent: QRL Agent（包含 critics）
        env: 环境实例
        n_pairs: 采样对数
        device: 设备
        seed: 随机种子
    
    Returns:
        评估指标字典
    """
    # 采样状态-目标对
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
    
    # 计算真实距离
    gt_dists = np.array([
        compute_manhattan_distance(s, g, env.grid_size)
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
    env: SimpleGrid2D,
    goal: Optional[np.ndarray] = None,
    step: int = 0,
    output_dir: str = './results',
    device: str = 'cpu'
):
    """
    可视化距离场热力图
    
    Args:
        agent: QRL Agent
        env: 环境实例
        goal: 目标状态（归一化坐标），如果为 None 则使用 env.goal_pos
        step: 训练步数
        output_dir: 输出目录
        device: 设备
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 确定目标
    if goal is None:
        gx, gy = env.goal_pos
        h, w = env.grid_size
        goal = np.array([
            gx / (h - 1),
            gy / (w - 1)
        ], dtype=np.float32)
    
    h, w = env.grid_size
    device_obj = torch.device(device)
    
    # 创建网格
    x_coords = np.arange(h, dtype=np.float32) / (h - 1)
    y_coords = np.arange(w, dtype=np.float32) / (w - 1)
    Y, X = np.meshgrid(y_coords, x_coords)
    
    # 计算所有网格点的距离
    states = np.stack([X.flatten(), Y.flatten()], axis=1).astype(np.float32)
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
    gx, gy = int(np.round(goal[0] * (h - 1))), int(np.round(goal[1] * (w - 1)))
    gx, gy = np.clip(gx, 0, h - 1), np.clip(gy, 0, w - 1)
    
    for i in range(h):
        for j in range(w):
            s = np.array([i / (h - 1), j / (w - 1)], dtype=np.float32)
            gt_dists[i, j] = compute_manhattan_distance(s, goal, (h, w))
    
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
    ax2.set_title('Ground Truth Distance (Manhattan)')
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

