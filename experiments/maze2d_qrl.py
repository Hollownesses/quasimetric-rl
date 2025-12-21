# experiments/maze2d_qrl.py
"""
2D Maze 环境的 QRL 训练和可视化
- 使用简单的 2D grid maze 环境
- 通过 2D heatmap 可视化准度量距离
- 验证准度量的几何特性
"""
import os
import random
from dataclasses import dataclass
from typing import Tuple, List, Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D


# ---------- 简单的 2D Maze 环境 ----------

class SimpleMaze2D(gym.Env):
    """
    简单的 2D grid maze 环境
    - 状态: (x, y) 坐标
    - 动作: 0=上, 1=右, 2=下, 3=左
    - 奖励: 到达目标时 +1，否则 -0.01（每步成本）
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, maze_size: Tuple[int, int] = (10, 10), 
                 walls: Optional[List[Tuple[int, int]]] = None,
                 start_pos: Optional[Tuple[int, int]] = None,
                 goal_pos: Optional[Tuple[int, int]] = None,
                 render_mode: Optional[str] = None):
        super().__init__()
        
        self.maze_size = maze_size
        self.height, self.width = maze_size
        
        # 默认墙壁（创建一个简单的迷宫）
        if walls is None:
            walls = self._create_default_walls()
        self.walls = set(walls)
        
        # 起始和目标位置
        self.start_pos = start_pos if start_pos else (1, 1)
        self.goal_pos = goal_pos if goal_pos else (self.height - 2, self.width - 2)
        
        # 确保起始和目标不在墙上
        assert self.start_pos not in self.walls, "起始位置不能在墙上"
        assert self.goal_pos not in self.walls, "目标位置不能在墙上"
        
        # 动作空间：0=上, 1=右, 2=下, 3=左
        self.action_space = spaces.Discrete(4)
        
        # 状态空间：连续坐标 (x, y)，归一化到 [0, 1]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(2,), dtype=np.float32
        )
        
        self.render_mode = render_mode
        self.agent_pos = None
        
    def _create_default_walls(self) -> List[Tuple[int, int]]:
        """创建默认的迷宫墙壁"""
        walls = []
        h, w = self.height, self.width
        
        # 外圈墙壁
        for i in range(h):
            walls.append((i, 0))
            walls.append((i, w - 1))
        for j in range(w):
            walls.append((0, j))
            walls.append((h - 1, j))
        
        # 内部障碍物（创建一个简单的迷宫）
        # 中间一堵墙，留一个缺口
        mid = h // 2
        for j in range(1, w - 1):
            if j != w // 2:  # 留一个缺口
                walls.append((mid, j))
        
        return walls
    
    def _get_obs(self):
        """获取归一化的观察"""
        x, y = self.agent_pos
        # 归一化到 [0, 1]
        obs = np.array([x / (self.height - 1), y / (self.width - 1)], dtype=np.float32)
        return obs
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = self.start_pos
        obs = self._get_obs()
        info = {}
        return obs, info
    
    def step(self, action):
        x, y = self.agent_pos
        
        # 动作映射：0=上(-x), 1=右(+y), 2=下(+x), 3=左(-y)
        dx, dy = [(-1, 0), (0, 1), (1, 0), (0, -1)][action]
        new_x, new_y = x + dx, y + dy
        
        # 检查边界和墙壁
        if (0 <= new_x < self.height and 0 <= new_y < self.width and 
            (new_x, new_y) not in self.walls):
            self.agent_pos = (new_x, new_y)
        
        obs = self._get_obs()
        
        # 奖励：到达目标 +1，否则 -0.01（每步成本）
        terminated = (self.agent_pos == self.goal_pos)
        reward = 1.0 if terminated else -0.01
        
        truncated = False
        info = {"is_success": terminated}
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """渲染迷宫（简单文本输出）"""
        if self.render_mode == "human":
            grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
            
            # 绘制墙壁
            for x, y in self.walls:
                grid[x][y] = '#'
            
            # 绘制起始和目标
            sx, sy = self.start_pos
            gx, gy = self.goal_pos
            grid[sx][sy] = 'S'
            grid[gx][gy] = 'G'
            
            # 绘制智能体
            ax, ay = self.agent_pos
            if grid[ax][ay] not in ['S', 'G']:
                grid[ax][ay] = 'A'
            
            print("\n".join(["".join(row) for row in grid]))
            print()


# ---------- 配置 ----------

@dataclass
class Config:
    maze_size: Tuple[int, int] = (10, 10)  # (height, width)
    seed: int = 0
    buffer_size: int = 20000
    init_transitions: int = 5000
    batch_size: int = 256
    total_steps: int = 10000
    collect_interval: int = 500
    
    # Local constraint
    use_lagrange: bool = True
    lambda_init: float = 0.01
    lambda_lr: float = 0.01
    local_epsilon: float = 0.25
    step_cost: float = 1.0  # 每步的成本（注意：这是 quasimetric 距离的单位，不是环境奖励）
    # 注意：step_cost 应该反映实际的步数成本。如果最短路径约 114 步，
    # 那么从起始到目标的距离应该约为 114 * step_cost
    
    # Loss weights
    global_weight: float = 1.0
    tri_weight: float = 1.0
    tri_margin: float = 0.0
    
    # Global push
    global_beta: float = 0.1
    global_offset: float = 2.0  # 调整为适应较小的距离值
    # 注意：如果距离在 [0, 2] 范围，offset 应该设为 ~2-3
    # 原项目的 15.0 适用于更大的距离值
    
    lr: float = 1e-3
    hidden_dim: int = 128
    device: str = "cpu"
    fig_dir: str = "figs_maze2d_qrl"
    
    # 可视化相关
    viz_interval: int = 1000  # 每隔多少步可视化一次
    heatmap_resolution: int = 50  # heatmap 分辨率


# ---------- 经验池 ----------

class SimpleReplayBuffer:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer = []

    def add(self, s, a, s2, r, done):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append((s, a, s2, r, done))

    def add_trajectory(self, traj):
        for s, a, s2, r, done in traj:
            self.add(s, a, s2, r, done)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s = np.array([b[0] for b in batch], dtype=np.float32)
        a = np.array([b[1] for b in batch], dtype=np.int64)
        s2 = np.array([b[2] for b in batch], dtype=np.float32)
        r = np.array([b[3] for b in batch], dtype=np.float32)
        done = np.array([b[4] for b in batch], dtype=np.float32)
        return s, a, s2, r, done

    def __len__(self):
        return len(self.buffer)


def collect_random_trajectory(env, max_steps=200):
    obs, _ = env.reset()
    traj = []
    for _ in range(max_steps):
        a = env.action_space.sample()
        obs2, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        traj.append((obs.copy(), int(a), obs2.copy(), float(r), done))
        obs = obs2
        if done:
            break
    return traj


def collect_random_transitions(env, buffer, n_transitions, max_steps=200):
    while len(buffer) < n_transitions:
        traj = collect_random_trajectory(env, max_steps=max_steps)
        buffer.add_trajectory(traj)


# ---------- 模型：Quasimetric 网络 ----------

class QuasimetricNet(nn.Module):
    def __init__(self, state_dim=2, hidden_dim=128):
        super().__init__()
        input_dim = state_dim * 2  # s, g
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, s, g):
        # s, g: (B, state_dim)
        x = torch.cat([s, g], dim=-1)
        out = self.net(x)
        # softplus 保证非负 & 可导
        return F.softplus(out).squeeze(-1)  # (B,)


# ---------- 损失函数 ----------

def local_constraint_loss(d_ssp, step_cost: float = 1.0, epsilon: float = 0.25):
    """原项目 QRL 的 local constraint loss"""
    sq_deviation = (d_ssp - step_cost).relu().square().mean()
    violation = sq_deviation - (epsilon ** 2)
    return sq_deviation, violation


def global_push_loss(d_sg, beta: float = 0.1, offset: float = 15.0):
    """原项目 QRL 的 global push loss"""
    tsfm_dist = F.softplus(offset - d_sg, beta=beta)
    return tsfm_dist.mean()


def triangle_inequality_loss(d_s1s3, d_s1s2, d_s2s3, margin: float = 0.0):
    """三角不等式损失"""
    v = d_s1s3 - d_s1s2 - d_s2s3 + margin
    return (F.relu(v) ** 2).mean()


# ---------- 可视化函数 ----------

def visualize_quasimetric_heatmap(model, env: SimpleMaze2D, goal: np.ndarray, 
                                   step: int, fig_dir: str, resolution: int = 50):
    """
    可视化从所有位置到目标的准度量距离（2D heatmap）
    """
    device = next(model.parameters()).device
    
    # 创建网格
    h, w = env.maze_size
    x_coords = np.linspace(0, 1, resolution)
    y_coords = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # 计算每个网格点的距离
    states = np.stack([X.flatten(), Y.flatten()], axis=1).astype(np.float32)
    states_t = torch.tensor(states, device=device)
    goal_t = torch.tensor(goal[None].repeat(len(states), 0), device=device)
    
    with torch.no_grad():
        distances = model(states_t, goal_t).cpu().numpy()
    
    distances = distances.reshape(resolution, resolution)
    
    # 创建图形
    fig = plt.figure(figsize=(16, 7))
    
    # 左图：Heatmap
    ax1 = fig.add_subplot(121)
    im = ax1.imshow(distances, origin='lower', extent=[0, 1, 0, 1], 
                    cmap='viridis', aspect='auto', interpolation='bilinear')
    ax1.set_xlabel('Y position (normalized)')
    ax1.set_ylabel('X position (normalized)')
    ax1.set_title(f'Quasimetric Distance to Goal (Step {step})')
    plt.colorbar(im, ax=ax1, label='Distance')
    
    # 叠加迷宫墙壁
    for wall in env.walls:
        wx, wy = wall
        # 转换为归一化坐标
        nx, ny = wx / (h - 1), wy / (w - 1)
        rect = patches.Rectangle((ny - 0.5/resolution, nx - 0.5/resolution),
                               1/resolution, 1/resolution,
                               linewidth=0, facecolor='black', alpha=0.7)
        ax1.add_patch(rect)
    
    # 标记目标和起始位置
    gx, gy = env.goal_pos
    sx, sy = env.start_pos
    ax1.plot(gy / (w - 1), gx / (h - 1), 'r*', markersize=20, label='Goal')
    ax1.plot(sy / (w - 1), sx / (h - 1), 'go', markersize=15, label='Start')
    ax1.legend()
    
    # 右图：3D 表面图
    ax2 = fig.add_subplot(122, projection='3d')
    surf = ax2.plot_surface(X, Y, distances, cmap='viridis', alpha=0.8)
    ax2.set_xlabel('Y position')
    ax2.set_ylabel('X position')
    ax2.set_zlabel('Distance')
    ax2.set_title('3D Quasimetric Distance Surface')
    fig.colorbar(surf, ax=ax2, shrink=0.5)
    
    plt.tight_layout()
    fname = os.path.join(fig_dir, f"heatmap_step{step}.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Step {step}: 已保存 heatmap 到 {fname}")


def verify_quasimetric_properties(model, states: torch.Tensor, device: str):
    """
    验证准度量的几何特性
    1. 非负性: d(s, g) >= 0
    2. 自反性: d(s, s) = 0 (近似)
    3. 三角不等式: d(s1, s3) <= d(s1, s2) + d(s2, s3)
    """
    results = {}
    
    with torch.no_grad():
        # 1. 非负性
        n_samples = min(100, len(states))
        idx = torch.randperm(len(states))[:n_samples]
        s1 = states[idx]
        s2 = states[torch.randperm(len(states))[:n_samples]]
        d = model(s1, s2)
        results['non_negativity'] = {
            'min': d.min().item(),
            'all_non_neg': (d >= 0).all().item()
        }
        
        # 2. 自反性
        d_self = model(s1, s1)
        results['reflexivity'] = {
            'mean': d_self.mean().item(),
            'max': d_self.max().item()
        }
        
        # 3. 三角不等式
        s3 = states[torch.randperm(len(states))[:n_samples]]
        d_12 = model(s1, s2)
        d_23 = model(s2, s3)
        d_13 = model(s1, s3)
        violations = (d_13 - d_12 - d_23).relu()
        results['triangle_inequality'] = {
            'mean_violation': violations.mean().item(),
            'max_violation': violations.max().item(),
            'violation_rate': (violations > 1e-5).float().mean().item()
        }
    
    return results


# ---------- 训练主循环 ----------

def main(cfg: Config):
    os.makedirs(cfg.fig_dir, exist_ok=True)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # 创建环境
    env = SimpleMaze2D(maze_size=cfg.maze_size)
    state_dim = env.observation_space.shape[0]
    
    # 获取目标状态（归一化）
    goal_pos = env.goal_pos
    goal_state = np.array([
        goal_pos[0] / (env.height - 1),
        goal_pos[1] / (env.width - 1)
    ], dtype=np.float32)

    buffer = SimpleReplayBuffer(cfg.buffer_size)
    print("Collecting initial random transitions...")
    collect_random_transitions(env, buffer, cfg.init_transitions)
    print("Initial buffer size:", len(buffer))

    device = torch.device(cfg.device)
    model = QuasimetricNet(state_dim=state_dim, hidden_dim=cfg.hidden_dim).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # Lagrange multiplier
    lambda_local = float(cfg.lambda_init)

    # 收集一些状态用于验证
    all_states = []
    for _ in range(100):
        obs, _ = env.reset()
        all_states.append(obs)
    all_states_t = torch.tensor(np.array(all_states), device=device)

    pbar = trange(cfg.total_steps, desc="training")
    for step in pbar:
        if len(buffer) < cfg.batch_size:
            collect_random_transitions(env, buffer, cfg.init_transitions)

        # 1) 从经验池采样 (s, s', r)
        s, a, s2, r, done = buffer.sample(cfg.batch_size)
        s_t = torch.tensor(s, device=device)
        s2_t = torch.tensor(s2, device=device)
        r_t = torch.tensor(r, device=device)

        # 2) 为 global loss 采样 (s, g)
        idx = np.random.choice(len(buffer), cfg.batch_size)
        s_batch = np.array([buffer.buffer[i][0] for i in idx], dtype=np.float32)
        g_batch = np.tile(goal_state[None], (cfg.batch_size, 1))
        s_b_t = torch.tensor(s_batch, device=device)
        g_b_t = torch.tensor(g_batch, device=device)

        # 3) Triangle inequality batch
        idx1 = np.random.choice(len(buffer), cfg.batch_size)
        idx2 = np.random.choice(len(buffer), cfg.batch_size)
        idx3 = np.random.choice(len(buffer), cfg.batch_size)
        s1 = np.array([buffer.buffer[i][0] for i in idx1], dtype=np.float32)
        s2 = np.array([buffer.buffer[i][0] for i in idx2], dtype=np.float32)
        s3 = np.array([buffer.buffer[i][0] for i in idx3], dtype=np.float32)
        s1_t = torch.tensor(s1, device=device)
        s2_t = torch.tensor(s2, device=device)
        s3_t = torch.tensor(s3, device=device)

        d_ssp = model(s_t, s2_t)
        d_sg = model(s_b_t, g_b_t)
        d_13 = model(s1_t, s3_t)
        d_12 = model(s1_t, s2_t)
        d_23 = model(s2_t, s3_t)

        # 计算损失
        sq_deviation, violation = local_constraint_loss(
            d_ssp, step_cost=cfg.step_cost, epsilon=cfg.local_epsilon
        )
        loss_global = global_push_loss(d_sg, beta=cfg.global_beta, offset=cfg.global_offset)
        loss_tri = triangle_inequality_loss(d_13, d_12, d_23, margin=cfg.tri_margin)

        if cfg.use_lagrange:
            loss = cfg.global_weight * loss_global + cfg.tri_weight * loss_tri + lambda_local * violation
        else:
            loss = cfg.global_weight * loss_global + cfg.tri_weight * loss_tri + sq_deviation

        optim.zero_grad()
        loss.backward()
        optim.step()

        # 更新 lambda
        if cfg.use_lagrange:
            lambda_local = max(0.0, lambda_local + cfg.lambda_lr * float(violation.detach().cpu().item()))

        if step % 100 == 0:
            pbar.set_postfix(
                lambda_=f"{lambda_local:.3f}",
                sq_dev=f"{sq_deviation.item():.4f}",
                violation=f"{violation.item():.4f}",
                global_loss=f"{loss_global.item():.4f}",
                tri_loss=f"{loss_tri.item():.4f}",
                total_loss=f"{loss.item():.4f}",
            )

        # 不定期再收一点随机数据
        if step > 0 and step % cfg.collect_interval == 0:
            collect_random_transitions(env, buffer, cfg.init_transitions // 2)

        # 可视化和验证
        if step % cfg.viz_interval == 0:
            # 可视化 heatmap
            visualize_quasimetric_heatmap(
                model, env, goal_state, step, cfg.fig_dir, 
                resolution=cfg.heatmap_resolution
            )
            
            # 验证准度量性质
            props = verify_quasimetric_properties(model, all_states_t, device)
            print(f"\nStep {step} - 准度量性质验证:")
            print(f"  非负性: min={props['non_negativity']['min']:.4f}, "
                  f"all_non_neg={props['non_negativity']['all_non_neg']}")
            print(f"  自反性: mean={props['reflexivity']['mean']:.4f}, "
                  f"max={props['reflexivity']['max']:.4f}")
            print(f"  三角不等式: mean_violation={props['triangle_inequality']['mean_violation']:.4f}, "
                  f"violation_rate={props['triangle_inequality']['violation_rate']:.2%}")

    env.close()
    torch.save(model.state_dict(), os.path.join(cfg.fig_dir, "quasimetric_net.pth"))
    print("Training done, model & figures saved to", cfg.fig_dir)


if __name__ == "__main__":
    cfg = Config()
    main(cfg)
