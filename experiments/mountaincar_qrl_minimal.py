# experiments/mountaincar_qrl_minimal.py
import os
import random
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange


# ---------- 配置 ----------

@dataclass
class Config:
    env_id: str = "MountainCar-v0"
    seed: int = 0
    buffer_size: int = 20000
    init_transitions: int = 5000
    batch_size: int = 256
    total_steps: int = 5000
    collect_interval: int = 500  # 每隔多少个优化 step 再收一批随机数据

    # Local constraint strength is controlled by a Lagrange multiplier (more paper-like)
    use_lagrange: bool = True
    lambda_init: float = 0.01  # 原项目默认值
    lambda_lr: float = 0.01
    local_epsilon: float = 0.25  # 原项目默认值: 约束 E[relu(d - cost)^2] <= epsilon^2
    step_cost: float = 1.0  # 每步的成本（原项目默认值）

    # Loss weights
    global_weight: float = 1.0
    tri_weight: float = 1.0
    tri_margin: float = 0.0

    # Make global push bounded (avoid pushing d to infinity)
    global_beta: float = 0.1  # softplus beta 参数
    global_offset: float = 15.0  # 原项目默认值 (softplus_offset)

    lr: float = 1e-3
    hidden_dim: int = 128
    device: str = "cpu"
    fig_dir: str = "figs_mountaincar_qrl"


# ---------- 环境 & 经验池 ----------

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


# ---------- 模型：简单 quasimetric 网络 ----------

class QuasimetricNet(nn.Module):
    def __init__(self, state_dim=2, hidden_dim=128):
        super().__init__()
        input_dim = state_dim * 2  # s, g
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
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


# ---------- 损失 ----------

def local_constraint_loss(d_ssp, step_cost: float = 1.0, epsilon: float = 0.25):
    """
    原项目 QRL 的 local constraint loss:
    - 约束: E[relu(d(s,s') - step_cost)^2] <= epsilon^2
    - 即不能高估观察到的局部距离/成本
    - 使用 Lagrange multiplier 进行约束优化
    """
    # sq_deviation = E[relu(d - step_cost)^2]
    sq_deviation = (d_ssp - step_cost).relu().square().mean()
    # violation = sq_deviation - epsilon^2 (如果 > 0 表示违反约束)
    violation = sq_deviation - (epsilon ** 2)
    return sq_deviation, violation


def global_push_loss(d_sg, beta: float = 0.1, offset: float = 15.0):
    """
    原项目 QRL 的 global push loss:
    - 目标: 最大化 E[d(s,g)]，推远所有状态-目标对
    - 使用 softplus 变换避免过度惩罚大距离
    - 原项目公式: F.softplus(offset - dists, beta=beta)
    - 等价于: F.softplus(beta * (offset - d)) / beta (当 beta 相同时)
    """
    # 原项目的实现: F.softplus(offset - d_sg, beta=beta)
    tsfm_dist = F.softplus(offset - d_sg, beta=beta)
    return tsfm_dist.mean()


def triangle_inequality_loss(d_s1s3, d_s1s2, d_s2s3, margin: float = 0.0):
    """
    三角不等式损失（可选）:
    - 原项目使用 IQE (Interval Quasimetric Embedding) 自动保证三角不等式
    - 简化版本可能需要显式约束: d(s1,s3) <= d(s1,s2) + d(s2,s3)
    """
    v = d_s1s3 - d_s1s2 - d_s2s3 + margin
    return (F.relu(v) ** 2).mean()


# ---------- 训练主循环 ----------

def main(cfg: Config):
    os.makedirs(cfg.fig_dir, exist_ok=True)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    env = gym.make(cfg.env_id)
    state_dim = env.observation_space.shape[0]

    buffer = SimpleReplayBuffer(cfg.buffer_size)
    print("Collecting initial random transitions...")
    collect_random_transitions(env, buffer, cfg.init_transitions)
    print("Initial buffer size:", len(buffer))

    device = torch.device(cfg.device)
    model = QuasimetricNet(state_dim=state_dim, hidden_dim=cfg.hidden_dim).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # Lagrange multiplier for local constraint (kept as a non-negative scalar)
    lambda_local = float(cfg.lambda_init)

    # 固定一组 goal（车在山顶附近，速度为 0）
    goal_pos = 0.5
    n_goals = 32
    goal_positions = np.linspace(goal_pos - 0.02, goal_pos + 0.02, n_goals)
    goal_states = np.stack([goal_positions, np.zeros_like(goal_positions)], axis=1).astype(
        np.float32
    )
    # goal_states_t = torch.tensor(goal_states, device=device)

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
        g_idx = np.random.randint(0, goal_states.shape[0], size=cfg.batch_size)
        g_batch = goal_states[g_idx]
        s_b_t = torch.tensor(s_batch, device=device)
        g_b_t = torch.tensor(g_batch, device=device)

        # 3) Triangle inequality batch: sample (s1, s2, s3) from replay
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

        # Local constraint: 原项目 QRL 的约束形式
        # 约束: E[relu(d(s,s') - step_cost)^2] <= epsilon^2
        sq_deviation, violation = local_constraint_loss(
            d_ssp, step_cost=cfg.step_cost, epsilon=cfg.local_epsilon
        )

        # Bounded global push (原项目实现)
        loss_global = global_push_loss(d_sg, beta=cfg.global_beta, offset=cfg.global_offset)
        
        # Triangle inequality loss (可选，原项目用 IQE 保证，这里保留作为正则化)
        loss_tri = triangle_inequality_loss(d_13, d_12, d_23, margin=cfg.tri_margin)

        if cfg.use_lagrange:
            # Lagrangian form: minimize (global + tri) + lambda * violation
            # 注意：原项目中 lambda 是梯度上升更新的（minimax），这里简化处理
            loss = cfg.global_weight * loss_global + cfg.tri_weight * loss_tri + lambda_local * violation
        else:
            # Fallback: fixed-weight sum
            loss = cfg.global_weight * loss_global + cfg.tri_weight * loss_tri + sq_deviation

        optim.zero_grad()
        loss.backward()
        optim.step()

        # Gradient-ascent update on lambda to enforce the constraint (keep lambda >= 0)
        # 原项目使用 optimizer 更新 lambda，这里简化用梯度上升
        if cfg.use_lagrange:
            # violation > 0 表示违反约束，需要增大 lambda
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

        # 不定期再收一点随机数据，丰富 buffer
        if step > 0 and step % cfg.collect_interval == 0:
            collect_random_transitions(env, buffer, cfg.init_transitions // 2)

        # 每 1000 步画一张 d(s, g) 随位置的曲线
        if step % 1000 == 0:
            with torch.no_grad():
                pos = np.linspace(-1.2, 0.6, 200)
                vel = np.zeros_like(pos)
                s_grid = np.stack([pos, vel], axis=1).astype(np.float32)
                s_grid_t = torch.tensor(s_grid, device=device)
                g = np.array([[goal_pos, 0.0]] * s_grid.shape[0], dtype=np.float32)
                g_t = torch.tensor(g, device=device)
                dvals = model(s_grid_t, g_t).cpu().numpy()

                # 验证 quasimetric 性质
                # 1. 非负性
                assert (dvals >= 0).all(), f"Step {step}: 距离必须非负！"
                
                # 2. 自反性（d(s,s) = 0，近似检查）
                s_self = s_grid_t[:10]  # 采样一些状态
                d_self = model(s_self, s_self).cpu().numpy()
                max_self_dist = d_self.max()
                if max_self_dist > 0.1:
                    print(f"警告 Step {step}: d(s,s) 应该接近 0，但最大值为 {max_self_dist:.4f}")

            import matplotlib.pyplot as plt

            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.plot(pos, dvals)
            plt.axvline(goal_pos, color="red", linestyle="--", label="goal")
            plt.xlabel("position")
            plt.ylabel("d_theta(s, g)")
            plt.title(f"Step {step} - Distance to Goal")
            plt.legend()
            plt.grid(True)
            
            # 添加损失信息
            plt.subplot(1, 2, 2)
            plt.text(0.5, 0.5, f"Loss Info:\n"
                    f"sq_dev: {sq_deviation.item():.4f}\n"
                    f"violation: {violation.item():.4f}\n"
                    f"lambda: {lambda_local:.4f}\n"
                    f"global: {loss_global.item():.4f}\n"
                    f"tri: {loss_tri.item():.4f}",
                    ha='center', va='center', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            plt.axis('off')
            plt.title("Training Status")
            
            fname = os.path.join(cfg.fig_dir, f"dpos_step{step}.png")
            plt.tight_layout()
            plt.savefig(fname, dpi=150)
            plt.close()
            
            print(f"Step {step}: 已保存可视化到 {fname}")

    env.close()
    torch.save(model.state_dict(), os.path.join(cfg.fig_dir, "quasimetric_net.pth"))
    print("Training done, model & figures saved to", cfg.fig_dir)


if __name__ == "__main__":
    cfg = Config()
    main(cfg)