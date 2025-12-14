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
    local_weight: float = 1.0
    global_weight: float = 0.1
    tri_weight: float = 1.0
    tri_margin: float = 0.0
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

def local_consistency_loss(d_ssp, r):
    # d(s,s') + r >= 0  ->  ReLU(d + r)^2
    x = d_ssp + r
    relu_x = F.relu(x)
    return (relu_x ** 2).mean()


def global_push_loss(d_sg):
    # 想让 d(s,g) 尽量大 -> 最小化 -E[d(s,g)]
    return -d_sg.mean()


def triangle_inequality_loss(d_s1s3, d_s1s2, d_s2s3, margin: float = 0.0):
    # Enforce: d(s1,s3) <= d(s1,s2) + d(s2,s3) (soft penalty)
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

        loss_local = local_consistency_loss(d_ssp, r_t)
        loss_global = global_push_loss(d_sg)
        loss_tri = triangle_inequality_loss(d_13, d_12, d_23, margin=cfg.tri_margin)
        loss = (
            cfg.local_weight * loss_local
            + cfg.global_weight * loss_global
            + cfg.tri_weight * loss_tri
        )

        optim.zero_grad()
        loss.backward()
        optim.step()

        if step % 100 == 0:
            pbar.set_postfix(
                local_loss=f"{loss_local.item():.4f}",
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

            import matplotlib.pyplot as plt

            plt.figure(figsize=(6, 3))
            plt.plot(pos, dvals)
            plt.axvline(goal_pos, color="red", linestyle="--", label="goal")
            plt.xlabel("position")
            plt.ylabel("d_theta(s, g)")
            plt.title(f"Step {step}")
            plt.legend()
            plt.grid(True)
            fname = os.path.join(cfg.fig_dir, f"dpos_step{step}.png")
            plt.savefig(fname)
            plt.close()

    env.close()
    torch.save(model.state_dict(), os.path.join(cfg.fig_dir, "quasimetric_net.pth"))
    print("Training done, model & figures saved to", cfg.fig_dir)


if __name__ == "__main__":
    cfg = Config()
    main(cfg)