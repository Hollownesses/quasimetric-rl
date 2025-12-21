# experiments/maze2d_qrl.py
"""
Open-grid (no walls) 2D environment: QRL-style quasimetric training and visualization
- Uses an obstacle-free 2D grid world for QRL loss sanity check
- Visualizes learned quasimetric via 2D heatmaps
- Verifies basic quasimetric properties
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

# --------- grad_mul utility for Lagrange multiplier (repo-style) ----------
class _GradMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale: float):
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None

def grad_mul(x: torch.Tensor, scale: float) -> torch.Tensor:
    return _GradMul.apply(x, scale)



# ---------- Open 2D Grid Environment (no obstacles) ----------
class OpenGrid2D(gym.Env):
    """
    Open 2D grid world (no obstacles) for QRL sanity-check.

    - State: normalized (x, y) in [0, 1]^2 mapped from discrete grid cells.
    - Actions: 0=up, 1=right, 2=down, 3=left (4-neighborhood).
    - Reward is not used by the simplified QRL losses here; we keep a step penalty for completeness.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        grid_size: Tuple[int, int] = (10, 10),
        start_pos: Optional[Tuple[int, int]] = None,
        goal_pos: Optional[Tuple[int, int]] = None,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 200,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.height, self.width = grid_size
        self.start_pos = start_pos if start_pos else (1, 1)
        self.goal_pos = goal_pos if goal_pos else (self.height - 2, self.width - 2)
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps

        # 0=up,1=right,2=down,3=left
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

        self.agent_pos: Tuple[int, int] = self.start_pos
        self._t = 0

    def _get_obs(self) -> np.ndarray:
        x, y = self.agent_pos
        return np.array([x / (self.height - 1), y / (self.width - 1)], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = self.start_pos
        self._t = 0
        return self._get_obs(), {}

    def step(self, action: int):
        x, y = self.agent_pos
        dx, dy = [(-1, 0), (0, 1), (1, 0), (0, -1)][int(action)]
        nx = int(np.clip(x + dx, 0, self.height - 1))
        ny = int(np.clip(y + dy, 0, self.width - 1))
        self.agent_pos = (nx, ny)

        self._t += 1
        terminated = (self.agent_pos == self.goal_pos)
        truncated = (self._t >= self.max_episode_steps)

        # Not used by our loss; kept for completeness
        reward = 0.0 if terminated else -1.0

        return self._get_obs(), float(reward), terminated, truncated, {"is_success": terminated}

    def render(self):
        if self.render_mode != "human":
            return
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        sx, sy = self.start_pos
        gx, gy = self.goal_pos
        grid[sx][sy] = 'S'
        grid[gx][gy] = 'G'
        ax, ay = self.agent_pos
        if grid[ax][ay] not in ['S', 'G']:
            grid[ax][ay] = 'A'
        print("\n".join(["".join(row) for row in grid]))
        print()


# ---------- 配置 ----------

@dataclass
class Config:
    grid_size: Tuple[int, int] = (10, 10)  # (height, width)
    seed: int = 0
    buffer_size: int = 20000
    init_transitions: int = 5000
    batch_size: int = 256
    total_steps: int = 10000
    collect_interval: int = 500

    # Local constraint
    use_lagrange: bool = False
    lambda_init: float = 0.01
    lambda_lr: float = 0.01
    local_epsilon: float = 0.25
    step_cost: float = 1.0  # local step cost in the learned quasimetric (open grid => shortest path ~ Manhattan distance)

    # Loss weights
    global_weight: float = 1.0
    tri_weight: float = 0.0
    tri_margin: float = 0.0
    # Anchor terms to avoid the trivial constant solution in open-grid sanity checks.
    # (The full repo has additional structure; in this demo we add two minimal anchors.)
    self_weight: float = 1.0        # encourages d(s,s) ~ 0
    step_weight: float = 1.0        # encourages d(s,s') ~ step_cost for one-step transitions

    # Optional tiny weight to prevent very large values even if other anchors are weak.
    scale_weight: float = 1e-3

    # Global push
    global_beta: float = 0.1
    global_offset: float = 20.0  # should be >= typical distances; tune ~ 2*(H+W) for open grids

    # Lagrange multiplier updates
    lagrange_update_every: int = 1  # update dual variable every N steps

    lr: float = 1e-3
    hidden_dim: int = 128
    device: str = "cpu"
    fig_dir: str = "figs_maze2d_qrl"

    # Visualization
    viz_interval: int = 1000  # Visualization interval
    heatmap_resolution: int = 10  # heatmap resolution (use grid resolution for open-grid sanity check)


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
    """
    Simple MLP quasimetric head for (s,g) -> d(s,g).
    - Enforces non-negativity via softplus.
    - IMPORTANT: No hard clamp, to avoid the 'constant plateau at max_dist' degenerate solution.
    """
    def __init__(self, state_dim: int = 2, hidden_dim: int = 128):
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

    def forward(self, s: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s, g], dim=-1)
        out = self.net(x)
        d = F.softplus(out).squeeze(-1)
        return d


def _snap_to_grid(s: torch.Tensor, h: int, w: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """normalized state -> discrete grid index (x,y) with rounding."""
    x = torch.round(s[..., 0] * (h - 1)).long()
    y = torch.round(s[..., 1] * (w - 1)).long()
    x = torch.clamp(x, 0, h - 1)
    y = torch.clamp(y, 0, w - 1)
    return x, y


def four_neighbor_next_states(s: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """
    Given normalized s: (B,2), return next states for 4-neighbors: (B,4,2)
    Action order: 0=up,1=right,2=down,3=left
    """
    x, y = _snap_to_grid(s, h, w)  # (B,), (B,)
    dx = torch.tensor([-1, 0, 1, 0], device=s.device).view(1, 4)
    dy = torch.tensor([0, 1, 0, -1], device=s.device).view(1, 4)

    xn = torch.clamp(x.view(-1, 1) + dx, 0, h - 1)
    yn = torch.clamp(y.view(-1, 1) + dy, 0, w - 1)

    sn = torch.stack([
        xn.float() / (h - 1),
        yn.float() / (w - 1)
    ], dim=-1)  # (B,4,2)
    return sn


# ---------- 损失函数 ----------


def local_constraint_stats_bellman_optimal(
    model: nn.Module,
    s: torch.Tensor,
    goal: torch.Tensor,
    h: int,
    w: int,
    step_cost: float = 1.0,
    epsilon: float = 0.25,
):
    """
    Bellman-optimal backup for 4-neighbor unit-cost grid:

        d(s,g) ≈ step_cost + min_{a in 4-neigh} d(s_a', g)

    We measure squared TD error:
        td = d(s,g) - (step_cost + min_a d(s_a',g))
        sq_deviation = mean(td^2)
        violation = sq_deviation - epsilon^2

    Also enforce terminal condition:
        if s == goal (on-grid), target = 0
    """
    B = s.shape[0]
    goal_b = goal.expand(B, -1)

    d_sg = model(s, goal_b)  # (B,)

    # 4 neighbor next states
    sn = four_neighbor_next_states(s, h, w)               # (B,4,2)
    sn_flat = sn.reshape(-1, 2)                           # (B*4,2)
    goal_flat = goal_b.unsqueeze(1).expand(B, 4, 2).reshape(-1, 2)

    with torch.no_grad():
        d_next = model(sn_flat, goal_flat).reshape(B, 4)  # (B,4)
        min_next, _ = torch.min(d_next, dim=1)            # (B,)

    target = step_cost + min_next  # (B,)

    # terminal: if s is exactly goal cell, target should be 0
    sx, sy = _snap_to_grid(s, h, w)
    gx, gy = _snap_to_grid(goal_b, h, w)
    is_goal = (sx == gx) & (sy == gy)
    target = torch.where(is_goal, torch.zeros_like(target), target)

    td = d_sg - target
    sq_deviation = td.square().mean()
    violation = sq_deviation - (epsilon ** 2)
    return d_sg, sq_deviation, violation


def global_push_loss(d_pairs: torch.Tensor, beta: float = 0.1, offset: float = 15.0):
    """Repo-style global push: mean(softplus(offset - d, beta)). We apply it to d(s,g) to prevent collapse."""
    tsfm = F.softplus(offset - d_pairs, beta=beta)
    return tsfm.mean()


def triangle_inequality_loss(d_s1s3, d_s1s2, d_s2s3, margin: float = 0.0):
    """三角不等式损失"""
    v = d_s1s3 - d_s1s2 - d_s2s3 + margin
    return (F.relu(v) ** 2).mean()


# ---------- 可视化函数 ----------

def visualize_quasimetric_heatmap(model, env: OpenGrid2D, goal: np.ndarray,
                                   step: int, fig_dir: str, resolution: int = 60):
    """
    Visualize the quasimetric distance to the goal as a 2D heatmap (open grid).
    """
    device = next(model.parameters()).device

    # Create grid (use discrete grid cells so L1/Manhattan structure is visible)
    h, w = env.grid_size
    x_coords = np.arange(h, dtype=np.float32) / (h - 1)
    y_coords = np.arange(w, dtype=np.float32) / (w - 1)
    Y, X = np.meshgrid(y_coords, x_coords)  # note: X is row (x), Y is col (y)

    # Compute distance for each grid cell center
    states = np.stack([X.flatten(), Y.flatten()], axis=1).astype(np.float32)
    states_t = torch.tensor(states, device=device)
    goal_t = torch.tensor(goal[None].repeat(len(states), 0), device=device)

    with torch.no_grad():
        distances = model(states_t, goal_t).cpu().numpy()

    distances = distances.reshape(h, w)

    # Plot
    fig = plt.figure(figsize=(16, 7))
    # Left: heatmap
    ax1 = fig.add_subplot(121)
    im = ax1.imshow(distances, origin='lower', extent=[0, 1, 0, 1],
                    cmap='viridis', aspect='auto', interpolation='nearest')
    ax1.set_xlabel('Y position (normalized)')
    ax1.set_ylabel('X position (normalized)')
    ax1.set_title(f'Quasimetric Distance to Goal (Step {step})')
    plt.colorbar(im, ax=ax1, label='Distance')

    # Mark goal and start
    gx, gy = env.goal_pos
    sx, sy = env.start_pos
    ax1.plot(gy / (w - 1), gx / (h - 1), 'r*', markersize=20, label='Goal')
    ax1.plot(sy / (w - 1), sx / (h - 1), 'go', markersize=15, label='Start')
    ax1.legend()

    # Right: 3D surface
    ax2 = fig.add_subplot(122, projection='3d')
    surf = ax2.plot_surface(Y, X, distances, cmap='viridis', alpha=0.8)
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


def _true_manhattan_to_goal_grid(env: OpenGrid2D) -> np.ndarray:
    """True unit-cost 4-neighbor distance-to-goal on an obstacle-free grid (Manhattan distance in steps)."""
    h, w = env.grid_size
    gx, gy = env.goal_pos
    xs = np.arange(h)[:, None]
    ys = np.arange(w)[None, :]
    return (np.abs(xs - gx) + np.abs(ys - gy)).astype(np.float32)


def eval_distance_field_mse(model: nn.Module, env: OpenGrid2D, goal: np.ndarray) -> Tuple[float, float, float]:
    """Evaluate d(s,goal) against true Manhattan distances on the discrete grid.

    Returns: (mse, mae, corr)
    """
    device = next(model.parameters()).device
    h, w = env.grid_size

    # Discrete grid states
    x_coords = np.arange(h, dtype=np.float32) / (h - 1)
    y_coords = np.arange(w, dtype=np.float32) / (w - 1)
    Y, X = np.meshgrid(y_coords, x_coords)
    states = np.stack([X.flatten(), Y.flatten()], axis=1).astype(np.float32)

    states_t = torch.tensor(states, device=device)
    goal_t = torch.tensor(goal[None].repeat(len(states), 0), device=device)

    with torch.no_grad():
        pred = model(states_t, goal_t).cpu().numpy().reshape(h, w)

    true = _true_manhattan_to_goal_grid(env)

    err = pred - true
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))

    # Pearson correlation (safe for small grids)
    p = pred.flatten()
    t = true.flatten()
    p = p - p.mean()
    t = t - t.mean()
    denom = (np.sqrt(np.sum(p * p)) * np.sqrt(np.sum(t * t)) + 1e-12)
    corr = float(np.sum(p * t) / denom)

    return mse, mae, corr


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

    # Build open grid environment
    env = OpenGrid2D(grid_size=cfg.grid_size)
    state_dim = env.observation_space.shape[0]

    # Derive a natural scale (in steps) for an obstacle-free grid.
    # Max Manhattan distance in steps is (H-1)+(W-1).
    max_steps_dist = float((env.height - 1) + (env.width - 1))
    # Global-push offset: keep small so it only prevents collapse (do NOT push everything to the diameter)
    cfg.global_offset = float(max(cfg.step_cost, 1.0))

    # Goal state (normalized)
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

    # Create a constant goal tensor for state-to-goal batches
    goal_t_const = torch.tensor(goal_state[None], device=device)

    # Dual variable (Lagrange multiplier) parameterized via softplus.
    raw_lagrange = nn.Parameter(
        torch.tensor(float(np.log(np.exp(cfg.lambda_init) - 1.0)), dtype=torch.float32, device=device)
    )

    optim_model = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    optim_lagrange = torch.optim.Adam([raw_lagrange], lr=cfg.lambda_lr)

    # Randomly sampled states in [0,1]^2 for property checks
    all_states = np.random.rand(256, 2).astype(np.float32)
    all_states_t = torch.tensor(all_states, device=device)

    pbar = trange(cfg.total_steps, desc="training")
    for step in pbar:
        if len(buffer) < cfg.batch_size:
            collect_random_transitions(env, buffer, cfg.init_transitions)

        # 1) Sample (s, s', r) for local constraint
        s, a, s2, r, done = buffer.sample(cfg.batch_size)
        s_t = torch.tensor(s, device=device)
        s2_t = torch.tensor(s2, device=device)
        r_t = torch.tensor(r, device=device)

        # 2) Global push (repo-style): apply to random state pairs to prevent collapse
        idxu = np.random.choice(len(buffer), cfg.batch_size)
        idxv = np.random.choice(len(buffer), cfg.batch_size)
        s_u = np.array([buffer.buffer[i][0] for i in idxu], dtype=np.float32)
        s_v = np.array([buffer.buffer[i][0] for i in idxv], dtype=np.float32)
        s_u_t = torch.tensor(s_u, device=device)
        s_v_t = torch.tensor(s_v, device=device)

        # 3) Triangle inequality batch
        idx1 = np.random.choice(len(buffer), cfg.batch_size)
        idx2 = np.random.choice(len(buffer), cfg.batch_size)
        idx3 = np.random.choice(len(buffer), cfg.batch_size)
        s1 = np.array([buffer.buffer[i][0] for i in idx1], dtype=np.float32)
        s2b = np.array([buffer.buffer[i][0] for i in idx2], dtype=np.float32)
        s3 = np.array([buffer.buffer[i][0] for i in idx3], dtype=np.float32)
        s1_t = torch.tensor(s1, device=device)
        s2b_t = torch.tensor(s2b, device=device)
        s3_t = torch.tensor(s3, device=device)

        # Global push uses random state-pair distances to prevent collapse
        d_pairs = model(s_u_t, s_v_t)

        d_13 = model(s1_t, s3_t)
        d_12 = model(s1_t, s2b_t)
        d_23 = model(s2b_t, s3_t)

        # Local constraint
        # Bellman-optimal local backup (4-neighbor min)
        d_sg, sq_deviation, violation = local_constraint_stats_bellman_optimal(
            model, s_t, goal_t_const, env.height, env.width,
            step_cost=cfg.step_cost, epsilon=cfg.local_epsilon
        )

        # 直接拟合 TD（比“只惩罚正 violation”稳定很多）
        loss_local = sq_deviation

        # Global push
        loss_global = global_push_loss(d_pairs, beta=cfg.global_beta, offset=cfg.global_offset)
        # Triangle inequality
        loss_tri = triangle_inequality_loss(d_13, d_12, d_23, margin=cfg.tri_margin)
        # Small scale regularizer to avoid unbounded growth without hard clamping
        loss_scale = d_pairs.square().mean()

        # --- Anchors for the open-grid sanity check ---
        # 1) Self-loss: encourages reflexivity d(s,s) -> 0
        d_self = model(s_t, s_t)
        loss_self = d_self.mean()

        # 2) One-step loss: anchors the scale for true one-step transitions.
        # NOTE: At the grid boundary, some actions get clipped and produce s' == s (cost 0).
        d_step = model(s_t, s2_t)
        sx, sy = _snap_to_grid(s_t, env.height, env.width)
        s2x, s2y = _snap_to_grid(s2_t, env.height, env.width)
        moved = ((sx != s2x) | (sy != s2y)).float()
        step_target = moved * cfg.step_cost
        loss_step = (d_step - step_target).square().mean()

        # --- Dual variable (lagrange multiplier) ---
        lagrange_mult = F.softplus(raw_lagrange)

        # NOTE: `violation` can be negative when the constraint is satisfied; in the primal objective
        # we only need to penalize *positive* violations for stability in this demo.
        lambda_const = lagrange_mult.detach()
        loss_model = (
            cfg.global_weight * loss_global
            + 1.0 * loss_local
            + cfg.self_weight * loss_self
            + cfg.step_weight * loss_step
            + cfg.tri_weight * loss_tri
            + cfg.scale_weight * loss_scale
        )

        optim_model.zero_grad()
        loss_model.backward()
        optim_model.step()

        # Dual ascent on the constraint violation (maximize lambda * violation)
        if cfg.use_lagrange and (step % cfg.lagrange_update_every == 0):
            optim_lagrange.zero_grad()
            loss_dual = -(F.softplus(raw_lagrange) * violation.detach())
            loss_dual.backward()
            optim_lagrange.step()

        loss = loss_model.detach()

        if step % 100 == 0:
            pbar.set_postfix(
                lambda_=f"{float(F.softplus(raw_lagrange).detach().cpu()):.3f}",
                sq_dev=f"{sq_deviation.item():.4f}",
                violation=f"{violation.item():.4f}",
                global_loss=f"{loss_global.item():.4f}",
                self_loss=f"{loss_self.item():.4f}",
                step_loss=f"{loss_step.item():.4f}",
                tri_loss=f"{loss_tri.item():.4f}",
                scale=f"{loss_scale.item():.4f}",
                model_loss=f"{loss_model.item():.4f}",
            )

        # Occasionally collect more random transitions
        if step > 0 and step % cfg.collect_interval == 0:
            collect_random_transitions(env, buffer, cfg.init_transitions // 2)

        # Visualization and property checks
        if step % cfg.viz_interval == 0:
            # Visualize heatmap
            visualize_quasimetric_heatmap(
                model, env, goal_state, step, cfg.fig_dir,
                resolution=cfg.heatmap_resolution
            )

            # Verify quasimetric properties
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
