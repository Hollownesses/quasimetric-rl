#!/usr/bin/env python3
"""
Dubins UAV 上的若干标准 TD-based goal-conditioned RL 算法实现：

- HER + DDPG
- Goal-conditioned SAC
- UVFA-style goal-conditioned value learning（纯 V(s,g)，无 quasimetric 结构约束）

这些实现刻意保持“教学版 / 研究原型”风格：
- 仅依赖 PyTorch 和本 repo 的 Dubins 环境
- 使用统一的接口，便于后续评价模块调用：
    - act(obs, goal_obs, eval_mode) -> action
    - value(obs, goal_obs) -> 标标量 V(s,g)

注意：
- 所有算法都假设 reward 结构与环境一致：每步 -dt，到达目标后 episode 结束。
- 目标表示统一采用环境的 state_to_observation(goal_state) 编码。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from minimal_qrl.envs import DubinsUAV2D
from tqdm import tqdm


Tensor = torch.Tensor


def _mlp(sizes: List[int], activation=nn.ReLU, output_activation: Optional[Callable[[], nn.Module]] = None) -> nn.Sequential:
    layers: List[nn.Module] = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if act is not None:
            layers.append(act())
    return nn.Sequential(*layers)


class GoalConditionedAgentBase(nn.Module):
    """
    统一接口：
    - act(obs, goal_obs, eval_mode) -> action
    - value(obs, goal_obs) -> 标量 V(s,g)（越小越好，近似 time-to-go）
    """

    def act(self, obs: np.ndarray, goal_obs: np.ndarray, eval_mode: bool = True) -> np.ndarray:  # pragma: no cover - 接口
        raise NotImplementedError

    def value(self, obs: np.ndarray, goal_obs: np.ndarray) -> float:  # pragma: no cover - 接口
        raise NotImplementedError

    def batch_value(self, obs_batch: np.ndarray, goal_obs_batch: np.ndarray) -> np.ndarray:  # pragma: no cover - 默认回退
        obs_batch = np.asarray(obs_batch, dtype=np.float32)
        goal_obs_batch = np.asarray(goal_obs_batch, dtype=np.float32)
        vals = [self.value(obs_batch[i], goal_obs_batch[i]) for i in range(len(obs_batch))]
        return np.asarray(vals, dtype=np.float32)


class QRLGoalValueAdapter(GoalConditionedAgentBase):
    """
    将已有 QRL Agent 适配为 GoalConditionedAgentBase：
    - 使用第一个 critic 计算 d(s,g)，避免评估时重复整套 critic 推理
    - 可显式指定 distance_scale；未指定时沿用环境的距离尺度

    ``distance_scale`` 的默认行为是为旧的固定 step-cost Dubins 实验保留的。
    对直接以环境代价训练的 QRL（例如通信巡检的 negative_reward 训练），
    应显式传入 ``distance_scale=1.0``。
    """

    def __init__(
        self,
        qrl_agent: nn.Module,
        env: DubinsUAV2D,
        device: torch.device,
        *,
        distance_scale: Optional[float] = None,
    ):
        super().__init__()
        self.qrl_agent = qrl_agent
        self.env = env
        self.device = device

        if distance_scale is None:
            self._scale = 1.0
            u = getattr(env, "unwrapped", env)
            if hasattr(u, "get_distance_scale"):
                self._scale = float(u.get_distance_scale())
        else:
            self._scale = float(distance_scale)
        if not np.isfinite(self._scale) or self._scale <= 0.0:
            raise ValueError(f"distance_scale must be finite and positive, got {self._scale}")

    @torch.no_grad()
    def value(self, obs: np.ndarray, goal_obs: np.ndarray) -> float:
        return float(self.batch_value(
            np.asarray(obs, dtype=np.float32)[None, :],
            np.asarray(goal_obs, dtype=np.float32)[None, :],
        )[0])

    @torch.no_grad()
    def batch_value(self, obs_batch: np.ndarray, goal_obs_batch: np.ndarray) -> np.ndarray:
        obs_batch = np.asarray(obs_batch, dtype=np.float32)
        goal_obs_batch = np.asarray(goal_obs_batch, dtype=np.float32)
        if obs_batch.ndim != 2 or goal_obs_batch.ndim != 2:
            raise ValueError("obs_batch and goal_obs_batch must both be rank-2 arrays")
        if len(obs_batch) != len(goal_obs_batch):
            raise ValueError("obs_batch and goal_obs_batch must have the same batch size")
        if len(obs_batch) == 0:
            return np.empty((0,), dtype=np.float32)

        critic = self.qrl_agent.critics[0]
        o = torch.as_tensor(obs_batch, device=self.device, dtype=torch.float32)
        g = torch.as_tensor(goal_obs_batch, device=self.device, dtype=torch.float32)
        z_g = critic.encoder(g)
        distance = critic.quasimetric_model(critic.encoder(o), z_g).reshape(-1)
        values = distance.detach().cpu().numpy().astype(np.float32)
        return values * float(self._scale)

    @torch.no_grad()
    def act(self, obs: np.ndarray, goal_obs: np.ndarray, eval_mode: bool = True) -> np.ndarray:
        """
        QRL 本身不包含直接 policy；为了在“成功率”指标中参与对比，
        这里提供一个简单的基于 QRL value 的贪心控制策略：
        - 在动作空间内均匀采样若干角速度
        - 选择使下一状态到 goal 的 d(s',g) 最小的动作
        """
        env: DubinsUAV2D = self.env
        # 动作空间为 1 维 Box
        low = float(env.action_space.low[0])
        high = float(env.action_space.high[0])
        n_candidates = 21 if eval_mode else 11
        omegas = np.linspace(low, high, n_candidates, dtype=np.float32)

        # 当前内部状态（使用 env.state，比 obs 更精确）
        state = env.state.copy()
        valid_actions = []
        valid_next_observations = []
        for w in omegas:
            # 单步 roll-out 近似下一状态（复用 Dubins 动力学）
            theta_new = env._normalize_angle(state[2] + w * env.dt)
            x_new = state[0] + env.v * np.cos(theta_new) * env.dt
            y_new = state[1] + env.v * np.sin(theta_new) * env.dt
            next_state = np.array([x_new, y_new, theta_new], dtype=np.float32)
            # 越界/碰撞直接跳过
            if not env.is_valid_state(next_state):
                continue
            valid_actions.append(float(w))
            valid_next_observations.append(env.state_to_observation(next_state))

        if not valid_actions:
            return np.array([0.0], dtype=np.float32)
        obs_batch = np.asarray(valid_next_observations, dtype=np.float32)
        goal_batch = np.repeat(np.asarray(goal_obs, dtype=np.float32)[None, :], len(valid_actions), axis=0)
        values = self.batch_value(obs_batch, goal_batch)
        best_index = int(np.argmin(values))
        return np.array([valid_actions[best_index]], dtype=np.float32)


class GCActor(nn.Module):
    """
    goal-conditioned 连续控制 actor：输入 [obs, goal_obs] 输出 action in [-1,1]，再按 env bound 缩放。
    """

    def __init__(self, obs_dim: int, goal_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = _mlp([obs_dim + goal_dim, hidden, hidden, act_dim], activation=nn.ReLU, output_activation=nn.Tanh)

    def forward(self, obs: Tensor, goal: Tensor) -> Tensor:
        x = torch.cat([obs, goal], dim=-1)
        return self.net(x)


class GCQCritic(nn.Module):
    """
    Q(s,a,g) critic：输入 [obs, goal_obs, action] -> 标量。
    """

    def __init__(self, obs_dim: int, goal_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = _mlp([obs_dim + goal_dim + act_dim, hidden, hidden, 1], activation=nn.ReLU, output_activation=None)

    def forward(self, obs: Tensor, goal: Tensor, act: Tensor) -> Tensor:
        x = torch.cat([obs, goal, act], dim=-1)
        return self.net(x).squeeze(-1)


class GCVValue(nn.Module):
    """
    UVFA-style 价值网络：V(s,g)。
    """

    def __init__(self, obs_dim: int, goal_dim: int, hidden: int = 256):
        super().__init__()
        self.net = _mlp([obs_dim + goal_dim, hidden, hidden, 1], activation=nn.ReLU, output_activation=None)

    def forward(self, obs: Tensor, goal: Tensor) -> Tensor:
        x = torch.cat([obs, goal], dim=-1)
        return self.net(x).squeeze(-1)


@dataclass
class AlgoConfig:
    """
    训练超参数（对三种算法统一管理，部分字段按需使用）。
    """

    total_env_steps: int = 200_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005  # target smoothing
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    value_lr: float = 3e-4  # UVFA

    # exploration
    action_noise_std: float = 0.2  # DDPG
    start_random_steps: int = 1_000

    # HER
    her_k: int = 4  # 每条轨迹采样多少个 future goals

    # SAC
    sac_alpha: float = 0.2
    target_entropy: Optional[float] = None  # 若为 None，则使用 -act_dim

    # logging / eval 钩子
    log_interval: int = 10_000


class ReplayBuffer:
    """
    简单统一的 replay buffer：支持 goal-conditioned 转移。
    """

    def __init__(self, obs_dim: int, goal_dim: int, act_dim: int, size: int = 1_000_000, device: torch.device = torch.device("cpu")):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.goal_buf = np.zeros((size, goal_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros((size,), dtype=np.float32)
        self.done_buf = np.zeros((size,), dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.max_size = size
        self.device = device

    def add(self, obs: np.ndarray, goal: np.ndarray, act: np.ndarray, rew: float, next_obs: np.ndarray, done: bool) -> None:
        i = self.ptr
        self.obs_buf[i] = obs
        self.goal_buf[i] = goal
        self.act_buf[i] = act
        self.rew_buf[i] = rew
        self.next_obs_buf[i] = next_obs
        self.done_buf[i] = float(done)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> Dict[str, Tensor]:
        assert self.size >= batch_size
        idxs = np.random.randint(0, self.size, size=batch_size)
        obs = torch.tensor(self.obs_buf[idxs], device=self.device, dtype=torch.float32)
        goal = torch.tensor(self.goal_buf[idxs], device=self.device, dtype=torch.float32)
        act = torch.tensor(self.act_buf[idxs], device=self.device, dtype=torch.float32)
        rew = torch.tensor(self.rew_buf[idxs], device=self.device, dtype=torch.float32)
        next_obs = torch.tensor(self.next_obs_buf[idxs], device=self.device, dtype=torch.float32)
        done = torch.tensor(self.done_buf[idxs], device=self.device, dtype=torch.float32)
        return dict(obs=obs, goal=goal, act=act, rew=rew, next_obs=next_obs, done=done)


def _extract_goal_obs(env: DubinsUAV2D) -> np.ndarray:
    """从环境中读取当前 episode 的 goal，并转为 goal_obs。"""
    assert env.goal is not None, "DubinsUAV2D.goal 尚未设置，请先 reset 环境"
    g_state = np.asarray(env.goal, dtype=np.float32)
    return env.state_to_observation(g_state)


def _is_goal_achieved(env: DubinsUAV2D, state: np.ndarray, goal_state: np.ndarray) -> bool:
    """复用环境内部的终止判定逻辑，用于 HER goal relabel。"""
    x, y, theta = float(state[0]), float(state[1]), float(state[2])
    gx, gy, gtheta = float(goal_state[0]), float(goal_state[1]), float(goal_state[2])
    pos_dist = np.sqrt((x - gx) ** 2 + (y - gy) ** 2)
    theta_diff = abs(env._normalize_angle(theta - gtheta))
    return (pos_dist < env.epsilon_pos) and (theta_diff < env.epsilon_theta)


def _is_task_success_for_goal(env: DubinsUAV2D, state: np.ndarray, goal_state: np.ndarray) -> bool:
    """Goal reaching check with the communication-inspection task feasibility gate when present."""
    achieved = _is_goal_achieved(env, state, goal_state)
    if not achieved:
        return False
    if hasattr(env, "is_task_feasible"):
        return bool(env.is_task_feasible(state))
    return True


def _future_goal_candidates(env: DubinsUAV2D, future_states: np.ndarray) -> np.ndarray:
    """For comm-aware HER, only relabel to future states that satisfy the task constraints."""
    if hasattr(env, "is_task_feasible"):
        feasible = [bool(env.is_task_feasible(s)) for s in future_states]
        future_states = future_states[np.asarray(feasible, dtype=bool)]
    return future_states


def _her_relabel_episode(
    env: DubinsUAV2D,
    episode_states: List[np.ndarray],
    episode_obs: List[np.ndarray],
    episode_actions: List[np.ndarray],
    episode_rewards: List[float],
    cfg: AlgoConfig,
    buf: ReplayBuffer,
) -> None:
    """
    基于 future strategy 的简单 HER：
    - 对每个时间步 t，随机选取最多 k 个未来时刻 j>t 的状态 s_j 作为新目标 g'
    - 在新目标下，reward 恒为 -dt，done 由是否在该步达到 g' 决定。

    这里使用内部状态 (x,y,theta) 作为 goal state，再通过 env.state_to_observation 得到 goal_obs。
    """
    T = len(episode_states) - 1  # 有 T 个转移
    dt = env.dt

    for t in range(T):
        s_t = episode_states[t]
        s_tp1 = episode_states[t + 1]
        obs_t = episode_obs[t]
        obs_tp1 = episode_obs[t + 1]
        a_t = episode_actions[t]

        # 1. 原始目标（环境自带 goal）
        g_state = np.asarray(env.goal, dtype=np.float32)
        g_obs = env.state_to_observation(g_state)
        done = _is_task_success_for_goal(env, s_tp1, g_state)
        buf.add(obs_t, g_obs, a_t, episode_rewards[t], obs_tp1, done)

        # 2. HER 目标
        future_idxs = np.arange(t + 1, T + 1, dtype=np.int64)
        if len(future_idxs) == 0:
            continue
        future_states = _future_goal_candidates(env, np.asarray([episode_states[j] for j in future_idxs], dtype=np.float32))
        if len(future_states) == 0:
            continue
        np.random.shuffle(future_idxs)
        selected = future_states[np.random.choice(len(future_states), size=min(cfg.her_k, len(future_states)), replace=False)]
        for g_state_her in selected:
            g_obs_her = env.state_to_observation(g_state_her)
            if hasattr(env, "compute_step_terms"):
                r_her = float(episode_rewards[t])
            else:
                r_her = -dt
            done_her = _is_task_success_for_goal(env, s_tp1, g_state_her)
            buf.add(obs_t, g_obs_her, a_t, r_her, obs_tp1, done_her)


class HERDDPGAgent(GoalConditionedAgentBase):
    """
    HER + DDPG 在 DubinsUAV2D 上的最小实现。
    - policy: 高斯噪声探索 + deterministic actor
    - critic: Q(s,a,g)
    """

    def __init__(
        self,
        env: DubinsUAV2D,
        cfg: AlgoConfig,
        device: torch.device,
    ):
        super().__init__()
        self.env = env
        self.cfg = cfg
        self.device = device

        obs_dim = int(env.observation_space.shape[0])
        goal_dim = int(env.observation_space.shape[0])  # 使用相同编码
        act_dim = int(env.action_space.shape[0])

        self.obs_dim = obs_dim
        self.goal_dim = goal_dim
        self.act_dim = act_dim

        # 网络
        self.actor = GCActor(obs_dim, goal_dim, act_dim).to(device)
        self.actor_target = GCActor(obs_dim, goal_dim, act_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = GCQCritic(obs_dim, goal_dim, act_dim).to(device)
        self.critic_target = GCQCritic(obs_dim, goal_dim, act_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        self.replay = ReplayBuffer(obs_dim, goal_dim, act_dim, size=200_000, device=device)

        # 动作 bound
        self.act_low = torch.tensor(env.action_space.low, device=device, dtype=torch.float32)
        self.act_high = torch.tensor(env.action_space.high, device=device, dtype=torch.float32)

    @torch.no_grad()
    def act(self, obs: np.ndarray, goal_obs: np.ndarray, eval_mode: bool = True) -> np.ndarray:
        o = torch.tensor(obs[None], device=self.device, dtype=torch.float32)
        g = torch.tensor(goal_obs[None], device=self.device, dtype=torch.float32)
        a = self.actor(o, g)
        # 将 [-1,1] 缩放到 env action 区间
        a = (self.act_low + (a + 1.0) * 0.5 * (self.act_high - self.act_low)).squeeze(0)
        if not eval_mode:
            noise = torch.randn_like(a) * self.cfg.action_noise_std
            a = a + noise
        a = torch.clamp(a, self.act_low, self.act_high)
        return a.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def value(self, obs: np.ndarray, goal_obs: np.ndarray) -> float:
        """
        使用 Q(s, π(s,g), g) 作为 V(s,g) 的 proxy，并取绝对值（cost-to-go 期望为正）。
        reward 为 -dt，因此较小（更接近 0）的 Q 表示更短时间。
        """
        o = torch.tensor(obs[None], device=self.device, dtype=torch.float32)
        g = torch.tensor(goal_obs[None], device=self.device, dtype=torch.float32)
        a = self.actor(o, g)
        a = self._scale_action(a)
        q = self.critic(o, g, a).cpu().item()
        return float(-q)  # 把“回报”翻转为“代价”近似

    def _scale_action(self, a: Tensor) -> Tensor:
        return self.act_low + (a + 1.0) * 0.5 * (self.act_high - self.act_low)

    def _update(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        obs = batch["obs"]
        goal = batch["goal"]
        act = batch["act"]
        rew = batch["rew"]
        next_obs = batch["next_obs"]
        done = batch["done"]

        # critic 更新
        with torch.no_grad():
            next_a = self._scale_action(self.actor_target(next_obs, goal))
            target_q = self.critic_target(next_obs, goal, next_a)
            y = rew + self.cfg.gamma * (1.0 - done) * target_q
        q = self.critic(obs, goal, act)
        critic_loss = F.mse_loss(q, y)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # actor 更新：max Q(s, π(s,g), g)
        a_pi = self._scale_action(self.actor(obs, goal))
        actor_loss = -self.critic(obs, goal, a_pi).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # target 网络软更新
        with torch.no_grad():
            for p, p_targ in zip(self.critic.parameters(), self.critic_target.parameters()):
                p_targ.data.mul_(1.0 - self.cfg.tau).add_(self.cfg.tau * p.data)
            for p, p_targ in zip(self.actor.parameters(), self.actor_target.parameters()):
                p_targ.data.mul_(1.0 - self.cfg.tau).add_(self.cfg.tau * p.data)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
        }


class GCSACAgent(GoalConditionedAgentBase):
    """
    Goal-conditioned SAC：
    - 双 Q + 状态价值网络（简化版，如果需要也可以省略 V）
    - policy: squashed Gaussian
    """

    def __init__(self, env: DubinsUAV2D, cfg: AlgoConfig, device: torch.device):
        super().__init__()
        self.env = env
        self.cfg = cfg
        self.device = device

        obs_dim = int(env.observation_space.shape[0])
        goal_dim = int(env.observation_space.shape[0])
        act_dim = int(env.action_space.shape[0])

        self.obs_dim = obs_dim
        self.goal_dim = goal_dim
        self.act_dim = act_dim

        self.actor = GCActor(obs_dim, goal_dim, act_dim).to(device)

        self.q1 = GCQCritic(obs_dim, goal_dim, act_dim).to(device)
        self.q2 = GCQCritic(obs_dim, goal_dim, act_dim).to(device)
        self.q1_targ = GCQCritic(obs_dim, goal_dim, act_dim).to(device)
        self.q2_targ = GCQCritic(obs_dim, goal_dim, act_dim).to(device)
        self.q1_targ.load_state_dict(self.q1.state_dict())
        self.q2_targ.load_state_dict(self.q2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=cfg.critic_lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=cfg.critic_lr)

        # 温度参数（显式 float32，避免 MPS 上 float64 不支持）
        self.log_alpha = torch.tensor(
            float(np.log(cfg.sac_alpha)), device=device, dtype=torch.float32, requires_grad=True
        )
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.critic_lr)
        self.target_entropy = cfg.target_entropy if cfg.target_entropy is not None else -float(act_dim)

        self.replay = ReplayBuffer(obs_dim, goal_dim, act_dim, size=200_000, device=device)
        self.act_low = torch.tensor(env.action_space.low, device=device, dtype=torch.float32)
        self.act_high = torch.tensor(env.action_space.high, device=device, dtype=torch.float32)

    @property
    def alpha(self) -> Tensor:
        return self.log_alpha.exp()

    def _sample_action(self, obs: Tensor, goal: Tensor) -> Tuple[Tensor, Tensor]:
        """
        SAC policy: squashed Gaussian。这里简单使用 actor 输出 mean，再固定 log_std。
        """
        mean = self.actor(obs, goal)
        log_std = torch.zeros_like(mean)
        std = log_std.exp()
        noise = torch.randn_like(mean)
        pre_tanh = mean + noise * std
        a = torch.tanh(pre_tanh)
        # log_prob（squash correction）
        log_prob = (
            -0.5 * ((pre_tanh - mean) / (std + 1e-6)) ** 2
            - log_std
            - 0.5 * np.log(2 * np.pi)
        ).sum(dim=-1, keepdim=True)
        log_prob -= torch.log(1.0 - a.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        # scale to env bounds
        a_scaled = self.act_low + (a + 1.0) * 0.5 * (self.act_high - self.act_low)
        return a_scaled, log_prob.squeeze(-1)

    @torch.no_grad()
    def act(self, obs: np.ndarray, goal_obs: np.ndarray, eval_mode: bool = True) -> np.ndarray:
        o = torch.tensor(obs[None], device=self.device, dtype=torch.float32)
        g = torch.tensor(goal_obs[None], device=self.device, dtype=torch.float32)
        if eval_mode:
            a = torch.tanh(self.actor(o, g))
            a = self.act_low + (a + 1.0) * 0.5 * (self.act_high - self.act_low)
        else:
            a, _ = self._sample_action(o, g)
        return a.squeeze(0).cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def value(self, obs: np.ndarray, goal_obs: np.ndarray) -> float:
        o = torch.tensor(obs[None], device=self.device, dtype=torch.float32)
        g = torch.tensor(goal_obs[None], device=self.device, dtype=torch.float32)
        a, logp = self._sample_action(o, g)
        q1 = self.q1(o, g, a)
        q2 = self.q2(o, g, a)
        q = torch.min(q1, q2) - self.alpha * logp
        return float(-q.cpu().item())

    def _update(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        obs = batch["obs"]
        goal = batch["goal"]
        act = batch["act"]
        rew = batch["rew"]
        next_obs = batch["next_obs"]
        done = batch["done"]

        # 1. 更新 Q
        with torch.no_grad():
            next_a, next_logp = self._sample_action(next_obs, goal)
            q1_t = self.q1_targ(next_obs, goal, next_a)
            q2_t = self.q2_targ(next_obs, goal, next_a)
            q_targ_min = torch.min(q1_t, q2_t)
            backup = rew + self.cfg.gamma * (1.0 - done) * (q_targ_min - self.alpha * next_logp)

        q1 = self.q1(obs, goal, act)
        q2 = self.q2(obs, goal, act)
        q1_loss = F.mse_loss(q1, backup)
        q2_loss = F.mse_loss(q2, backup)

        self.q1_opt.zero_grad()
        q1_loss.backward()
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        q2_loss.backward()
        self.q2_opt.step()

        # 2. 更新 actor（最小化 α logπ - Q）
        a_pi, logp_pi = self._sample_action(obs, goal)
        q1_pi = self.q1(obs, goal, a_pi)
        q2_pi = self.q2(obs, goal, a_pi)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha * logp_pi - q_pi).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # 3. 更新 α
        alpha_loss = -(self.log_alpha * (logp_pi.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # 4. target network
        with torch.no_grad():
            for p, p_t in zip(self.q1.parameters(), self.q1_targ.parameters()):
                p_t.data.mul_(1.0 - self.cfg.tau).add_(self.cfg.tau * p.data)
            for p, p_t in zip(self.q2.parameters(), self.q2_targ.parameters()):
                p_t.data.mul_(1.0 - self.cfg.tau).add_(self.cfg.tau * p.data)

        return {
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self.alpha.item()),
        }


class UVFAValueAgent(GoalConditionedAgentBase):
    """
    仅学习 V(s,g) 的 UVFA：
    - 使用 TD(0) 回归：V(s,g) ≈ r + γ V(s',g)
    - 不训练 policy，评估时不参与成功率（或使用简单“指向目标”的手工 controller）
    """

    def __init__(self, env: DubinsUAV2D, cfg: AlgoConfig, device: torch.device):
        super().__init__()
        self.env = env
        self.cfg = cfg
        self.device = device

        obs_dim = int(env.observation_space.shape[0])
        goal_dim = int(env.observation_space.shape[0])
        self.obs_dim = obs_dim
        self.goal_dim = goal_dim

        self.v = GCVValue(obs_dim, goal_dim).to(device)
        self.v_targ = GCVValue(obs_dim, goal_dim).to(device)
        self.v_targ.load_state_dict(self.v.state_dict())
        self.opt = torch.optim.Adam(self.v.parameters(), lr=cfg.value_lr)

        self.replay = ReplayBuffer(obs_dim, goal_dim, act_dim=1, size=200_000, device=device)

    @torch.no_grad()
    def value(self, obs: np.ndarray, goal_obs: np.ndarray) -> float:
        o = torch.tensor(obs[None], device=self.device, dtype=torch.float32)
        g = torch.tensor(goal_obs[None], device=self.device, dtype=torch.float32)
        v = self.v(o, g).cpu().item()
        return float(-v)  # reward 为 -dt，翻转为“代价”近似

    @torch.no_grad()
    def act(self, obs: np.ndarray, goal_obs: np.ndarray, eval_mode: bool = True) -> np.ndarray:
        """
        UVFA 不自带 policy，这里给一个非常简单的 hand-crafted controller：
        - 始终将朝向朝向目标点的直线
        - 使用比例控制计算角速度
        仅用于成功率评估中给 UVFA 提供一个可运行的 baseline。
        """
        env: DubinsUAV2D = self.env
        state = env.state.copy()
        gx, gy, _ = env.goal
        dx, dy = gx - state[0], gy - state[1]
        target_theta = np.arctan2(dy, dx)
        err = env._normalize_angle(target_theta - state[2])
        omega = float(np.clip(err * 2.0, -env.omega_max, env.omega_max))
        return np.array([omega], dtype=np.float32)

    def _update(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        obs = batch["obs"]
        goal = batch["goal"]
        rew = batch["rew"]
        next_obs = batch["next_obs"]
        done = batch["done"]

        with torch.no_grad():
            v_next = self.v_targ(next_obs, goal)
            target = rew + self.cfg.gamma * (1.0 - done) * v_next

        v = self.v(obs, goal)
        loss = F.mse_loss(v, target)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # target 更新
        with torch.no_grad():
            for p, p_t in zip(self.v.parameters(), self.v_targ.parameters()):
                p_t.data.mul_(1.0 - self.cfg.tau).add_(self.cfg.tau * p.data)

        return {"value_loss": float(loss.item())}


def train_td_agent(
    algo: str,
    env: DubinsUAV2D,
    cfg: AlgoConfig,
    device: torch.device,
    train_goal_radius: Optional[float] = None,
    log_fn: Optional[Callable[[int, Dict[str, float]], None]] = None,
) -> GoalConditionedAgentBase:
    """
    在 DubinsUAV2D 上训练一个 TD-based goal-conditioned RL Agent。
    - algo: "her_ddpg" / "gc_sac" / "uvfa"
    - 使用统一的 total_env_steps 作为训练预算
    """
    if algo == "her_ddpg":
        agent: GoalConditionedAgentBase = HERDDPGAgent(env, cfg, device)
        is_her = True
    elif algo == "gc_sac":
        agent = GCSACAgent(env, cfg, device)
        is_her = False
    elif algo == "uvfa":
        agent = UVFAValueAgent(env, cfg, device)
        is_her = False
    else:
        raise ValueError(f"未知算法: {algo}")

    total_steps = cfg.total_env_steps
    obs_dim = agent.obs_dim if hasattr(agent, "obs_dim") else int(env.observation_space.shape[0])

    step = 0
    episode = 0

    # 用于 OOD 训练划分：可选限制 goal 半径在 r<=train_goal_radius
    x_min, y_min, x_max, y_max = env.bounds
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    center = np.array([cx, cy], dtype=np.float32)

    def _goal_in_train_region(goal_state: np.ndarray) -> bool:
        if train_goal_radius is None:
            return True
        r = np.linalg.norm(goal_state[:2] - center)
        return r <= float(train_goal_radius)

    pbar = tqdm(total=total_steps, desc=f"train {algo}", unit="step", leave=True)
    while step < total_steps:
        # reset，生成新的 start, goal
        # 若设置了 train_goal_radius，则重复 reset，直到 goal 落在训练半径内
        for _ in range(1000):
            obs, _ = env.reset()
            g_state = np.asarray(env.goal, dtype=np.float32)
            if _goal_in_train_region(g_state):
                break
        goal_obs = _extract_goal_obs(env)

        # 记录整条轨迹（用于 HER）
        ep_states: List[np.ndarray] = [env.state.copy()]
        ep_obs: List[np.ndarray] = [obs.copy()]
        ep_actions: List[np.ndarray] = []
        ep_rewards: List[float] = []

        done = False
        truncated = False

        while not (done or truncated) and step < total_steps:
            eval_mode = False
            if step < cfg.start_random_steps and algo in ("her_ddpg", "gc_sac"):
                act = env.action_space.sample().astype(np.float32)
            else:
                act = agent.act(obs, goal_obs, eval_mode=eval_mode)

            next_obs, reward, done, truncated, info = env.step(act)

            # 填充 buffer
            if isinstance(agent, UVFAValueAgent):
                agent.replay.add(obs, goal_obs, act=np.zeros((1,), dtype=np.float32), rew=reward, next_obs=next_obs, done=done)
            else:
                agent.replay.add(obs, goal_obs, act, reward, next_obs, done)

            ep_states.append(env.state.copy())
            ep_obs.append(next_obs.copy())
            ep_actions.append(act.copy())
            ep_rewards.append(float(reward))

            obs = next_obs
            step += 1
            pbar.update(1)

            # 每一步尝试更新
            if agent.replay.size >= cfg.batch_size:
                batch = agent.replay.sample(cfg.batch_size)
                if isinstance(agent, HERDDPGAgent):
                    stats = agent._update(batch)
                elif isinstance(agent, GCSACAgent):
                    stats = agent._update(batch)
                elif isinstance(agent, UVFAValueAgent):
                    stats = agent._update(batch)
                else:
                    stats = {}
                if log_fn is not None and step % cfg.log_interval == 0:
                    log_fn(step, stats)

        # episode 结束后，对 HER 进行额外 relabel
        episode += 1
        if is_her and len(ep_states) >= 2:
            _her_relabel_episode(env, ep_states, ep_obs, ep_actions, ep_rewards, cfg, agent.replay)

    pbar.close()
    return agent
