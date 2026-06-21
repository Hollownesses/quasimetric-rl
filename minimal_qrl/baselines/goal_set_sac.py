from __future__ import annotations

import csv
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from minimal_qrl.envs import CommInspectionDubinsUAV2D
from minimal_qrl.gc_agents import GoalConditionedAgentBase


def _mlp(sizes: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 2):
        layers.extend([nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()])
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    return nn.Sequential(*layers)


@dataclass
class GoalSetSACConfig:
    hidden_dim: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    batch_size: int = 256
    replay_size: int = 500_000
    total_env_steps: int = 300_000
    start_random_steps: int = 5_000
    updates_per_step: int = 1
    log_interval: int = 1_000
    checkpoint_interval: int = 50_000
    log_std_min: float = -5.0
    log_std_max: float = 2.0


class GoalSetReplayBuffer:
    def __init__(
        self,
        obs_dim: int,
        goal_dim: int,
        act_dim: int,
        capacity: int,
        device: torch.device,
    ) -> None:
        self.capacity = int(capacity)
        self.device = device
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.goal = np.zeros((capacity, goal_dim), dtype=np.float32)
        self.action = np.zeros((capacity, act_dim), dtype=np.float32)
        self.reward = np.zeros((capacity,), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.float32)
        self.size = 0
        self.position = 0

    def add(
        self,
        obs: np.ndarray,
        goal: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        i = self.position
        self.obs[i] = obs
        self.goal[i] = goal
        self.action[i] = action
        self.reward[i] = float(reward)
        self.next_obs[i] = next_obs
        self.done[i] = float(done)
        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, Tensor]:
        idx = np.random.randint(0, self.size, size=int(batch_size))
        return {
            "obs": torch.as_tensor(self.obs[idx], device=self.device),
            "goal": torch.as_tensor(self.goal[idx], device=self.device),
            "action": torch.as_tensor(self.action[idx], device=self.device),
            "reward": torch.as_tensor(self.reward[idx], device=self.device),
            "next_obs": torch.as_tensor(self.next_obs[idx], device=self.device),
            "done": torch.as_tensor(self.done[idx], device=self.device),
        }


class SquashedGaussianActor(nn.Module):
    def __init__(self, obs_dim: int, goal_dim: int, act_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = _mlp([obs_dim + goal_dim, hidden_dim, hidden_dim, 2 * act_dim])

    def distribution_parameters(
        self,
        obs: Tensor,
        goal: Tensor,
        log_std_min: float,
        log_std_max: float,
    ) -> tuple[Tensor, Tensor]:
        mean, log_std = self.net(torch.cat([obs, goal], dim=-1)).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, float(log_std_min), float(log_std_max))
        return mean, log_std


class GoalSetQCritic(nn.Module):
    def __init__(self, obs_dim: int, goal_dim: int, act_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = _mlp([obs_dim + goal_dim + act_dim, hidden_dim, hidden_dim, 1])

    def forward(self, obs: Tensor, goal: Tensor, action: Tensor) -> Tensor:
        return self.net(torch.cat([obs, goal, action], dim=-1)).squeeze(-1)


class GoalSetSACAgent(GoalConditionedAgentBase):
    def __init__(
        self,
        env: CommInspectionDubinsUAV2D,
        cfg: GoalSetSACConfig,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.env = env
        self.cfg = cfg
        self.device = device
        self.obs_dim = int(env.observation_space.shape[0])
        self.goal_dim = self.obs_dim
        self.act_dim = int(env.action_space.shape[0])

        self.actor = SquashedGaussianActor(
            self.obs_dim, self.goal_dim, self.act_dim, cfg.hidden_dim
        ).to(device)
        self.q1 = GoalSetQCritic(
            self.obs_dim, self.goal_dim, self.act_dim, cfg.hidden_dim
        ).to(device)
        self.q2 = GoalSetQCritic(
            self.obs_dim, self.goal_dim, self.act_dim, cfg.hidden_dim
        ).to(device)
        self.q1_target = GoalSetQCritic(
            self.obs_dim, self.goal_dim, self.act_dim, cfg.hidden_dim
        ).to(device)
        self.q2_target = GoalSetQCritic(
            self.obs_dim, self.goal_dim, self.act_dim, cfg.hidden_dim
        ).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=cfg.critic_lr)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=cfg.critic_lr)
        self.log_alpha = torch.zeros((), device=device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
        self.target_entropy = -float(self.act_dim)
        self.action_low = torch.as_tensor(env.action_space.low, device=device)
        self.action_high = torch.as_tensor(env.action_space.high, device=device)

    @property
    def alpha(self) -> Tensor:
        return self.log_alpha.exp()

    def _sample_action(
        self,
        obs: Tensor,
        goal: Tensor,
        *,
        deterministic: bool,
    ) -> tuple[Tensor, Tensor]:
        mean, log_std = self.actor.distribution_parameters(
            obs, goal, self.cfg.log_std_min, self.cfg.log_std_max
        )
        if deterministic:
            pre_tanh = mean
        else:
            pre_tanh = mean + log_std.exp() * torch.randn_like(mean)
        unit_action = torch.tanh(pre_tanh)
        scale = 0.5 * (self.action_high - self.action_low)
        bias = 0.5 * (self.action_high + self.action_low)
        action = bias + scale * unit_action

        normal_log_prob = -0.5 * (
            ((pre_tanh - mean) / (log_std.exp() + 1e-8)) ** 2
            + 2.0 * log_std
            + np.log(2.0 * np.pi)
        )
        log_prob = normal_log_prob.sum(dim=-1)
        log_prob -= torch.log(1.0 - unit_action.pow(2) + 1e-6).sum(dim=-1)
        log_prob -= torch.log(scale.clamp_min(1e-8)).sum()
        return action, log_prob

    @torch.no_grad()
    def act(self, obs: np.ndarray, goal_obs: np.ndarray, eval_mode: bool = True) -> np.ndarray:
        obs_t = torch.as_tensor(obs[None], device=self.device, dtype=torch.float32)
        goal_t = torch.as_tensor(goal_obs[None], device=self.device, dtype=torch.float32)
        action, _ = self._sample_action(obs_t, goal_t, deterministic=bool(eval_mode))
        return action[0].cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def value(self, obs: np.ndarray, goal_obs: np.ndarray) -> float:
        obs_t = torch.as_tensor(obs[None], device=self.device, dtype=torch.float32)
        goal_t = torch.as_tensor(goal_obs[None], device=self.device, dtype=torch.float32)
        action, _ = self._sample_action(obs_t, goal_t, deterministic=True)
        value = torch.minimum(
            self.q1(obs_t, goal_t, action), self.q2(obs_t, goal_t, action)
        )
        return float(-value.item())

    def update(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        obs = batch["obs"]
        goal = batch["goal"]
        action = batch["action"]
        reward = batch["reward"]
        next_obs = batch["next_obs"]
        done = batch["done"]

        with torch.no_grad():
            next_action, next_log_prob = self._sample_action(
                next_obs, goal, deterministic=False
            )
            next_q = torch.minimum(
                self.q1_target(next_obs, goal, next_action),
                self.q2_target(next_obs, goal, next_action),
            )
            target = reward + self.cfg.gamma * (1.0 - done) * (
                next_q - self.alpha.detach() * next_log_prob
            )

        q1_loss = F.mse_loss(self.q1(obs, goal, action), target)
        q2_loss = F.mse_loss(self.q2(obs, goal, action), target)
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        policy_action, log_prob = self._sample_action(obs, goal, deterministic=False)
        policy_q = torch.minimum(
            self.q1(obs, goal, policy_action), self.q2(obs, goal, policy_action)
        )
        actor_loss = (self.alpha.detach() * log_prob - policy_q).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        with torch.no_grad():
            for source, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
                target_param.data.lerp_(source.data, self.cfg.tau)
            for source, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
                target_param.data.lerp_(source.data, self.cfg.tau)

        return {
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.item()),
        }


def save_goal_set_sac_checkpoint(
    path: Path,
    agent: GoalSetSACAgent,
    *,
    env_steps: int,
    seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "agent": agent.state_dict(),
            "config": asdict(agent.cfg),
            "env_steps": int(env_steps),
            "seed": int(seed),
            "obs_dim": int(agent.obs_dim),
            "act_dim": int(agent.act_dim),
        },
        path,
    )


def load_goal_set_sac_checkpoint(
    path: Path,
    env: CommInspectionDubinsUAV2D,
    device: torch.device,
) -> tuple[GoalSetSACAgent, Dict[str, int]]:
    payload = torch.load(path, map_location=device)
    cfg = GoalSetSACConfig(**payload.get("config", {}))
    agent = GoalSetSACAgent(env, cfg, device)
    agent.load_state_dict(payload["agent"] if "agent" in payload else payload)
    agent.eval()
    return agent, {
        "env_steps": int(payload.get("env_steps", 0)),
        "seed": int(payload.get("seed", 0)),
    }


def train_goal_set_sac(
    env: CommInspectionDubinsUAV2D,
    cfg: GoalSetSACConfig,
    device: torch.device,
    output_dir: Path,
    *,
    seed: int,
) -> GoalSetSACAgent:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(int(seed))
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    env.action_space.seed(int(seed))

    agent = GoalSetSACAgent(env, cfg, device)
    replay = GoalSetReplayBuffer(
        agent.obs_dim,
        agent.goal_dim,
        agent.act_dim,
        cfg.replay_size,
        device,
    )
    metrics_path = output_dir / "train_metrics.csv"
    with open(metrics_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["env_step", "q1_loss", "q2_loss", "actor_loss", "alpha_loss", "alpha"],
        )
        writer.writeheader()

    writer_tb = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer_tb = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
    except ImportError:
        writer_tb = None

    episode_index = 0
    obs, _ = env.reset(seed=int(seed))
    goal_obs = env.abstract_goal_observation().astype(np.float32)
    last_stats: Dict[str, float] = {}
    for env_step in range(1, int(cfg.total_env_steps) + 1):
        if env_step <= int(cfg.start_random_steps):
            action = env.action_space.sample().astype(np.float32)
        else:
            action = agent.act(obs, goal_obs, eval_mode=False)
        next_obs, reward, terminated, truncated, _info = env.step(action)
        replay.add(obs, goal_obs, action, reward, next_obs, bool(terminated))
        obs = next_obs

        if replay.size >= int(cfg.batch_size):
            for _ in range(max(1, int(cfg.updates_per_step))):
                last_stats = agent.update(replay.sample(cfg.batch_size))

        if terminated or truncated:
            episode_index += 1
            obs, _ = env.reset(seed=int(seed) + episode_index)
            goal_obs = env.abstract_goal_observation().astype(np.float32)

        if last_stats and env_step % max(1, int(cfg.log_interval)) == 0:
            row = {"env_step": int(env_step), **last_stats}
            with open(metrics_path, "a", encoding="utf-8", newline="") as f:
                csv.DictWriter(f, fieldnames=row.keys()).writerow(row)
            if writer_tb is not None:
                for key, value in last_stats.items():
                    writer_tb.add_scalar(f"train/{key}", value, env_step)

        if env_step % max(1, int(cfg.checkpoint_interval)) == 0:
            save_goal_set_sac_checkpoint(
                output_dir / f"checkpoint_{env_step}.pth",
                agent,
                env_steps=env_step,
                seed=seed,
            )

    save_goal_set_sac_checkpoint(
        output_dir / "checkpoint_final.pth",
        agent,
        env_steps=int(cfg.total_env_steps),
        seed=seed,
    )
    if writer_tb is not None:
        writer_tb.close()
    agent.eval()
    return agent
