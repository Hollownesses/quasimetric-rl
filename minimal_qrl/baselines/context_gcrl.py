from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from minimal_qrl.envs import CommInspectionDubinsUAV2D
from minimal_qrl.gc_agents import GoalConditionedAgentBase


def _mlp(sizes: List[int], *, output_activation: Optional[nn.Module] = None) -> nn.Sequential:
    layers: List[nn.Module] = []
    for index in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[index], sizes[index + 1]))
        if index < len(sizes) - 2:
            layers.append(nn.ReLU())
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


def parameter_count(module: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


@dataclass
class ContextGCRLConfig:
    hidden_dim: int = 256
    representation_dim: int = 64
    residual_dim: int = 64
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
    action_noise_std: float = 0.2
    her_k: int = 4
    teacher_ratio: float = 1.0
    log_interval: int = 1_000
    checkpoint_interval: int = 50_000
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    contrastive_temperature: float = 1.0


class DeterministicGoalActor(nn.Module):
    def __init__(self, obs_dim: int, goal_dim: int, act_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = _mlp(
            [obs_dim + goal_dim, hidden_dim, hidden_dim, act_dim],
            output_activation=nn.Tanh(),
        )

    def forward(self, obs: Tensor, goal: Tensor) -> Tensor:
        return self.net(torch.cat([obs, goal], dim=-1))


class MonolithicGoalCritic(nn.Module):
    def __init__(self, obs_dim: int, goal_dim: int, act_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = _mlp([obs_dim + goal_dim + act_dim, hidden_dim, hidden_dim, 1])

    def forward(self, obs: Tensor, action: Tensor, goal: Tensor) -> Tensor:
        return self.net(torch.cat([obs, action, goal], dim=-1)).squeeze(-1)


class MRNGoalCritic(nn.Module):
    """Metric Residual Network critic using an L2 metric plus asymmetric residual."""

    def __init__(
        self,
        obs_dim: int,
        goal_dim: int,
        act_dim: int,
        hidden_dim: int,
        representation_dim: int,
        residual_dim: int,
    ) -> None:
        super().__init__()
        self.state_action_encoder = _mlp(
            [obs_dim + act_dim, hidden_dim, representation_dim]
        )
        self.state_goal_encoder = _mlp(
            [obs_dim + goal_dim, hidden_dim, representation_dim]
        )
        self.metric_projection = nn.Linear(representation_dim, representation_dim)
        self.residual_projection = nn.Linear(representation_dim, residual_dim)

    def distance_from_embeddings(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        metric = torch.linalg.vector_norm(
            self.metric_projection(x) - self.metric_projection(y), ord=2, dim=-1
        )
        hx = self.residual_projection(x)
        hy = self.residual_projection(y)
        residual = torch.relu(hx - hy).amax(dim=-1)
        return metric + residual, metric, residual

    def components(self, obs: Tensor, action: Tensor, goal: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x = self.state_action_encoder(torch.cat([obs, action], dim=-1))
        y = self.state_goal_encoder(torch.cat([obs, goal], dim=-1))
        return self.distance_from_embeddings(x, y)

    def forward(self, obs: Tensor, action: Tensor, goal: Tensor) -> Tensor:
        distance, _metric, _residual = self.components(obs, action, goal)
        return -distance


def _closest_mrn_hidden_dim(
    obs_dim: int,
    goal_dim: int,
    act_dim: int,
    cfg: ContextGCRLConfig,
) -> int:
    reference = MonolithicGoalCritic(
        obs_dim, goal_dim, act_dim, cfg.hidden_dim
    )
    target_count = parameter_count(reference)
    candidates = list(range(8, 513, 8))
    return min(
        candidates,
        key=lambda width: abs(
            parameter_count(MRNGoalCritic(
                obs_dim,
                goal_dim,
                act_dim,
                width,
                cfg.representation_dim,
                cfg.residual_dim,
            )) - target_count
        ),
    )


class _DDPGBase(GoalConditionedAgentBase):
    def __init__(
        self,
        env: CommInspectionDubinsUAV2D,
        cfg: ContextGCRLConfig,
        device: torch.device,
        *,
        critic_kind: str,
    ) -> None:
        super().__init__()
        self.env = env
        self.cfg = cfg
        self.device = torch.device(device)
        self.obs_dim = int(env.observation_space.shape[0])
        self.goal_dim = self.obs_dim
        self.act_dim = int(env.action_space.shape[0])
        self.action_low = torch.as_tensor(env.action_space.low, device=self.device)
        self.action_high = torch.as_tensor(env.action_space.high, device=self.device)
        self.actor = DeterministicGoalActor(
            self.obs_dim, self.goal_dim, self.act_dim, cfg.hidden_dim
        ).to(self.device)
        self.actor_target = DeterministicGoalActor(
            self.obs_dim, self.goal_dim, self.act_dim, cfg.hidden_dim
        ).to(self.device)
        self.critic_kind = str(critic_kind)
        self.mrn_hidden_dim = 0
        if critic_kind == "monolithic":
            critic_factory = lambda: MonolithicGoalCritic(
                self.obs_dim, self.goal_dim, self.act_dim, cfg.hidden_dim
            )
        elif critic_kind == "mrn":
            self.mrn_hidden_dim = _closest_mrn_hidden_dim(
                self.obs_dim, self.goal_dim, self.act_dim, cfg
            )
            critic_factory = lambda: MRNGoalCritic(
                self.obs_dim,
                self.goal_dim,
                self.act_dim,
                self.mrn_hidden_dim,
                cfg.representation_dim,
                cfg.residual_dim,
            )
        else:
            raise ValueError(f"unknown critic kind: {critic_kind}")
        self.critic = critic_factory().to(self.device)
        self.critic_target = critic_factory().to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

    def _scale_action(self, unit_action: Tensor) -> Tensor:
        return self.action_low + 0.5 * (unit_action + 1.0) * (
            self.action_high - self.action_low
        )

    def _actor_action(self, obs: Tensor, goal: Tensor, *, target: bool = False) -> Tensor:
        actor = self.actor_target if target else self.actor
        return self._scale_action(actor(obs, goal))

    @torch.no_grad()
    def act(self, obs: np.ndarray, goal_obs: np.ndarray, eval_mode: bool = True) -> np.ndarray:
        obs_t = torch.as_tensor(obs[None], device=self.device, dtype=torch.float32)
        goal_t = torch.as_tensor(goal_obs[None], device=self.device, dtype=torch.float32)
        action = self._actor_action(obs_t, goal_t)[0]
        if not eval_mode:
            scale = 0.5 * (self.action_high - self.action_low)
            action = action + torch.randn_like(action) * self.cfg.action_noise_std * scale
        action = torch.maximum(torch.minimum(action, self.action_high), self.action_low)
        return action.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def batch_value(self, obs_batch: np.ndarray, goal_obs_batch: np.ndarray) -> np.ndarray:
        obs = torch.as_tensor(obs_batch, device=self.device, dtype=torch.float32)
        goal = torch.as_tensor(goal_obs_batch, device=self.device, dtype=torch.float32)
        action = self._actor_action(obs, goal)
        return (-self.critic(obs, action, goal)).cpu().numpy().astype(np.float32)

    def value(self, obs: np.ndarray, goal_obs: np.ndarray) -> float:
        return float(self.batch_value(obs[None], goal_obs[None])[0])

    def update(self, batch: Dict[str, Tensor | List[str]]) -> Dict[str, float]:
        obs = batch["obs"]
        goal = batch["goal"]
        action = batch["action"]
        reward = batch["reward"]
        next_obs = batch["next_obs"]
        done = batch["done"]
        assert isinstance(obs, Tensor) and isinstance(goal, Tensor)
        assert isinstance(action, Tensor) and isinstance(reward, Tensor)
        assert isinstance(next_obs, Tensor) and isinstance(done, Tensor)
        with torch.no_grad():
            next_action = self._actor_action(next_obs, goal, target=True)
            target_q = self.critic_target(next_obs, next_action, goal)
            backup = reward + self.cfg.gamma * (1.0 - done) * target_q
        predicted_q = self.critic(obs, action, goal)
        critic_loss = F.mse_loss(predicted_q, backup)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        policy_action = self._actor_action(obs, goal)
        actor_loss = -self.critic(obs, policy_action, goal).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        with torch.no_grad():
            for source, target in zip(self.actor.parameters(), self.actor_target.parameters()):
                target.data.lerp_(source.data, self.cfg.tau)
            for source, target in zip(self.critic.parameters(), self.critic_target.parameters()):
                target.data.lerp_(source.data, self.cfg.tau)
        stats = {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "q_mean": float(predicted_q.mean().item()),
        }
        if isinstance(self.critic, MRNGoalCritic):
            with torch.no_grad():
                _distance, metric, residual = self.critic.components(obs, action, goal)
            stats.update({
                "mrn_metric_mean": float(metric.mean().item()),
                "mrn_residual_mean": float(residual.mean().item()),
            })
        return stats

    def optimizer_state_dict(self) -> Dict[str, object]:
        return {
            "actor": self.actor_optimizer.state_dict(),
            "critic": self.critic_optimizer.state_dict(),
        }

    def load_optimizer_state_dict(self, state: Dict[str, object]) -> None:
        if "actor" in state:
            self.actor_optimizer.load_state_dict(state["actor"])  # type: ignore[arg-type]
        if "critic" in state:
            self.critic_optimizer.load_state_dict(state["critic"])  # type: ignore[arg-type]


class ContextHERDDPGAgent(_DDPGBase):
    def __init__(self, env: CommInspectionDubinsUAV2D, cfg: ContextGCRLConfig, device: torch.device) -> None:
        super().__init__(env, cfg, device, critic_kind="monolithic")


class MRNContextHERDDPGAgent(_DDPGBase):
    def __init__(self, env: CommInspectionDubinsUAV2D, cfg: ContextGCRLConfig, device: torch.device) -> None:
        super().__init__(env, cfg, device, critic_kind="mrn")


class SquashedGaussianGoalActor(nn.Module):
    def __init__(self, obs_dim: int, goal_dim: int, act_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = _mlp([obs_dim + goal_dim, hidden_dim, hidden_dim, 2 * act_dim])

    def parameters_for_distribution(
        self,
        obs: Tensor,
        goal: Tensor,
        log_std_min: float,
        log_std_max: float,
    ) -> tuple[Tensor, Tensor]:
        mean, log_std = self.net(torch.cat([obs, goal], dim=-1)).chunk(2, dim=-1)
        return mean, log_std.clamp(float(log_std_min), float(log_std_max))


class ContextContrastiveRLAgent(GoalConditionedAgentBase):
    def __init__(self, env: CommInspectionDubinsUAV2D, cfg: ContextGCRLConfig, device: torch.device) -> None:
        super().__init__()
        self.env = env
        self.cfg = cfg
        self.device = torch.device(device)
        self.obs_dim = int(env.observation_space.shape[0])
        self.goal_dim = self.obs_dim
        self.act_dim = int(env.action_space.shape[0])
        self.action_low = torch.as_tensor(env.action_space.low, device=self.device)
        self.action_high = torch.as_tensor(env.action_space.high, device=self.device)
        self.actor = SquashedGaussianGoalActor(
            self.obs_dim, self.goal_dim, self.act_dim, cfg.hidden_dim
        ).to(self.device)
        self.state_action_encoder = _mlp([
            self.obs_dim + self.act_dim,
            cfg.hidden_dim,
            cfg.hidden_dim,
            cfg.representation_dim,
        ]).to(self.device)
        self.goal_encoder = _mlp([
            self.goal_dim,
            cfg.hidden_dim,
            cfg.hidden_dim,
            cfg.representation_dim,
        ]).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        critic_parameters = list(self.state_action_encoder.parameters()) + list(self.goal_encoder.parameters())
        self.critic_optimizer = torch.optim.Adam(critic_parameters, lr=cfg.critic_lr)
        self.log_alpha = torch.zeros((), device=self.device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
        self.target_entropy = -float(self.act_dim)

    @property
    def alpha(self) -> Tensor:
        return self.log_alpha.exp()

    def _sample_action(self, obs: Tensor, goal: Tensor, *, deterministic: bool) -> tuple[Tensor, Tensor]:
        mean, log_std = self.actor.parameters_for_distribution(
            obs, goal, self.cfg.log_std_min, self.cfg.log_std_max
        )
        pre_tanh = mean if deterministic else mean + log_std.exp() * torch.randn_like(mean)
        unit_action = torch.tanh(pre_tanh)
        scale = 0.5 * (self.action_high - self.action_low)
        bias = 0.5 * (self.action_high + self.action_low)
        action = bias + scale * unit_action
        normal_log_prob = -0.5 * (
            ((pre_tanh - mean) / (log_std.exp() + 1e-8)).pow(2)
            + 2.0 * log_std
            + math.log(2.0 * math.pi)
        )
        log_prob = normal_log_prob.sum(dim=-1)
        log_prob -= torch.log(1.0 - unit_action.pow(2) + 1e-6).sum(dim=-1)
        log_prob -= torch.log(scale.clamp_min(1e-8)).sum()
        return action, log_prob

    def _representations(self, obs: Tensor, action: Tensor, goal: Tensor) -> tuple[Tensor, Tensor]:
        state_action = F.normalize(
            self.state_action_encoder(torch.cat([obs, action], dim=-1)), dim=-1
        )
        goal_representation = F.normalize(self.goal_encoder(goal), dim=-1)
        return state_action, goal_representation

    def score(self, obs: Tensor, action: Tensor, goal: Tensor) -> Tensor:
        state_action, goal_representation = self._representations(obs, action, goal)
        return (state_action * goal_representation).sum(dim=-1) / max(
            float(self.cfg.contrastive_temperature), 1e-6
        )

    def contrastive_loss(
        self,
        obs: Tensor,
        action: Tensor,
        goal: Tensor,
        device_ids: List[str],
    ) -> tuple[Tensor, Dict[str, float]]:
        state_action, goal_representation = self._representations(obs, action, goal)
        logits = state_action @ goal_representation.T
        logits = logits / max(float(self.cfg.contrastive_temperature), 1e-6)
        batch_size = logits.shape[0]
        diagonal = torch.eye(batch_size, device=logits.device, dtype=torch.bool)
        same_device = torch.as_tensor(
            [[left == right for right in device_ids] for left in device_ids],
            device=logits.device,
            dtype=torch.bool,
        )
        negative_mask = ~(diagonal | same_device)
        positive_logits = logits[diagonal]
        negative_logits = logits[negative_mask]
        positive_loss = F.softplus(-positive_logits).mean()
        negative_loss = (
            F.softplus(negative_logits).mean()
            if negative_logits.numel() > 0
            else torch.zeros((), device=logits.device)
        )
        loss = positive_loss + negative_loss
        with torch.no_grad():
            positive_accuracy = (positive_logits > 0.0).float().mean()
            negative_accuracy = (
                (negative_logits < 0.0).float().mean()
                if negative_logits.numel() > 0
                else torch.ones((), device=logits.device)
            )
        return loss, {
            "contrastive_positive_accuracy": float(positive_accuracy.item()),
            "contrastive_negative_accuracy": float(negative_accuracy.item()),
            "contrastive_positive_score": float(positive_logits.mean().item()),
            "contrastive_negative_score": float(negative_logits.mean().item()) if negative_logits.numel() else 0.0,
        }

    @torch.no_grad()
    def act(self, obs: np.ndarray, goal_obs: np.ndarray, eval_mode: bool = True) -> np.ndarray:
        obs_t = torch.as_tensor(obs[None], device=self.device, dtype=torch.float32)
        goal_t = torch.as_tensor(goal_obs[None], device=self.device, dtype=torch.float32)
        action, _ = self._sample_action(obs_t, goal_t, deterministic=bool(eval_mode))
        return action[0].cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def batch_value(self, obs_batch: np.ndarray, goal_obs_batch: np.ndarray) -> np.ndarray:
        obs = torch.as_tensor(obs_batch, device=self.device, dtype=torch.float32)
        goal = torch.as_tensor(goal_obs_batch, device=self.device, dtype=torch.float32)
        action, _ = self._sample_action(obs, goal, deterministic=True)
        return F.softplus(-self.score(obs, action, goal)).cpu().numpy().astype(np.float32)

    def value(self, obs: np.ndarray, goal_obs: np.ndarray) -> float:
        return float(self.batch_value(obs[None], goal_obs[None])[0])

    def update(self, batch: Dict[str, Tensor | List[str]]) -> Dict[str, float]:
        obs = batch["obs"]
        goal = batch["goal"]
        action = batch["action"]
        device_ids = batch["device_ids"]
        assert isinstance(obs, Tensor) and isinstance(goal, Tensor) and isinstance(action, Tensor)
        assert isinstance(device_ids, list)
        critic_loss, stats = self.contrastive_loss(obs, action, goal, device_ids)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        critic_parameters = list(self.state_action_encoder.parameters()) + list(self.goal_encoder.parameters())
        for parameter in critic_parameters:
            parameter.requires_grad_(False)
        policy_action, log_prob = self._sample_action(obs, goal, deterministic=False)
        policy_score = self.score(obs, policy_action, goal)
        actor_loss = (self.alpha.detach() * log_prob - policy_score).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        for parameter in critic_parameters:
            parameter.requires_grad_(True)

        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        stats.update({
            "contrastive_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.item()),
        })
        return stats

    def optimizer_state_dict(self) -> Dict[str, object]:
        return {
            "actor": self.actor_optimizer.state_dict(),
            "critic": self.critic_optimizer.state_dict(),
            "alpha": self.alpha_optimizer.state_dict(),
        }

    def load_optimizer_state_dict(self, state: Dict[str, object]) -> None:
        if "actor" in state:
            self.actor_optimizer.load_state_dict(state["actor"])  # type: ignore[arg-type]
        if "critic" in state:
            self.critic_optimizer.load_state_dict(state["critic"])  # type: ignore[arg-type]
        if "alpha" in state:
            self.alpha_optimizer.load_state_dict(state["alpha"])  # type: ignore[arg-type]


AGENT_TYPES = {
    "context_her_ddpg": ContextHERDDPGAgent,
    "context_contrastive_rl": ContextContrastiveRLAgent,
    "mrn_context_her_ddpg": MRNContextHERDDPGAgent,
}


def make_context_agent(
    algorithm: str,
    env: CommInspectionDubinsUAV2D,
    cfg: ContextGCRLConfig,
    device: torch.device,
) -> GoalConditionedAgentBase:
    try:
        agent_type = AGENT_TYPES[str(algorithm)]
    except KeyError as exc:
        raise ValueError(f"unknown context GCRL algorithm: {algorithm}") from exc
    return agent_type(env, cfg, device)


def context_agent_metadata(agent: GoalConditionedAgentBase) -> Dict[str, object]:
    cfg = getattr(agent, "cfg")
    payload: Dict[str, object] = {
        "config": asdict(cfg),
        "obs_dim": int(getattr(agent, "obs_dim")),
        "act_dim": int(getattr(agent, "act_dim")),
        "actor_parameters": parameter_count(getattr(agent, "actor")),
    }
    critic = getattr(agent, "critic", None)
    if critic is not None:
        payload["critic_parameters"] = parameter_count(critic)
    else:
        payload["critic_parameters"] = parameter_count(getattr(agent, "state_action_encoder")) + parameter_count(getattr(agent, "goal_encoder"))
    if isinstance(agent, MRNContextHERDDPGAgent):
        payload["mrn_hidden_dim"] = int(agent.mrn_hidden_dim)
    return payload
