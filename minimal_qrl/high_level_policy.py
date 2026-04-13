from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from minimal_qrl.envs import CommInspectionDubinsUAV2D
from minimal_qrl.gc_agents import GoalConditionedAgentBase


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    )


class FrozenQRLNavigationFeatures:
    def __init__(
        self,
        qrl_agent: nn.Module,
        *,
        obs_dim: int,
        device: torch.device,
        critic_index: int = 0,
        use_distance: bool = True,
        use_latent: bool = True,
    ):
        critics = list(getattr(qrl_agent, "critics", []))
        if not critics:
            raise ValueError("QRL agent 不包含 critic，无法构建高层导航特征")
        if critic_index < 0 or critic_index >= len(critics):
            raise ValueError(f"非法的 critic_index={critic_index}，当前 critic 数量={len(critics)}")

        self.qrl_agent = qrl_agent
        self.critic_index = int(critic_index)
        self.critic = critics[self.critic_index]
        self.device = device
        self.obs_dim = int(obs_dim)
        self.use_distance = bool(use_distance)
        self.use_latent = bool(use_latent)

        for param in self.critic.parameters():
            param.requires_grad = False
        self.critic.eval()

        with torch.no_grad():
            dummy = torch.zeros((1, self.obs_dim), dtype=torch.float32, device=self.device)
            latent = self.critic.encoder(dummy)
        self.latent_dim = int(latent.shape[-1])
        self.feature_dim = 0
        if self.use_distance:
            self.feature_dim += 1
        if self.use_latent:
            self.feature_dim += 3 * self.latent_dim

    @torch.no_grad()
    def encode_batch(
        self,
        obs_batch: np.ndarray | torch.Tensor,
        goal_obs_batch: np.ndarray | torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        goal_t = torch.as_tensor(goal_obs_batch, dtype=torch.float32, device=self.device)
        z_t = self.critic.encoder(obs_t)
        z_g = self.critic.encoder(goal_t)
        d = self.critic.quasimetric_model(z_t, z_g).unsqueeze(-1)
        return {
            "distance": d.detach(),
            "z_t": z_t.detach(),
            "z_g": z_g.detach(),
            "delta_z": (z_g - z_t).detach(),
        }

    @torch.no_grad()
    def build_state_tensor(
        self,
        obs_batch: np.ndarray | torch.Tensor,
        goal_obs_batch: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        goal_t = torch.as_tensor(goal_obs_batch, dtype=torch.float32, device=self.device)
        features = self.encode_batch(obs_t, goal_t)
        parts = [obs_t, goal_t]
        if self.use_distance:
            parts.append(features["distance"])
        if self.use_latent:
            parts.extend([features["z_t"], features["z_g"], features["delta_z"]])
        return torch.cat(parts, dim=-1)

    @torch.no_grad()
    def build_state(
        self,
        obs: np.ndarray,
        goal_obs: np.ndarray,
    ) -> np.ndarray:
        state_t = self.build_state_tensor(obs[None], goal_obs[None])
        return state_t[0].detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def distance_to_goal(
        self,
        obs: np.ndarray,
        goal_obs: np.ndarray,
    ) -> float:
        features = self.encode_batch(obs[None], goal_obs[None])
        return float(features["distance"][0, 0].cpu().item())


class SquashedGaussianActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.backbone = _mlp(state_dim, hidden_dim, hidden_dim)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(state)
        mean = self.mean_head(feat)
        log_std = torch.clamp(self.log_std_head(feat), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(
        self,
        state: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state)
        if deterministic:
            pre_tanh = mean
            action = torch.tanh(pre_tanh)
            log_prob = torch.zeros((state.shape[0],), dtype=state.dtype, device=state.device)
            return action, log_prob

        std = log_std.exp()
        noise = torch.randn_like(mean)
        pre_tanh = mean + std * noise
        action = torch.tanh(pre_tanh)

        log_prob = (
            -0.5 * (((pre_tanh - mean) / (std + 1e-6)) ** 2)
            - log_std
            - 0.5 * np.log(2.0 * np.pi)
        ).sum(dim=-1)
        log_prob -= torch.log(1.0 - action.pow(2) + 1e-6).sum(dim=-1)
        return action, log_prob


class TwinQCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = _mlp(state_dim + action_dim, hidden_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.net(x).squeeze(-1)


class HighLevelReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, size: int, device: torch.device):
        self.state_buf = np.zeros((size, state_dim), dtype=np.float32)
        self.action_buf = np.zeros((size, action_dim), dtype=np.float32)
        self.reward_buf = np.zeros((size,), dtype=np.float32)
        self.next_state_buf = np.zeros((size, state_dim), dtype=np.float32)
        self.done_buf = np.zeros((size,), dtype=np.float32)
        self.discount_buf = np.zeros((size,), dtype=np.float32)
        self.segment_len_buf = np.zeros((size,), dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.max_size = int(size)
        self.device = device

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        discount: float,
        segment_len: int,
    ) -> None:
        idx = self.ptr
        self.state_buf[idx] = np.asarray(state, dtype=np.float32)
        self.action_buf[idx] = np.asarray(action, dtype=np.float32)
        self.reward_buf[idx] = float(reward)
        self.next_state_buf[idx] = np.asarray(next_state, dtype=np.float32)
        self.done_buf[idx] = float(done)
        self.discount_buf[idx] = float(discount)
        self.segment_len_buf[idx] = float(segment_len)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        if self.size < batch_size:
            raise ValueError(f"ReplayBuffer 样本不足: size={self.size}, batch_size={batch_size}")
        idxs = np.random.randint(0, self.size, size=batch_size)
        return {
            "state": torch.as_tensor(self.state_buf[idxs], dtype=torch.float32, device=self.device),
            "action": torch.as_tensor(self.action_buf[idxs], dtype=torch.float32, device=self.device),
            "reward": torch.as_tensor(self.reward_buf[idxs], dtype=torch.float32, device=self.device),
            "next_state": torch.as_tensor(self.next_state_buf[idxs], dtype=torch.float32, device=self.device),
            "done": torch.as_tensor(self.done_buf[idxs], dtype=torch.float32, device=self.device),
            "discount": torch.as_tensor(self.discount_buf[idxs], dtype=torch.float32, device=self.device),
            "segment_len": torch.as_tensor(self.segment_len_buf[idxs], dtype=torch.float32, device=self.device),
        }


@dataclass
class HighLevelSACConfig:
    train_steps: int = 5000
    batch_size: int = 128
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    init_alpha: float = 0.2
    replay_size: int = 200_000
    start_random_steps: int = 1000
    updates_per_step: int = 1
    hidden_dim: int = 256
    save_interval: int = 1000
    log_interval: int = 100
    seed: int = 0
    high_level_period: int = 5
    subgoal_max_radius: float = 1.5
    subgoal_relative_param: str = "polar_local"
    use_qrl_distance: bool = True
    use_qrl_latent: bool = True
    qrl_critic_index: int = 0
    target_entropy: Optional[float] = None


class CostAwareSubgoalPolicy(nn.Module):
    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int = 3,
        hidden_dim: int = 256,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        tau: float = 0.005,
        init_alpha: float = 0.2,
        target_entropy: Optional[float] = None,
        subgoal_max_radius: float = 1.5,
        subgoal_relative_param: str = "polar_local",
        device: torch.device,
    ):
        super().__init__()
        if str(subgoal_relative_param) != "polar_local":
            raise ValueError(f"当前仅支持 subgoal_relative_param='polar_local'，得到 {subgoal_relative_param}")

        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.tau = float(tau)
        self.target_entropy = float(target_entropy) if target_entropy is not None else -float(self.action_dim)
        self.subgoal_max_radius = float(subgoal_max_radius)
        self.subgoal_relative_param = str(subgoal_relative_param)
        self.device = device

        self.actor = SquashedGaussianActor(self.state_dim, self.action_dim, self.hidden_dim).to(device)
        self.q1 = TwinQCritic(self.state_dim, self.action_dim, self.hidden_dim).to(device)
        self.q2 = TwinQCritic(self.state_dim, self.action_dim, self.hidden_dim).to(device)
        self.q1_targ = TwinQCritic(self.state_dim, self.action_dim, self.hidden_dim).to(device)
        self.q2_targ = TwinQCritic(self.state_dim, self.action_dim, self.hidden_dim).to(device)
        self.q1_targ.load_state_dict(self.q1.state_dict())
        self.q2_targ.load_state_dict(self.q2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=float(actor_lr))
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=float(critic_lr))
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=float(critic_lr))
        self.log_alpha = torch.tensor(
            float(np.log(max(float(init_alpha), 1e-6))),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=float(critic_lr))

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def act(
        self,
        state: np.ndarray,
        *,
        eval_mode: bool = True,
    ) -> np.ndarray:
        state_t = torch.as_tensor(state[None], dtype=torch.float32, device=self.device)
        action_t, _ = self.actor.sample(state_t, deterministic=bool(eval_mode))
        return action_t[0].detach().cpu().numpy().astype(np.float32)

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        state = batch["state"]
        action = batch["action"]
        reward = batch["reward"]
        next_state = batch["next_state"]
        done = batch["done"]
        discount = batch["discount"]
        segment_len = batch["segment_len"]

        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state, deterministic=False)
            next_q1 = self.q1_targ(next_state, next_action)
            next_q2 = self.q2_targ(next_state, next_action)
            next_v = torch.min(next_q1, next_q2) - self.alpha.detach() * next_log_prob
            target_q = reward + discount * (1.0 - done) * next_v

        q1_pred = self.q1(state, action)
        q2_pred = self.q2(state, action)
        q1_loss = F.mse_loss(q1_pred, target_q)
        q2_loss = F.mse_loss(q2_pred, target_q)

        self.q1_opt.zero_grad(set_to_none=True)
        q1_loss.backward()
        self.q1_opt.step()

        self.q2_opt.zero_grad(set_to_none=True)
        q2_loss.backward()
        self.q2_opt.step()

        policy_action, log_prob = self.actor.sample(state, deterministic=False)
        q1_pi = self.q1(state, policy_action)
        q2_pi = self.q2(state, policy_action)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha.detach() * log_prob - q_pi).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        with torch.no_grad():
            for param, target_param in zip(self.q1.parameters(), self.q1_targ.parameters()):
                target_param.data.mul_(1.0 - self.tau).add_(self.tau * param.data)
            for param, target_param in zip(self.q2.parameters(), self.q2_targ.parameters()):
                target_param.data.mul_(1.0 - self.tau).add_(self.tau * param.data)

        return {
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.item()),
            "mean_discount": float(discount.mean().item()),
            "mean_segment_len_batch": float(segment_len.mean().item()),
        }


def decode_relative_subgoal(
    raw_action: np.ndarray,
    env: CommInspectionDubinsUAV2D,
    *,
    subgoal_max_radius: float,
) -> Dict[str, Any]:
    clipped = np.clip(np.asarray(raw_action, dtype=np.float32).reshape(3), -1.0, 1.0)
    current_state = np.asarray(env.state, dtype=np.float32).reshape(3)

    radius = 0.5 * (float(clipped[0]) + 1.0) * float(subgoal_max_radius)
    bearing_offset = float(np.pi * clipped[1])
    heading_offset = float(np.pi * clipped[2])

    travel_bearing = float(current_state[2] + bearing_offset)
    raw_theta = float(env._normalize_angle(float(current_state[2] + heading_offset)))
    raw_subgoal = np.array(
        [
            float(current_state[0] + radius * np.cos(travel_bearing)),
            float(current_state[1] + radius * np.sin(travel_bearing)),
            raw_theta,
        ],
        dtype=np.float32,
    )

    repair_info = env.repair_state_with_info(raw_subgoal)
    repaired = np.asarray(repair_info["repaired_state"], dtype=np.float32)
    repair_metrics = env.compute_repair_metrics(raw_subgoal, repaired)
    return {
        "raw_action": clipped.astype(np.float32),
        "radius": float(radius),
        "bearing_offset": float(bearing_offset),
        "heading_offset": float(heading_offset),
        "raw_subgoal": raw_subgoal.astype(np.float32),
        "repaired_subgoal": repaired.astype(np.float32),
        "executed_subgoal": repaired.astype(np.float32),
        "raw_valid": bool(env.is_valid_state(raw_subgoal)),
        "used_nearby_repair": bool(repair_info["used_nearby_repair"]),
        "used_global_repair_fallback": bool(repair_info["used_global_fallback"]),
        "repair_distance": float(repair_metrics["repair_distance"]),
        "repair_dtheta": float(repair_metrics["repair_dtheta"]),
        "raw_task_score": float(env.compute_task_score(raw_subgoal)),
        "repaired_task_score": float(env.compute_task_score(repaired)),
        "executed_task_score": float(env.compute_task_score(repaired)),
    }


def select_high_level_subgoal(
    policy: CostAwareSubgoalPolicy,
    nav_features: FrozenQRLNavigationFeatures,
    env: CommInspectionDubinsUAV2D,
    obs: np.ndarray,
    goal_obs: np.ndarray,
    *,
    eval_mode: bool,
) -> Dict[str, Any]:
    high_level_state = nav_features.build_state(obs, goal_obs)
    raw_action = policy.act(high_level_state, eval_mode=eval_mode)
    choice = decode_relative_subgoal(
        raw_action,
        env,
        subgoal_max_radius=float(policy.subgoal_max_radius),
    )
    choice["high_level_state"] = high_level_state.astype(np.float32)
    choice["qrl_distance_to_final"] = float(nav_features.distance_to_goal(obs, goal_obs))
    return choice


def save_high_level_policy_checkpoint(
    path: str,
    policy: CostAwareSubgoalPolicy,
    *,
    train_step: int,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "train_step": int(train_step),
        "state_dim": int(policy.state_dim),
        "action_dim": int(policy.action_dim),
        "hidden_dim": int(policy.hidden_dim),
        "tau": float(policy.tau),
        "target_entropy": float(policy.target_entropy),
        "subgoal_max_radius": float(policy.subgoal_max_radius),
        "subgoal_relative_param": str(policy.subgoal_relative_param),
        "actor": policy.actor.state_dict(),
        "q1": policy.q1.state_dict(),
        "q2": policy.q2.state_dict(),
        "q1_targ": policy.q1_targ.state_dict(),
        "q2_targ": policy.q2_targ.state_dict(),
        "log_alpha": float(policy.log_alpha.detach().cpu().item()),
        "metadata": metadata or {},
    }
    torch.save(payload, path)


def load_high_level_policy_checkpoint(
    path: str,
    *,
    device: torch.device,
) -> tuple[CostAwareSubgoalPolicy, Dict[str, Any]]:
    ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict) or "actor" not in ckpt:
        raise ValueError(f"非法的 high-level policy checkpoint: {path}")
    policy = CostAwareSubgoalPolicy(
        state_dim=int(ckpt["state_dim"]),
        action_dim=int(ckpt.get("action_dim", 3)),
        hidden_dim=int(ckpt.get("hidden_dim", 256)),
        tau=float(ckpt.get("tau", 0.005)),
        init_alpha=float(np.exp(float(ckpt.get("log_alpha", np.log(0.2))))),
        target_entropy=float(ckpt.get("target_entropy", -float(int(ckpt.get("action_dim", 3))))),
        subgoal_max_radius=float(ckpt.get("subgoal_max_radius", 1.5)),
        subgoal_relative_param=str(ckpt.get("subgoal_relative_param", "polar_local")),
        device=device,
    )
    policy.actor.load_state_dict(ckpt["actor"])
    policy.q1.load_state_dict(ckpt["q1"])
    policy.q2.load_state_dict(ckpt["q2"])
    policy.q1_targ.load_state_dict(ckpt.get("q1_targ", ckpt["q1"]))
    policy.q2_targ.load_state_dict(ckpt.get("q2_targ", ckpt["q2"]))
    policy.log_alpha.data.copy_(
        torch.tensor(float(ckpt.get("log_alpha", np.log(0.2))), dtype=torch.float32, device=device)
    )
    policy.eval()
    return policy, dict(ckpt.get("metadata", {}))


def train_high_level_policy(
    policy: CostAwareSubgoalPolicy,
    nav_features: FrozenQRLNavigationFeatures,
    low_level_agent: GoalConditionedAgentBase,
    env_factory: Callable[[], CommInspectionDubinsUAV2D],
    planner_fn: Callable[..., np.ndarray],
    planner_cfg: Any,
    device: torch.device,
    cfg: HighLevelSACConfig,
    *,
    log_fn: Optional[Callable[[int, Dict[str, float]], None]] = None,
    checkpoint_fn: Optional[Callable[[int, Dict[str, float]], None]] = None,
) -> Dict[str, float]:
    del device  # policy / nav_features 已各自持有 device
    rng = np.random.default_rng(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    torch.manual_seed(int(cfg.seed))

    env = env_factory()
    replay = HighLevelReplayBuffer(
        state_dim=int(policy.state_dim),
        action_dim=int(policy.action_dim),
        size=int(cfg.replay_size),
        device=policy.device,
    )

    obs, _ = env.reset(seed=int(cfg.seed))
    goal_obs = env.state_to_observation(np.asarray(env.goal, dtype=np.float32))
    current_high_state = nav_features.build_state(obs, goal_obs)
    final_metrics: Dict[str, float] = {}

    progress = tqdm(range(1, int(cfg.train_steps) + 1), desc="HighLevelSAC", leave=True)
    for step in progress:
        if step <= int(cfg.start_random_steps):
            raw_action = rng.uniform(-1.0, 1.0, size=(policy.action_dim,)).astype(np.float32)
        else:
            raw_action = policy.act(current_high_state, eval_mode=False)

        subgoal_choice = decode_relative_subgoal(
            raw_action,
            env,
            subgoal_max_radius=float(policy.subgoal_max_radius),
        )
        executed_subgoal = np.asarray(subgoal_choice["executed_subgoal"], dtype=np.float32)

        segment_reward = 0.0
        segment_len = 0
        done = False
        truncated = False
        last_obs = obs
        last_info: Dict[str, Any] = {}
        for _ in range(max(1, int(cfg.high_level_period))):
            action = planner_fn(
                low_level_agent,
                env,
                goal_obs,
                planner_cfg,
                subgoal_state=executed_subgoal,
            )
            last_obs, reward, done, truncated, last_info = env.step(action)
            segment_reward += float(reward)
            segment_len += 1

            if env.is_subgoal_reached(
                env.state,
                executed_subgoal,
                pos_tolerance=float(planner_cfg.subgoal_reached_pos_tolerance),
                theta_tolerance=float(planner_cfg.subgoal_reached_theta_tolerance),
            ):
                break
            if done or truncated:
                break

        next_goal_obs = env.state_to_observation(np.asarray(env.goal, dtype=np.float32))
        next_high_state = nav_features.build_state(last_obs, next_goal_obs)
        episode_done = bool(done or truncated)
        replay.add(
            current_high_state,
            subgoal_choice["raw_action"],
            float(segment_reward),
            next_high_state,
            episode_done,
            float(cfg.gamma ** max(segment_len, 1)),
            int(segment_len),
        )

        update_metrics: Dict[str, float] = {}
        if replay.size >= int(cfg.batch_size):
            for _ in range(max(1, int(cfg.updates_per_step))):
                update_metrics = policy.update(replay.sample(int(cfg.batch_size)))

        qrl_distance = float(nav_features.distance_to_goal(obs, goal_obs))
        final_metrics = {
            "segment_reward": float(segment_reward),
            "segment_len": float(segment_len),
            "qrl_distance_to_final": float(qrl_distance),
            "raw_subgoal_valid_rate": float(subgoal_choice["raw_valid"]),
            "mean_repair_distance": float(subgoal_choice["repair_distance"]),
            "mean_repair_dtheta": float(subgoal_choice["repair_dtheta"]),
            "mean_taskscore_raw_subgoal": float(subgoal_choice["raw_task_score"]),
            "mean_taskscore_repaired_subgoal": float(subgoal_choice["repaired_task_score"]),
            "episode_done_rate": float(episode_done),
            "last_comm_margin": float(last_info.get("comm_margin", 0.0)) if last_info else 0.0,
            "last_obs_margin": float(last_info.get("obs_margin", 0.0)) if last_info else 0.0,
            **update_metrics,
        }
        progress.set_postfix(
            reward=f"{final_metrics['segment_reward']:.3f}",
            seg_len=f"{final_metrics['segment_len']:.1f}",
            repair=f"{final_metrics['mean_repair_distance']:.3f}",
        )

        if log_fn is not None:
            log_fn(step, final_metrics)
        if checkpoint_fn is not None and int(cfg.save_interval) > 0 and step % int(cfg.save_interval) == 0:
            checkpoint_fn(step, final_metrics)

        if episode_done:
            reset_seed = int(rng.integers(0, 1_000_000_000))
            obs, _ = env.reset(seed=reset_seed)
            goal_obs = env.state_to_observation(np.asarray(env.goal, dtype=np.float32))
            current_high_state = nav_features.build_state(obs, goal_obs)
        else:
            obs = np.asarray(last_obs, dtype=np.float32)
            goal_obs = next_goal_obs
            current_high_state = next_high_state

    progress.close()
    policy.eval()
    return final_metrics
