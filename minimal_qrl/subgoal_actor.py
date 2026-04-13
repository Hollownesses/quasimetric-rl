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


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    )


@dataclass
class SubgoalActorTrainConfig:
    train_steps: int = 5000
    batch_size: int = 64
    lr: float = 3e-4
    hidden_dim: int = 256
    num_candidates: int = 64
    lambda_final: float = 0.3
    lambda_task: float = 1.0
    seed: int = 0
    save_interval: int = 1000


class SubgoalActor(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.hidden_dim = int(hidden_dim)
        self.net = _mlp(self.obs_dim * 2, self.hidden_dim, 4)

    def forward(self, obs: torch.Tensor, goal_obs: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, goal_obs], dim=-1))

    @staticmethod
    def _decode_output(
        raw_out: torch.Tensor,
        env: CommInspectionDubinsUAV2D,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_mid = 0.5 * (env.x_min + env.x_max)
        y_mid = 0.5 * (env.y_min + env.y_max)
        x_half = 0.5 * (env.x_max - env.x_min)
        y_half = 0.5 * (env.y_max - env.y_min)

        x = x_mid + x_half * torch.tanh(raw_out[..., 0])
        y = y_mid + y_half * torch.tanh(raw_out[..., 1])
        heading = F.normalize(raw_out[..., 2:4], dim=-1, eps=1e-6)
        sin_theta = heading[..., 0]
        cos_theta = heading[..., 1]
        theta = torch.atan2(sin_theta, cos_theta)
        state = torch.stack([x, y, theta], dim=-1)
        return state, sin_theta, cos_theta

    def predict_state_tensor(
        self,
        obs: torch.Tensor,
        goal_obs: torch.Tensor,
        env: CommInspectionDubinsUAV2D,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_out = self.forward(obs, goal_obs)
        return self._decode_output(raw_out, env)

    @torch.no_grad()
    def predict_state(
        self,
        obs: np.ndarray,
        goal_obs: np.ndarray,
        env: CommInspectionDubinsUAV2D,
        device: torch.device,
    ) -> np.ndarray:
        obs_t = torch.as_tensor(obs[None], dtype=torch.float32, device=device)
        goal_t = torch.as_tensor(goal_obs[None], dtype=torch.float32, device=device)
        state_t, _, _ = self.predict_state_tensor(obs_t, goal_t, env)
        return state_t[0].detach().cpu().numpy().astype(np.float32)


def evaluate_high_level_objective(
    env: CommInspectionDubinsUAV2D,
    agent: GoalConditionedAgentBase,
    state: np.ndarray,
    candidate: np.ndarray,
    goal_obs: np.ndarray,
    *,
    lambda_final: float,
    lambda_task: float,
) -> float:
    cand_obs = env.state_to_observation(candidate)
    c_reach = env.compute_goal_reaching_cost_estimate(state=state, goal=candidate)
    d_final = float(agent.value(cand_obs, goal_obs))
    task_score = env.compute_task_score(candidate)
    return float(c_reach + float(lambda_final) * d_final - float(lambda_task) * task_score)


def evaluate_high_level_objective_batch(
    env: CommInspectionDubinsUAV2D,
    agent: GoalConditionedAgentBase,
    state: np.ndarray,
    candidates: np.ndarray,
    goal_obs: np.ndarray,
    *,
    lambda_final: float,
    lambda_task: float,
) -> np.ndarray:
    candidates = np.asarray(candidates, dtype=np.float32).reshape(-1, 3)
    if candidates.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)

    cand_obs_batch = np.asarray(
        [env.state_to_observation(candidate) for candidate in candidates],
        dtype=np.float32,
    )
    goal_obs_batch = np.repeat(goal_obs[None], candidates.shape[0], axis=0).astype(np.float32)
    d_final = agent.batch_value(cand_obs_batch, goal_obs_batch)
    c_reach = np.asarray(
        [env.compute_goal_reaching_cost_estimate(state=state, goal=candidate) for candidate in candidates],
        dtype=np.float32,
    )
    task_scores = np.asarray([env.compute_task_score(candidate) for candidate in candidates], dtype=np.float32)
    return c_reach + float(lambda_final) * d_final - float(lambda_task) * task_scores


def _sample_teacher_candidate_set(
    env: CommInspectionDubinsUAV2D,
    raw_subgoal: np.ndarray,
    repaired_subgoal: np.ndarray,
    num_candidates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    candidates = [np.asarray(repaired_subgoal, dtype=np.float32)]

    n_local = max(8, int(num_candidates) // 3)
    pos_scale = 0.1 * max(env.x_max - env.x_min, env.y_max - env.y_min)
    for _ in range(n_local):
        perturbed = np.asarray(raw_subgoal, dtype=np.float32).copy()
        perturbed[:2] += rng.normal(0.0, pos_scale, size=2).astype(np.float32)
        perturbed[2] = float(perturbed[2] + rng.normal(0.0, 0.5))
        candidates.append(env.repair_state(perturbed))

    random_pool = []
    while len(candidates) + len(random_pool) < max(num_candidates, 1):
        random_pool.append(env.sample_valid_state(seed=int(rng.integers(0, 1_000_000_000))))

    if random_pool:
        scores = [env.compute_task_score(c) for c in random_pool]
        best_idx = np.argsort(np.asarray(scores))[::-1][: max(1, len(random_pool) // 6)]
        for idx in best_idx:
            candidates.append(np.asarray(random_pool[int(idx)], dtype=np.float32))
        candidates.extend(random_pool)

    return np.asarray(candidates[: max(num_candidates, 1)], dtype=np.float32)


def select_teacher_subgoal(
    actor: SubgoalActor,
    agent: GoalConditionedAgentBase,
    env: CommInspectionDubinsUAV2D,
    obs: np.ndarray,
    goal_obs: np.ndarray,
    *,
    device: torch.device,
    num_candidates: int,
    lambda_final: float,
    lambda_task: float,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    state = env.observation_to_state(obs)
    raw_subgoal = actor.predict_state(obs, goal_obs, env, device=device)
    repair_info = env.repair_state_with_info(raw_subgoal)
    repaired_subgoal = np.asarray(repair_info["repaired_state"], dtype=np.float32)
    candidates = _sample_teacher_candidate_set(env, raw_subgoal, repaired_subgoal, num_candidates, rng)
    candidate_scores = evaluate_high_level_objective_batch(
        env,
        agent,
        state,
        candidates,
        goal_obs,
        lambda_final=lambda_final,
        lambda_task=lambda_task,
    )
    best_idx = int(np.argmin(candidate_scores)) if len(candidate_scores) > 0 else 0
    best_score = float(candidate_scores[best_idx]) if len(candidate_scores) > 0 else float("inf")
    best_candidate = np.asarray(candidates[best_idx], dtype=np.float32) if len(candidates) > 0 else repaired_subgoal

    repair_metrics = env.compute_repair_metrics(raw_subgoal, repaired_subgoal)
    return {
        "raw_subgoal": np.asarray(raw_subgoal, dtype=np.float32),
        "repaired_subgoal": np.asarray(repaired_subgoal, dtype=np.float32),
        "teacher_subgoal": np.asarray(best_candidate, dtype=np.float32),
        "candidate_count": int(len(candidates)),
        "raw_valid": bool(env.is_valid_state(raw_subgoal)),
        "used_nearby_repair": bool(repair_info["used_nearby_repair"]),
        "used_global_repair_fallback": bool(repair_info["used_global_fallback"]),
        "repair_distance": float(repair_metrics["repair_distance"]),
        "repair_dtheta": float(repair_metrics["repair_dtheta"]),
        "raw_task_score": float(env.compute_task_score(raw_subgoal)),
        "repaired_task_score": float(env.compute_task_score(repaired_subgoal)),
        "teacher_objective": float(best_score),
    }


def save_subgoal_actor_checkpoint(
    path: str,
    actor: SubgoalActor,
    *,
    train_step: int,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "train_step": int(train_step),
        "obs_dim": int(actor.obs_dim),
        "hidden_dim": int(actor.hidden_dim),
        "actor": actor.state_dict(),
        "metadata": metadata or {},
    }
    torch.save(payload, path)


def load_subgoal_actor_checkpoint(path: str, device: torch.device) -> tuple[SubgoalActor, Dict[str, Any]]:
    ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict) or "actor" not in ckpt:
        raise ValueError(f"非法的 subgoal actor checkpoint: {path}")
    actor = SubgoalActor(
        obs_dim=int(ckpt["obs_dim"]),
        hidden_dim=int(ckpt.get("hidden_dim", 256)),
    )
    actor.load_state_dict(ckpt["actor"])
    actor.to(device)
    actor.eval()
    return actor, dict(ckpt.get("metadata", {}))


def train_subgoal_actor(
    actor: SubgoalActor,
    agent: GoalConditionedAgentBase,
    env_factory: Callable[[], CommInspectionDubinsUAV2D],
    device: torch.device,
    cfg: SubgoalActorTrainConfig,
    *,
    log_fn: Optional[Callable[[int, Dict[str, float]], None]] = None,
    checkpoint_fn: Optional[Callable[[int, Dict[str, float]], None]] = None,
) -> Dict[str, float]:
    optimizer = torch.optim.Adam(actor.parameters(), lr=float(cfg.lr))
    actor.train()

    rng = np.random.default_rng(int(cfg.seed))
    env = env_factory()

    final_metrics: Dict[str, float] = {}
    progress = tqdm(range(1, int(cfg.train_steps) + 1), desc="SubgoalActor", leave=True)
    for step in progress:
        batch_obs = []
        batch_goals = []
        batch_teacher = []
        raw_valid_flags = []
        repair_distances = []
        repair_dthetas = []
        raw_task_scores = []
        repaired_task_scores = []

        for _ in range(int(cfg.batch_size)):
            seed = int(rng.integers(0, 1_000_000_000))
            obs, _ = env.reset(seed=seed)
            goal_obs = env.state_to_observation(np.asarray(env.goal, dtype=np.float32))
            teacher = select_teacher_subgoal(
                actor,
                agent,
                env,
                obs,
                goal_obs,
                device=device,
                num_candidates=int(cfg.num_candidates),
                lambda_final=float(cfg.lambda_final),
                lambda_task=float(cfg.lambda_task),
                rng=rng,
            )
            batch_obs.append(np.asarray(obs, dtype=np.float32))
            batch_goals.append(np.asarray(goal_obs, dtype=np.float32))
            batch_teacher.append(np.asarray(teacher["teacher_subgoal"], dtype=np.float32))
            raw_valid_flags.append(float(teacher["raw_valid"]))
            repair_distances.append(float(teacher["repair_distance"]))
            repair_dthetas.append(float(teacher["repair_dtheta"]))
            raw_task_scores.append(float(teacher["raw_task_score"]))
            repaired_task_scores.append(float(teacher["repaired_task_score"]))

        obs_t = torch.as_tensor(np.asarray(batch_obs), dtype=torch.float32, device=device)
        goals_t = torch.as_tensor(np.asarray(batch_goals), dtype=torch.float32, device=device)
        teacher_t = torch.as_tensor(np.asarray(batch_teacher), dtype=torch.float32, device=device)

        pred_state, pred_sin, pred_cos = actor.predict_state_tensor(obs_t, goals_t, env)
        target_sin = torch.sin(teacher_t[:, 2])
        target_cos = torch.cos(teacher_t[:, 2])

        pos_loss = F.mse_loss(pred_state[:, :2], teacher_t[:, :2])
        heading_loss = F.mse_loss(pred_sin, target_sin) + F.mse_loss(pred_cos, target_cos)
        loss = pos_loss + heading_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        final_metrics = {
            "loss": float(loss.item()),
            "pos_loss": float(pos_loss.item()),
            "heading_loss": float(heading_loss.item()),
            "raw_actor_output_valid_rate": float(np.mean(raw_valid_flags)),
            "mean_repair_distance": float(np.mean(repair_distances)),
            "mean_repair_dtheta": float(np.mean(repair_dthetas)),
            "mean_taskscore_raw_subgoal": float(np.mean(raw_task_scores)),
            "mean_taskscore_repaired_subgoal": float(np.mean(repaired_task_scores)),
        }
        progress.set_postfix(
            loss=f"{final_metrics['loss']:.3f}",
            raw_valid=f"{final_metrics['raw_actor_output_valid_rate']:.3f}",
            repair=f"{final_metrics['mean_repair_distance']:.3f}",
        )
        if log_fn is not None:
            log_fn(step, final_metrics)
        if checkpoint_fn is not None and int(cfg.save_interval) > 0 and step % int(cfg.save_interval) == 0:
            checkpoint_fn(step, final_metrics)

    progress.close()
    actor.eval()
    return final_metrics
