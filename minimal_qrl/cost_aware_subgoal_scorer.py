from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from minimal_qrl.envs import CommInspectionDubinsUAV2D
from minimal_qrl.eval.dubins_execution_mode_eval import DubinsLookaheadConfig
from minimal_qrl.gc_agents import GoalConditionedAgentBase
from minimal_qrl.subgoal_actor import SubgoalActor, _sample_teacher_candidate_set


NAV_FEATURE_KEYS = (
    "d_qrl_obs_cand",
    "d_qrl_cand_goal",
    "d_qrl_obs_goal",
    "c_reach_est",
    "task_score",
    "obs_margin",
    "comm_margin",
    "task_feasible",
)

TOP_MODEL_COST_CONTEXT_KEYS = (
    "collision_cost",
    "out_of_bounds_cost",
    "communication_break_cost",
    "observation_violation_cost_weight",
    "communication_violation_cost_weight",
    "observation_failure_cost",
    "taskscore_beta_obs",
    "taskscore_beta_comm",
    "taskscore_beta_feas",
    "planner_alpha_subgoal",
    "planner_alpha_final",
    "planner_alpha_task_terminal",
)


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    )


@dataclass
class CostAwareSubgoalScorerTrainConfig:
    train_steps: int = 5000
    batch_size: int = 16
    lr: float = 3e-4
    hidden_dim: int = 256
    num_candidates: int = 64
    rollout_steps: int = 5
    seed: int = 0
    save_interval: int = 1000


class CostAwareSubgoalScorer(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        *,
        nav_feature_dim: int = len(NAV_FEATURE_KEYS),
        cost_context_dim: int = len(TOP_MODEL_COST_CONTEXT_KEYS),
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.nav_feature_dim = int(nav_feature_dim)
        self.cost_context_dim = int(cost_context_dim)
        self.hidden_dim = int(hidden_dim)
        in_dim = self.obs_dim * 3 + self.nav_feature_dim + self.cost_context_dim
        self.net = _mlp(in_dim, self.hidden_dim, 1)

    def forward(
        self,
        obs: torch.Tensor,
        goal_obs: torch.Tensor,
        cand_obs: torch.Tensor,
        nav_features: torch.Tensor,
        cost_context: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([obs, goal_obs, cand_obs, nav_features, cost_context], dim=-1)
        return self.net(x).squeeze(-1)

    @torch.no_grad()
    def predict_costs(
        self,
        obs: np.ndarray,
        goal_obs: np.ndarray,
        cand_obs: np.ndarray,
        nav_features: np.ndarray,
        cost_context: np.ndarray,
        *,
        device: torch.device,
    ) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        goal_t = torch.as_tensor(goal_obs, dtype=torch.float32, device=device)
        cand_t = torch.as_tensor(cand_obs, dtype=torch.float32, device=device)
        nav_t = torch.as_tensor(nav_features, dtype=torch.float32, device=device)
        cost_t = torch.as_tensor(cost_context, dtype=torch.float32, device=device)
        return self.forward(obs_t, goal_t, cand_t, nav_t, cost_t).detach().cpu().numpy().astype(np.float32)


def build_top_model_cost_context(
    env: CommInspectionDubinsUAV2D,
    lookahead_cfg: DubinsLookaheadConfig,
) -> np.ndarray:
    return np.asarray(
        [
            float(env.collision_cost),
            float(env.out_of_bounds_cost),
            float(env.communication_break_cost),
            float(env.observation_violation_cost_weight),
            float(env.communication_violation_cost_weight),
            float(env.observation_failure_cost),
            float(env.taskscore_beta_obs),
            float(env.taskscore_beta_comm),
            float(env.taskscore_beta_feas),
            float(lookahead_cfg.alpha_subgoal),
            float(lookahead_cfg.alpha_final),
            float(lookahead_cfg.alpha_task_terminal),
        ],
        dtype=np.float32,
    )


def compute_top_model_nav_features(
    agent: GoalConditionedAgentBase,
    env: CommInspectionDubinsUAV2D,
    obs: np.ndarray,
    goal_obs: np.ndarray,
    candidate: np.ndarray,
) -> np.ndarray:
    state = env.observation_to_state(obs)
    cand_obs = env.state_to_observation(candidate)
    return np.asarray(
        [
            float(agent.value(obs, cand_obs)),
            float(agent.value(cand_obs, goal_obs)),
            float(agent.value(obs, goal_obs)),
            float(env.compute_goal_reaching_cost_estimate(state=state, goal=candidate)),
            float(env.compute_task_score(candidate)),
            float(env.compute_observation_margin(candidate)),
            float(env.compute_comm_quality(candidate)["margin"]),
            1.0 if env.is_task_feasible(candidate) else 0.0,
        ],
        dtype=np.float32,
    )


def build_cost_aware_candidate_bundle(
    actor: SubgoalActor,
    env: CommInspectionDubinsUAV2D,
    obs: np.ndarray,
    goal_obs: np.ndarray,
    *,
    device: torch.device,
    num_candidates: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    raw_subgoal = actor.predict_state(obs, goal_obs, env, device=device)
    repair_info = env.repair_state_with_info(raw_subgoal)
    repaired_subgoal = np.asarray(repair_info["repaired_state"], dtype=np.float32)
    candidates = _sample_teacher_candidate_set(
        env,
        raw_subgoal,
        repaired_subgoal,
        int(num_candidates),
        rng,
    )
    repair_metrics = env.compute_repair_metrics(raw_subgoal, repaired_subgoal)
    return {
        "raw_subgoal": np.asarray(raw_subgoal, dtype=np.float32),
        "repaired_subgoal": np.asarray(repaired_subgoal, dtype=np.float32),
        "candidates": np.asarray(candidates, dtype=np.float32),
        "raw_valid": bool(env.is_valid_state(raw_subgoal)),
        "used_nearby_repair": bool(repair_info["used_nearby_repair"]),
        "used_global_repair_fallback": bool(repair_info["used_global_fallback"]),
        "repair_distance": float(repair_metrics["repair_distance"]),
        "repair_dtheta": float(repair_metrics["repair_dtheta"]),
        "raw_task_score": float(env.compute_task_score(raw_subgoal)),
        "repaired_task_score": float(env.compute_task_score(repaired_subgoal)),
        "candidate_count": int(len(candidates)),
    }


def build_cost_aware_inputs(
    scorer: CostAwareSubgoalScorer,
    agent: GoalConditionedAgentBase,
    env: CommInspectionDubinsUAV2D,
    obs: np.ndarray,
    goal_obs: np.ndarray,
    candidates: np.ndarray,
    lookahead_cfg: DubinsLookaheadConfig,
) -> Dict[str, np.ndarray]:
    candidates = np.asarray(candidates, dtype=np.float32).reshape(-1, 3)
    num_candidates = int(candidates.shape[0])
    cand_obs = np.asarray(
        [env.state_to_observation(candidate) for candidate in candidates],
        dtype=np.float32,
    )
    nav_features = np.asarray(
        [
            compute_top_model_nav_features(
                agent=agent,
                env=env,
                obs=obs,
                goal_obs=goal_obs,
                candidate=candidate,
            )
            for candidate in candidates
        ],
        dtype=np.float32,
    )
    cost_context = np.repeat(
        build_top_model_cost_context(env, lookahead_cfg)[None],
        num_candidates,
        axis=0,
    ).astype(np.float32)
    obs_batch = np.repeat(np.asarray(obs, dtype=np.float32)[None], num_candidates, axis=0).astype(np.float32)
    goal_batch = np.repeat(np.asarray(goal_obs, dtype=np.float32)[None], num_candidates, axis=0).astype(np.float32)
    return {
        "obs": obs_batch,
        "goal_obs": goal_batch,
        "cand_obs": cand_obs,
        "nav_features": nav_features,
        "cost_context": cost_context,
    }


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _evaluate_candidate_sequences(
    agent: GoalConditionedAgentBase,
    env: CommInspectionDubinsUAV2D,
    goal_obs: np.ndarray,
    cfg: DubinsLookaheadConfig,
    omegas: np.ndarray,
    base_state: dict,
    *,
    subgoal_state: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(omegas.shape[0])
    costs = np.zeros((n,), dtype=np.float32)
    first_actions = omegas[:, 0].astype(np.float32).copy()
    subgoal_obs = env.state_to_observation(subgoal_state) if subgoal_state is not None else None

    for i in range(n):
        env.set_state(base_state)
        total_cost = 0.0
        reached_subgoal = False
        success = False

        for t in range(int(omegas.shape[1])):
            w = float(omegas[i, t])
            action = np.array([w], dtype=np.float32)
            _obs, _reward, terminated, truncated, info = env.step(action)

            if bool(cfg.use_env_stage_cost):
                total_cost += _safe_float(info.get("cost_total"))
            else:
                if cfg.step_cost_weight > 0.0:
                    total_cost += float(cfg.step_cost_weight) * abs(w)
                if cfg.collision_penalty > 0.0 and bool(info.get("collision", False)):
                    total_cost += float(cfg.collision_penalty)

            if subgoal_state is not None and env.is_subgoal_reached(
                env.state,
                subgoal_state,
                pos_tolerance=float(cfg.subgoal_reached_pos_tolerance),
                theta_tolerance=float(cfg.subgoal_reached_theta_tolerance),
            ):
                reached_subgoal = True

            if terminated:
                success = True
                break
            if truncated:
                break

        terminal_obs = env.state_to_observation(env.state)
        terminal_task_score = env.compute_task_score(env.state)
        terminal_cost = 0.0
        if not success:
            terminal_cost += float(cfg.alpha_final) * float(agent.value(terminal_obs, goal_obs))
            terminal_cost -= float(cfg.alpha_task_terminal) * float(terminal_task_score)
            if subgoal_obs is not None and not reached_subgoal:
                terminal_cost += float(cfg.alpha_subgoal) * float(agent.value(terminal_obs, subgoal_obs))

        costs[i] = float(total_cost + terminal_cost)

    env.set_state(base_state)
    return costs, first_actions


def _plan_low_level_action(
    agent: GoalConditionedAgentBase,
    env: CommInspectionDubinsUAV2D,
    goal_obs: np.ndarray,
    cfg: DubinsLookaheadConfig,
    *,
    subgoal_state: Optional[np.ndarray] = None,
) -> np.ndarray:
    horizon = max(1, int(cfg.horizon))
    num_sequences = max(1, int(cfg.num_sequences))
    base_state = env.get_state()

    low = float(env.action_space.low[0])
    high = float(env.action_space.high[0])
    n_bias = int(min(max(0, cfg.biased_sequences), num_sequences))
    n_rand = int(max(0, num_sequences - n_bias))
    rand = (
        np.random.uniform(low, high, size=(n_rand, horizon)).astype(np.float32)
        if n_rand > 0
        else np.zeros((0, horizon), dtype=np.float32)
    )
    if n_bias > 0:
        bias = np.zeros((n_bias, horizon), dtype=np.float32)
        desired = subgoal_state if subgoal_state is not None else np.asarray(env.goal, dtype=np.float32)
        dx = float(desired[0] - env.state[0])
        dy = float(desired[1] - env.state[1])
        err = env._normalize_angle(float(np.arctan2(dy, dx) - env.state[2]))
        w0 = float(np.clip(float(cfg.bias_kp) * err, low, high))
        bias[0, :] = w0
        for idx in range(1, n_bias):
            scale = max(0.0, 1.0 - 0.15 * float(idx))
            bias[idx, :] = float(np.clip(w0 * scale, low, high))
    else:
        bias = np.zeros((0, horizon), dtype=np.float32)
    omegas0 = (
        np.concatenate([bias, rand], axis=0)
        if (n_bias + n_rand) > 0
        else np.zeros((1, horizon), dtype=np.float32)
    )

    if cfg.use_cem:
        om_range = float(high - low)
        std = np.full((horizon,), float(cfg.cem_std_init_frac) * 0.5 * om_range, dtype=np.float32)
        mean = (
            np.mean(omegas0, axis=0).astype(np.float32)
            if omegas0.shape[0] > 0
            else np.zeros((horizon,), dtype=np.float32)
        )
        n_elite = max(1, int(float(cfg.cem_elite_frac) * float(num_sequences)))
        best_first = np.array([0.0], dtype=np.float32)
        best_cost = float("inf")

        for _ in range(max(1, int(cfg.cem_iters))):
            samples = np.random.normal(
                loc=mean[None, :],
                scale=std[None, :],
                size=(num_sequences, horizon),
            ).astype(np.float32)
            samples = np.clip(samples, low, high)
            if n_bias > 0:
                samples[: min(n_bias, samples.shape[0])] = bias[: min(n_bias, samples.shape[0])]
            costs, firsts = _evaluate_candidate_sequences(
                agent,
                env,
                goal_obs,
                cfg,
                samples,
                base_state,
                subgoal_state=subgoal_state,
            )
            idx = int(np.argmin(costs))
            if float(costs[idx]) < best_cost:
                best_cost = float(costs[idx])
                best_first = np.array([float(firsts[idx])], dtype=np.float32)
            elite = samples[np.argsort(costs)[:n_elite]]
            mean = np.mean(elite, axis=0).astype(np.float32)
            std = (np.std(elite, axis=0) + 1e-4).astype(np.float32)

        env.set_state(base_state)
        return best_first

    costs, firsts = _evaluate_candidate_sequences(
        agent,
        env,
        goal_obs,
        cfg,
        omegas0,
        base_state,
        subgoal_state=subgoal_state,
    )
    best_idx = int(np.argmin(costs))
    env.set_state(base_state)
    return np.array([float(firsts[best_idx])], dtype=np.float32)


def rollout_cost_label_for_subgoal(
    agent: GoalConditionedAgentBase,
    env: CommInspectionDubinsUAV2D,
    goal_obs: np.ndarray,
    lookahead_cfg: DubinsLookaheadConfig,
    *,
    subgoal_state: np.ndarray,
    rollout_steps: int,
    base_state: Optional[dict] = None,
) -> Dict[str, Any]:
    if base_state is None:
        base_state = env.get_state()
    env.set_state(base_state)
    total_cost = 0.0
    reached_subgoal = False
    success = False
    truncated = False

    for _ in range(max(1, int(rollout_steps))):
        action = _plan_low_level_action(
            agent,
            env,
            goal_obs,
            lookahead_cfg,
            subgoal_state=subgoal_state,
        )
        _obs, _reward, terminated, truncated, info = env.step(action)
        if bool(lookahead_cfg.use_env_stage_cost):
            total_cost += _safe_float(info.get("cost_total"))
        else:
            if lookahead_cfg.step_cost_weight > 0.0:
                total_cost += float(lookahead_cfg.step_cost_weight) * abs(float(action[0]))
            if lookahead_cfg.collision_penalty > 0.0 and bool(info.get("collision", False)):
                total_cost += float(lookahead_cfg.collision_penalty)

        if env.is_subgoal_reached(
            env.state,
            subgoal_state,
            pos_tolerance=float(lookahead_cfg.subgoal_reached_pos_tolerance),
            theta_tolerance=float(lookahead_cfg.subgoal_reached_theta_tolerance),
        ):
            reached_subgoal = True

        if terminated:
            success = True
            break
        if truncated:
            break

    terminal_obs = env.state_to_observation(env.state)
    terminal_task_score = env.compute_task_score(env.state)
    terminal_cost = 0.0
    if not success:
        terminal_cost += float(lookahead_cfg.alpha_final) * float(agent.value(terminal_obs, goal_obs))
        terminal_cost -= float(lookahead_cfg.alpha_task_terminal) * float(terminal_task_score)
        if not reached_subgoal:
            subgoal_obs = env.state_to_observation(subgoal_state)
            terminal_cost += float(lookahead_cfg.alpha_subgoal) * float(agent.value(terminal_obs, subgoal_obs))

    label = float(total_cost + terminal_cost)
    info = {
        "rollout_cost_label": label,
        "rollout_stage_cost": float(total_cost),
        "rollout_terminal_cost": float(terminal_cost),
        "success": bool(success),
        "truncated": bool(truncated),
        "reached_subgoal": bool(reached_subgoal),
        "final_state": np.asarray(env.state, dtype=np.float32).copy(),
    }
    env.set_state(base_state)
    return info


def evaluate_candidate_rollout_labels(
    agent: GoalConditionedAgentBase,
    env: CommInspectionDubinsUAV2D,
    goal_obs: np.ndarray,
    lookahead_cfg: DubinsLookaheadConfig,
    candidates: np.ndarray,
    *,
    rollout_steps: int,
    base_state: Optional[dict] = None,
) -> Dict[str, Any]:
    candidates = np.asarray(candidates, dtype=np.float32).reshape(-1, 3)
    if base_state is None:
        base_state = env.get_state()
    labels = []
    details = []
    for candidate in candidates:
        result = rollout_cost_label_for_subgoal(
            agent,
            env,
            goal_obs,
            lookahead_cfg,
            subgoal_state=candidate,
            rollout_steps=rollout_steps,
            base_state=base_state,
        )
        labels.append(float(result["rollout_cost_label"]))
        details.append(result)
    return {
        "labels": np.asarray(labels, dtype=np.float32),
        "details": details,
    }


@torch.no_grad()
def select_cost_aware_subgoal(
    actor: SubgoalActor,
    scorer: CostAwareSubgoalScorer,
    agent: GoalConditionedAgentBase,
    env: CommInspectionDubinsUAV2D,
    obs: np.ndarray,
    goal_obs: np.ndarray,
    *,
    actor_device: torch.device,
    scorer_device: torch.device,
    lookahead_cfg: DubinsLookaheadConfig,
    num_candidates: int,
    rollout_steps: int,
    rng: np.random.Generator,
    evaluate_rollout_labels: bool = True,
) -> Dict[str, Any]:
    bundle = build_cost_aware_candidate_bundle(
        actor,
        env,
        obs,
        goal_obs,
        device=actor_device,
        num_candidates=num_candidates,
        rng=rng,
    )
    candidates = np.asarray(bundle["candidates"], dtype=np.float32)
    inputs = build_cost_aware_inputs(
        scorer,
        agent,
        env,
        obs,
        goal_obs,
        candidates,
        lookahead_cfg,
    )
    pred_costs = scorer.predict_costs(
        inputs["obs"],
        inputs["goal_obs"],
        inputs["cand_obs"],
        inputs["nav_features"],
        inputs["cost_context"],
        device=scorer_device,
    )
    best_idx = int(np.argmin(pred_costs)) if len(pred_costs) > 0 else 0
    executed_subgoal = np.asarray(candidates[best_idx], dtype=np.float32)

    rollout_labels = None
    top1_match = None
    mse = None
    selected_rollout_label = None
    if evaluate_rollout_labels:
        label_result = evaluate_candidate_rollout_labels(
            agent,
            env,
            goal_obs,
            lookahead_cfg,
            candidates,
            rollout_steps=rollout_steps,
        )
        rollout_labels = np.asarray(label_result["labels"], dtype=np.float32)
        if rollout_labels.shape == pred_costs.shape and rollout_labels.size > 0:
            label_best_idx = int(np.argmin(rollout_labels))
            top1_match = 1.0 if label_best_idx == best_idx else 0.0
            mse = float(np.mean((pred_costs - rollout_labels) ** 2))
            selected_rollout_label = float(rollout_labels[best_idx])

    return {
        **bundle,
        "executed_subgoal": executed_subgoal,
        "used_teacher_fallback": False,
        "executed_task_score": float(env.compute_task_score(executed_subgoal)),
        "pred_costs": pred_costs,
        "selected_pred_cost": float(pred_costs[best_idx]) if len(pred_costs) > 0 else 0.0,
        "rollout_labels": rollout_labels,
        "selected_rollout_label": selected_rollout_label,
        "top1_match": top1_match,
        "eval_mse": mse,
        "selected_by": "cost_aware",
    }


def save_cost_aware_subgoal_scorer_checkpoint(
    path: str,
    scorer: CostAwareSubgoalScorer,
    *,
    train_step: int,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "train_step": int(train_step),
        "obs_dim": int(scorer.obs_dim),
        "nav_feature_dim": int(scorer.nav_feature_dim),
        "cost_context_dim": int(scorer.cost_context_dim),
        "hidden_dim": int(scorer.hidden_dim),
        "scorer": scorer.state_dict(),
        "metadata": metadata or {},
    }
    torch.save(payload, path)


def load_cost_aware_subgoal_scorer_checkpoint(
    path: str,
    device: torch.device,
) -> tuple[CostAwareSubgoalScorer, Dict[str, Any]]:
    ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict) or "scorer" not in ckpt:
        raise ValueError(f"非法的 cost-aware scorer checkpoint: {path}")
    scorer = CostAwareSubgoalScorer(
        obs_dim=int(ckpt["obs_dim"]),
        nav_feature_dim=int(ckpt.get("nav_feature_dim", len(NAV_FEATURE_KEYS))),
        cost_context_dim=int(ckpt.get("cost_context_dim", len(TOP_MODEL_COST_CONTEXT_KEYS))),
        hidden_dim=int(ckpt.get("hidden_dim", 256)),
    )
    scorer.load_state_dict(ckpt["scorer"])
    scorer.to(device)
    scorer.eval()
    return scorer, dict(ckpt.get("metadata", {}))


def train_cost_aware_subgoal_scorer(
    scorer: CostAwareSubgoalScorer,
    actor: SubgoalActor,
    agent: GoalConditionedAgentBase,
    env_factory: Callable[[], CommInspectionDubinsUAV2D],
    *,
    actor_device: torch.device,
    scorer_device: torch.device,
    lookahead_cfg: DubinsLookaheadConfig,
    cfg: CostAwareSubgoalScorerTrainConfig,
    log_fn: Optional[Callable[[int, Dict[str, float]], None]] = None,
    checkpoint_fn: Optional[Callable[[int, Dict[str, float]], None]] = None,
) -> Dict[str, float]:
    optimizer = torch.optim.Adam(scorer.parameters(), lr=float(cfg.lr))
    scorer.train()
    actor.eval()

    rng = np.random.default_rng(int(cfg.seed))
    env = env_factory()
    final_metrics: Dict[str, float] = {}
    progress = tqdm(range(1, int(cfg.train_steps) + 1), desc="CostAwareTopModel", leave=True)
    for step in progress:
        batch_obs = []
        batch_goals = []
        batch_cands = []
        batch_nav = []
        batch_cost_ctx = []
        batch_labels = []
        group_top1 = []
        group_selected_pred = []
        group_selected_label = []

        for _ in range(int(cfg.batch_size)):
            seed = int(rng.integers(0, 1_000_000_000))
            obs, _ = env.reset(seed=seed)
            goal_obs = env.state_to_observation(np.asarray(env.goal, dtype=np.float32))
            bundle = build_cost_aware_candidate_bundle(
                actor,
                env,
                obs,
                goal_obs,
                device=actor_device,
                num_candidates=int(cfg.num_candidates),
                rng=rng,
            )
            inputs = build_cost_aware_inputs(
                scorer,
                agent,
                env,
                obs,
                goal_obs,
                bundle["candidates"],
                lookahead_cfg,
            )
            label_result = evaluate_candidate_rollout_labels(
                agent,
                env,
                goal_obs,
                lookahead_cfg,
                bundle["candidates"],
                rollout_steps=int(cfg.rollout_steps),
            )
            labels = np.asarray(label_result["labels"], dtype=np.float32)
            batch_obs.append(inputs["obs"])
            batch_goals.append(inputs["goal_obs"])
            batch_cands.append(inputs["cand_obs"])
            batch_nav.append(inputs["nav_features"])
            batch_cost_ctx.append(inputs["cost_context"])
            batch_labels.append(labels)

        obs_t = torch.as_tensor(np.concatenate(batch_obs, axis=0), dtype=torch.float32, device=scorer_device)
        goals_t = torch.as_tensor(np.concatenate(batch_goals, axis=0), dtype=torch.float32, device=scorer_device)
        cands_t = torch.as_tensor(np.concatenate(batch_cands, axis=0), dtype=torch.float32, device=scorer_device)
        nav_t = torch.as_tensor(np.concatenate(batch_nav, axis=0), dtype=torch.float32, device=scorer_device)
        cost_ctx_t = torch.as_tensor(np.concatenate(batch_cost_ctx, axis=0), dtype=torch.float32, device=scorer_device)
        labels_t = torch.as_tensor(np.concatenate(batch_labels, axis=0), dtype=torch.float32, device=scorer_device)

        pred_t = scorer(obs_t, goals_t, cands_t, nav_t, cost_ctx_t)
        loss = F.mse_loss(pred_t, labels_t)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        offset = 0
        pred_np = pred_t.detach().cpu().numpy().astype(np.float32)
        for labels in batch_labels:
            size = int(len(labels))
            group_pred = pred_np[offset:offset + size]
            offset += size
            pred_best_idx = int(np.argmin(group_pred))
            label_best_idx = int(np.argmin(labels))
            group_top1.append(1.0 if pred_best_idx == label_best_idx else 0.0)
            group_selected_pred.append(float(group_pred[pred_best_idx]))
            group_selected_label.append(float(labels[pred_best_idx]))

        final_metrics = {
            "loss": float(loss.item()),
            "top1_match_rate": float(np.mean(group_top1)) if group_top1 else 0.0,
            "mean_selected_pred_cost": float(np.mean(group_selected_pred)) if group_selected_pred else 0.0,
            "mean_selected_rollout_cost_label": float(np.mean(group_selected_label)) if group_selected_label else 0.0,
            "mean_label_cost": float(np.mean(np.concatenate(batch_labels, axis=0))) if batch_labels else 0.0,
        }
        progress.set_postfix(
            loss=f"{final_metrics['loss']:.3f}",
            top1=f"{final_metrics['top1_match_rate']:.3f}",
        )
        if log_fn is not None:
            log_fn(step, final_metrics)
        if checkpoint_fn is not None and int(cfg.save_interval) > 0 and step % int(cfg.save_interval) == 0:
            checkpoint_fn(step, final_metrics)

    progress.close()
    scorer.eval()
    return final_metrics
