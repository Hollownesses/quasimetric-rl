"""Small working-set feasible directions for the original one-step QRL edges.

The dual variables in the QP solver are transient numerical variables.  They
are never fitted, checkpointed, or used as training labels.
"""

from __future__ import annotations

import time
from typing import Optional

import attrs
import numpy as np
import torch
from torch import nn

from ....data import BatchData, Dataset
from ...utils import LossResult
from . import CriticBatchInfo
from .global_push import GlobalPushLoss
from .local_constraint import LocalConstraintLoss


def solve_direction(
    g: torch.Tensor, A: torch.Tensor, residual: torch.Tensor, *,
    kappa: float, elastic_penalty: Optional[float] = None,
    sweeps: int = 1000, tolerance: float = 1e-7,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project g onto A v <= -kappa*r using a small dual Gram QP.

    A box bound on alpha is the exact dual of an L1 elastic slack.  The returned
    slack is measured in the original, unscaled linearized constraint units.
    """
    if A.numel() == 0:
        return g.clone(), g.new_zeros(0), g.new_zeros(0)
    # MPS has no float64 kernels; the 32-64 row Gram solve lives on CPU.
    a64 = A.detach().to(device="cpu", dtype=torch.float64)
    gram = (a64 @ a64.T).numpy()
    q = (a64 @ g.detach().to(device="cpu", dtype=torch.float64)
         + kappa * residual.detach().to(device="cpu", dtype=torch.float64)).numpy()
    alpha = np.zeros(len(q), dtype=np.float64)
    ka = np.zeros(len(q), dtype=np.float64)
    upper = np.inf if elastic_penalty is None else float(elastic_penalty)
    for _ in range(sweeps):
        change = 0.0
        for i in range(len(q)):
            if gram[i, i] <= 1e-20:
                continue
            updated = np.clip(alpha[i] + (q[i] - ka[i]) / gram[i, i], 0.0, upper)
            delta = updated - alpha[i]
            if delta:
                alpha[i] = updated
                ka += delta * gram[:, i]
                change = max(change, abs(delta))
        if change <= tolerance:
            break
    alpha_t = torch.as_tensor(alpha, device=A.device, dtype=A.dtype)
    v = g - A.T @ alpha_t
    slack = (A @ v + kappa * residual).clamp_min(0)
    return v, alpha_t, slack


@attrs.define(kw_only=True)
class FeasibleDirectionConf:
    kappa: float = 1.0
    step_size: float = 5e-5
    trust_radius: float = 0.05
    backtracks: int = 10
    work_edges: int = 48
    candidate_multiplier: int = 4
    audit_interval: int = 100
    audit_chunk: int = 1024
    feasibility_tolerance: float = 1e-5
    anchor_tolerance: float = 1e-6
    elastic_penalty: float = 100.0
    qp_sweeps: int = 1000

    def __attrs_post_init__(self):
        if min(self.kappa, self.step_size, self.trust_radius, self.elastic_penalty) <= 0:
            raise ValueError("feasible-direction scales must be positive")
        if min(self.work_edges, self.candidate_multiplier, self.audit_interval,
               self.audit_chunk, self.qp_sweeps) <= 0 or self.backtracks < 0:
            raise ValueError("feasible-direction counts must be positive")
        if min(self.feasibility_tolerance, self.anchor_tolerance) < 0:
            raise ValueError("feasibility tolerances must be nonnegative")


class FeasibleDirectionUpdater(nn.Module):
    def __init__(self, conf: FeasibleDirectionConf):
        super().__init__()
        self.conf = conf
        self.replay: Optional[Dataset] = None
        self.register_buffer("steps", torch.zeros((), dtype=torch.long))
        self.register_buffer("history_indices", torch.empty(0, dtype=torch.long))
        self.register_buffer("audit_high_indices", torch.empty(0, dtype=torch.long))
        self.register_buffer("audit_near_indices", torch.empty(0, dtype=torch.long))
        self._audit_info: dict[str, float] = {}
        self._audited_this_process = False

    def set_replay(self, replay: Dataset) -> None:
        self.replay = replay

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # The cached edge set changes size between checkpoints.
        for name in ("history_indices", "audit_high_indices", "audit_near_indices"):
            key = prefix + name
            if key in state_dict:
                setattr(self, name, torch.empty_like(state_dict[key]))
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def _edge_values(self, critic: nn.Module, local: LocalConstraintLoss,
                     indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.replay is not None
        indices = indices.cpu().long()
        raw = self.replay.raw_data
        obs_indices = indices + self.replay.indices_to_episode_indices[indices]
        device = next(critic.parameters()).device
        obs = raw.all_observations[obs_indices].to(device)
        next_obs = raw.all_observations[obs_indices + 1].to(device)
        dist = critic(obs, next_obs)
        if local.cost_source == "fixed":
            cost = torch.full_like(dist, local.step_cost)
        else:
            cost = (-raw.rewards[indices].to(device=device, dtype=dist.dtype)).clamp_min(0)
        return dist - cost, cost

    @torch.no_grad()
    def audit(self, critic: nn.Module, local: LocalConstraintLoss) -> dict[str, float]:
        assert self.replay is not None
        n = len(self.replay)
        residuals, costs = [], []
        for start in range(0, n, self.conf.audit_chunk):
            r, c = self._edge_values(critic, local,
                                     torch.arange(start, min(n, start + self.conf.audit_chunk)))
            residuals.append(r.detach().cpu())
            costs.append(c.detach().cpu())
        r, c = torch.cat(residuals), torch.cat(costs)
        count = min(self.conf.work_edges, n)
        self.audit_high_indices = torch.topk(r, count).indices.long()
        self.audit_near_indices = torch.topk(-r.abs(), count).indices.long()
        positive = r.clamp_min(0)
        anchors = c == 0
        self._audit_info = {
            "audit_edges": float(n),
            "audit_max_violation": float(positive.max()),
            "audit_tail_violation": float(torch.quantile(positive, 0.99)),
            "audit_violation_fraction": float((r > self.conf.feasibility_tolerance).float().mean()),
            "audit_anchor_count": float(anchors.sum()),
            "audit_anchor_max_violation": float(positive[anchors].max()) if bool(anchors.any()) else 0.0,
            "audit_anchor_violation_fraction": float(
                (r[anchors] > self.conf.anchor_tolerance).float().mean())
                if bool(anchors.any()) else 0.0,
        }
        self._audited_this_process = True
        return self._audit_info

    @torch.no_grad()
    def _workset(self, critic: nn.Module, local: LocalConstraintLoss) -> torch.Tensor:
        assert self.replay is not None
        n, m = len(self.replay), min(self.conf.work_edges, len(self.replay))
        candidates = torch.randint(n, (min(n, m * self.conf.candidate_multiplier),))
        r, _ = self._edge_values(critic, local, candidates)
        r = r.detach().cpu()
        random_n = max(1, m // 3)
        high_n = max(1, m // 4)
        near_n = max(1, m // 8)
        history_n = max(1, m // 8)
        audit_n = max(1, m // 12)
        def valid_cached(values: torch.Tensor, count: int) -> torch.Tensor:
            values = values.cpu()
            return values[(values >= 0) & (values < n)][:count]
        parts = [candidates[:random_n], candidates[torch.topk(r, min(high_n, len(r))).indices],
                 candidates[torch.topk(-r.abs(), min(near_n, len(r))).indices],
                 valid_cached(self.history_indices, history_n),
                 valid_cached(self.audit_high_indices, audit_n),
                 valid_cached(self.audit_near_indices, audit_n)]
        ordered = list(dict.fromkeys(torch.cat(parts).tolist()))[:m]
        if len(ordered) < m:
            ordered = list(dict.fromkeys(ordered + torch.randperm(n).tolist()))[:m]
        return torch.tensor(ordered, dtype=torch.long)

    @staticmethod
    def _flatten(grads, params) -> torch.Tensor:
        return torch.cat([(torch.zeros_like(p) if grad is None else grad).reshape(-1)
                          for grad, p in zip(grads, params)])

    @staticmethod
    def _write_flat(params, original, direction, scale: float) -> None:
        offset = 0
        with torch.no_grad():
            for p, old in zip(params, original):
                size = p.numel()
                p.copy_(old + scale * direction[offset:offset + size].view_as(p))
                offset += size

    def step(self, data: BatchData, batch_info: CriticBatchInfo,
             push: GlobalPushLoss, local: LocalConstraintLoss) -> LossResult:
        if self.replay is None:
            raise RuntimeError("feasible-direction mode requires set_replay(dataset)")
        started = time.perf_counter()
        critic = batch_info.critic
        params = [p for p in critic.parameters() if p.requires_grad]
        self.steps += 1
        audited = (not self._audited_this_process or
                   int(self.steps) % self.conf.audit_interval == 0)
        if audited:
            self.audit(critic, local)
        indices = self._workset(critic, local)
        residual, cost = self._edge_values(critic, local, indices)
        push_result = push(data, batch_info)
        objective = -push_result.loss
        g = self._flatten(torch.autograd.grad(objective, params, retain_graph=False,
                                               allow_unused=True), params).detach()
        rows = []
        for i in range(len(indices)):
            row = torch.autograd.grad(residual[i], params,
                                      retain_graph=i + 1 < len(indices), allow_unused=True)
            rows.append(self._flatten(row, params).detach())
        A = torch.stack(rows)
        r0 = residual.detach()
        anchor = cost.detach() == 0
        tolerance = torch.where(anchor, self.conf.anchor_tolerance,
                                self.conf.feasibility_tolerance)
        restoring = bool((r0 > tolerance).any())
        target = torch.zeros_like(g) if restoring else g
        v, alpha, slack = solve_direction(target, A, r0, kappa=self.conf.kappa,
                                          sweeps=self.conf.qp_sweeps)
        hard_slack = slack.detach().clone()
        elastic = (not bool(torch.isfinite(v).all()) or
                   not bool(torch.isfinite(slack).all()) or
                   bool((slack > tolerance).any()))
        if elastic:
            restoring = True
            v, alpha, slack = solve_direction(torch.zeros_like(g), A, r0,
                kappa=self.conf.kappa, elastic_penalty=self.conf.elastic_penalty,
                sweeps=self.conf.qp_sweeps)
        norm = float(torch.linalg.vector_norm(v))
        if not np.isfinite(norm):
            v = torch.zeros_like(g)
            norm = 0.0
        scale = min(self.conf.step_size, self.conf.trust_radius / max(norm, 1e-30))
        original = [p.detach().clone() for p in params]
        old_violation = r0.clamp_min(0)
        old_sum = float(old_violation.sum())
        old_max = float(old_violation.max())
        accepted = False
        rejects = 0
        attempts = 0
        post_r = r0
        actual_ascent = 0.0
        for attempt in range(self.conf.backtracks + 1):
            if norm <= 1e-20 or scale <= 0:
                break
            attempts += 1
            self._write_flat(params, original, v, scale)
            with torch.no_grad():
                trial_r, _ = self._edge_values(critic, local, indices)
                if restoring:
                    new_sum = float(trial_r.clamp_min(0).sum())
                    new_max = float(trial_r.clamp_min(0).max())
                    okay = (new_sum < old_sum - max(1e-12, old_sum * 1e-8) and
                            new_max <= old_max + self.conf.feasibility_tolerance)
                else:
                    okay = bool((trial_r <= tolerance).all())
                    if okay:
                        zx = critic.encoder(data.observations)
                        zy = critic.encoder(data.next_observations)
                        trial_info = CriticBatchInfo(critic=critic, zx=zx, zy=zy)
                        trial_j = -push(data, trial_info).loss
                        actual_ascent = float(trial_j - objective.detach())
                        okay = actual_ascent >= -1e-8
            if okay:
                accepted, post_r = True, trial_r.detach()
                break
            rejects += 1
            scale *= 0.5
        if not accepted:
            self._write_flat(params, original, v, 0.0)
            scale = 0.0
        # Cache observed violations, not QP multipliers.
        post_cpu = post_r.detach().cpu()
        ranking = torch.argsort(post_cpu, descending=True)
        bad = indices[ranking][post_cpu[ranking] > tolerance.detach().cpu()[ranking]]
        self.history_indices = torch.tensor(list(dict.fromkeys(torch.cat([
            bad[:self.conf.work_edges], self.history_indices.cpu()
        ]).tolist()))[:self.conf.work_edges], dtype=torch.long)
        info = dict(push_result.info)
        info["feasible_direction"] = {
            "work_edges": float(len(indices)), "candidate_edges": float(min(len(self.replay),
                self.conf.work_edges * self.conf.candidate_multiplier)),
            "restoration": float(restoring), "elastic": float(elastic),
            "qp_hard_slack_max": float(hard_slack.max()),
            "qp_slack_max": float(slack.max()), "qp_slack_mean": float(slack.mean()),
            "qp_multiplier_max": float(alpha.max()), "direction_norm": norm,
            "push_gradient_norm": float(torch.linalg.vector_norm(g)),
            "directional_ascent": float(torch.dot(g, v)), "actual_ascent": actual_ascent,
            "accepted": float(accepted), "backtrack_rejections": float(rejects),
            "backtrack_rejection_rate": float(rejects / max(1, rejects + int(accepted))),
            "update_rejection_rate": float(not accepted),
            "step_scale": float(scale), "work_max_violation_before": old_max,
            "work_max_violation_after": float(post_r.clamp_min(0).max()),
            "work_anchor_count": float(anchor.sum()),
            "anchor_tolerance": float(self.conf.anchor_tolerance),
            "work_anchor_max_violation": float(post_r[anchor].clamp_min(0).max())
                if bool(anchor.any()) else 0.0,
            "wall_time_sec": time.perf_counter() - started,
            "audit_edges_visited": float(len(self.replay) if audited else 0),
            "edge_residual_evaluations": float(len(indices) * (1 + attempts) +
                min(len(self.replay), self.conf.work_edges * self.conf.candidate_multiplier) +
                (len(self.replay) if audited else 0)),
            **self._audit_info,
        }
        return LossResult(loss=push_result.loss.detach(), info=info)
