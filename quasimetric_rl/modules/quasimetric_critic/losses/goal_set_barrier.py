"""Boundary-aware soft replay constraint for a quotient goal-set critic.

This is an unconstrained stochastic objective. Full replay violations are
audited, but no hard feasibility claim is made for its AdamW updates.
"""

import time

import attrs
import torch
import torch.nn.functional as F

from ....data import BatchData
from ...utils import LossResult
from . import CriticBatchInfo
from .feasible_direction import FeasibleDirectionConf, FeasibleDirectionUpdater
from .global_push import GlobalPushLoss
from .local_constraint import LocalConstraintLoss


@attrs.define(kw_only=True)
class GoalSetBarrierConf:
    weight: float = 500.0
    margin: float = 0.01
    temperature: float = 0.01
    topk: int = 8
    work_edges: int = 64
    candidate_multiplier: int = 4
    audit_interval: int = 25
    audit_chunk: int = 1024
    feasibility_tolerance: float = 1e-5
    anchor_tolerance: float = 1e-6

    def __attrs_post_init__(self):
        if self.weight <= 0 or self.margin < 0 or self.temperature <= 0:
            raise ValueError("barrier weight and temperature must be positive; margin nonnegative")
        if min(self.topk, self.work_edges, self.candidate_multiplier,
               self.audit_interval, self.audit_chunk) <= 0:
            raise ValueError("barrier counts must be positive")


class GoalSetBarrierUpdater(FeasibleDirectionUpdater):
    def __init__(self, conf: GoalSetBarrierConf):
        super().__init__(FeasibleDirectionConf(
            work_edges=conf.work_edges,
            candidate_multiplier=conf.candidate_multiplier,
            audit_interval=conf.audit_interval,
            audit_chunk=conf.audit_chunk,
            feasibility_tolerance=conf.feasibility_tolerance,
            anchor_tolerance=conf.anchor_tolerance,
        ))
        self.barrier_conf = conf

    @torch.no_grad()
    def _workset(self, critic: torch.nn.Module,
                 local: LocalConstraintLoss) -> torch.Tensor:
        assert self.replay is not None
        n, m = len(self.replay), min(self.conf.work_edges, len(self.replay))
        candidates = torch.randint(n, (min(n, m * self.conf.candidate_multiplier),))
        candidate_r, _ = self._edge_values(critic, local, candidates)
        candidate_r = candidate_r.detach().cpu()
        eighth = max(1, m // 8)
        parts = [
            self.audit_high_indices.cpu()[:m // 2],
            self.history_indices.cpu()[:eighth],
            self.audit_near_indices.cpu()[:eighth],
            candidates[:eighth],
            candidates[torch.topk(candidate_r, min(eighth, len(candidates))).indices],
        ]
        ordered = list(dict.fromkeys(torch.cat(parts).tolist()))[:m]
        if len(ordered) < m:
            ordered = list(dict.fromkeys(ordered + torch.randperm(n).tolist()))[:m]
        return torch.tensor(ordered, dtype=torch.long)

    def step(self, data: BatchData, batch_info: CriticBatchInfo,
             push: GlobalPushLoss, local: LocalConstraintLoss) -> LossResult:
        if self.replay is None:
            raise RuntimeError("goal-set barrier requires set_replay(dataset)")
        started = time.perf_counter()
        self.steps += 1
        audited = (not self._audited_this_process or
                   int(self.steps) % self.conf.audit_interval == 0)
        if audited:
            self.audit(batch_info.critic, local)
        indices = self._workset(batch_info.critic, local)
        residual, cost = self._edge_values(batch_info.critic, local, indices)
        residual_cpu = residual.detach().cpu()
        bad_order = torch.argsort(residual_cpu, descending=True)
        violating = indices[bad_order][
            residual_cpu[bad_order] > self.conf.feasibility_tolerance]
        self.history_indices = torch.tensor(list(dict.fromkeys(torch.cat([
            violating[:self.conf.work_edges], self.history_indices.cpu()
        ]).tolist()))[:self.conf.work_edges], dtype=torch.long)
        eligible = (cost > 0) | (residual > self.conf.anchor_tolerance)
        active_residual = residual[eligible]
        if active_residual.numel():
            penalty = self.barrier_conf.temperature * F.softplus(
                (active_residual + self.barrier_conf.margin)
                / self.barrier_conf.temperature)
            k = min(self.barrier_conf.topk, penalty.numel())
            top_penalty = torch.topk(penalty, k).values.mean()
        else:
            top_penalty = residual.sum() * 0.0
        push_result = push(data, batch_info)
        loss = push_result.loss + self.barrier_conf.weight * top_penalty
        info = dict(push_result.info)
        info["goal_set_barrier"] = {
            "work_edges": float(len(indices)),
            "work_max_violation": float(residual.detach().clamp_min(0).max()),
            "work_real_violation_fraction": float(
                (active_residual.detach() > self.conf.feasibility_tolerance)
                .float().mean()) if active_residual.numel() else 0.0,
            "top_penalty": float(top_penalty.detach()),
            "weighted_penalty": float((self.barrier_conf.weight * top_penalty).detach()),
            "wall_time_sec": time.perf_counter() - started,
            "audit_edges_visited": float(len(self.replay) if audited else 0),
            **self._audit_info,
        }
        return LossResult(loss=loss, info=info)
