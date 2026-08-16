from __future__ import annotations

import copy
from typing import Optional

import attrs
import torch

from ....data import BatchData
from ...utils import LossResult
from ..models import QuasimetricCritic
from . import CriticBatchInfo, CriticLossBase


def _zero_result(zero: torch.Tensor) -> LossResult:
    detached = zero.detach()
    return LossResult(
        loss=zero,
        info=dict(
            dist=detached,
            bound=detached,
            excess=detached,
            violation=detached,
            count=detached,
        ),
    )


def _upper_bound_result(
    *,
    dist: torch.Tensor,
    bound: torch.Tensor,
    weight: float,
) -> LossResult:
    """One-sided relative violation; behavior path costs are never regression targets."""

    excess = (dist - bound).relu()
    relative_excess = excess / (bound.detach().clamp_min(0.0) + 1.0)
    violation = relative_excess.square().mean()
    return LossResult(
        loss=float(weight) * violation,
        info=dict(
            dist=dist.mean(),
            bound=bound.mean(),
            excess=excess.mean(),
            violation=violation,
            count=torch.as_tensor(float(dist.numel()), device=dist.device),
        ),
    )


class TemporalPathConstraintLoss(CriticLossBase):
    """Use executed trajectory suffixes as non-expert state-to-state upper bounds."""

    @attrs.define(kw_only=True)
    class Conf:
        weight: float = attrs.field(default=0.0, validator=attrs.validators.ge(0.0))
        min_future_steps: int = attrs.field(default=2, validator=attrs.validators.ge(1))

        def make(self) -> "TemporalPathConstraintLoss":
            return TemporalPathConstraintLoss(
                weight=self.weight,
                min_future_steps=self.min_future_steps,
            )

    def __init__(self, *, weight: float, min_future_steps: int):
        super().__init__()
        self.weight = float(weight)
        self.min_future_steps = int(min_future_steps)

    def forward(self, data: BatchData, critic_batch_info: CriticBatchInfo) -> LossResult:
        zero = critic_batch_info.zx.sum() * 0.0
        infos = data.transition_infos or {}
        if (
            self.weight == 0.0
            or "temporal_future_cost" not in infos
            or "temporal_future_steps" not in infos
        ):
            return _zero_result(zero)

        device = critic_batch_info.zx.device
        mask = infos["temporal_future_steps"].to(device=device) >= self.min_future_steps
        abstract_edge = infos.get("abstract_goal_edge")
        if abstract_edge is not None:
            mask &= ~abstract_edge.to(device=device, dtype=torch.bool)
        if not bool(mask.any()):
            return _zero_result(zero)

        future_obs = data.future_observations.to(
            device=device,
            dtype=data.observations.dtype,
        )[mask]
        z_future = critic_batch_info.critic.encoder(future_obs)
        dist = critic_batch_info.critic.quasimetric_model(
            critic_batch_info.zx[mask],
            z_future,
        )
        bound = infos["temporal_future_cost"].to(
            device=device,
            dtype=dist.dtype,
        )[mask].reshape_as(dist).clamp_min(0.0)
        result = _upper_bound_result(dist=dist, bound=bound, weight=self.weight)
        result.info["future_steps"] = infos["temporal_future_steps"].to(
            device=device,
            dtype=dist.dtype,
        )[mask].float().mean()
        return result

    def extra_repr(self) -> str:
        return f"weight={self.weight:g}, min_future_steps={self.min_future_steps}"


class GoalReturnConstraintLoss(CriticLossBase):
    """Use only naturally successful behavior returns as task-goal upper bounds."""

    @attrs.define(kw_only=True)
    class Conf:
        weight: float = attrs.field(default=0.0, validator=attrs.validators.ge(0.0))

        def make(self) -> "GoalReturnConstraintLoss":
            return GoalReturnConstraintLoss(weight=self.weight)

    def __init__(self, *, weight: float):
        super().__init__()
        self.weight = float(weight)

    def forward(self, data: BatchData, critic_batch_info: CriticBatchInfo) -> LossResult:
        zero = critic_batch_info.zx.sum() * 0.0
        infos = data.transition_infos or {}
        if (
            self.weight == 0.0
            or "goal_return_mask" not in infos
            or "goal_return_cost" not in infos
            or "task_goal_observations" not in infos
        ):
            return _zero_result(zero)

        device = critic_batch_info.zx.device
        mask = infos["goal_return_mask"].to(device=device, dtype=torch.bool)
        abstract_edge = infos.get("abstract_goal_edge")
        if abstract_edge is not None:
            mask &= ~abstract_edge.to(device=device, dtype=torch.bool)
        if not bool(mask.any()):
            return _zero_result(zero)

        task_goals = infos["task_goal_observations"].to(
            device=device,
            dtype=data.observations.dtype,
        )[mask]
        z_goal = critic_batch_info.critic.encoder(task_goals)
        dist = critic_batch_info.critic.quasimetric_model(
            critic_batch_info.zx[mask],
            z_goal,
        )
        bound = infos["goal_return_cost"].to(
            device=device,
            dtype=dist.dtype,
        )[mask].reshape_as(dist).clamp_min(0.0)
        return _upper_bound_result(dist=dist, bound=bound, weight=self.weight)

    def extra_repr(self) -> str:
        return f"weight={self.weight:g}"


class NstepGoalConsistencyLoss(CriticLossBase):
    """Optional semi-gradient task-goal bound using an EMA target critic."""

    @attrs.define(kw_only=True)
    class Conf:
        weight: float = attrs.field(default=0.0, validator=attrs.validators.ge(0.0))
        min_future_steps: int = attrs.field(default=2, validator=attrs.validators.ge(1))
        target_tau: float = attrs.field(
            default=0.005,
            validator=attrs.validators.and_(
                attrs.validators.gt(0.0),
                attrs.validators.le(1.0),
            ),
        )

        def make(self, critic: QuasimetricCritic) -> "NstepGoalConsistencyLoss":
            return NstepGoalConsistencyLoss(
                critic=critic,
                weight=self.weight,
                min_future_steps=self.min_future_steps,
                target_tau=self.target_tau,
            )

    def __init__(
        self,
        *,
        critic: QuasimetricCritic,
        weight: float,
        min_future_steps: int,
        target_tau: float,
    ):
        super().__init__()
        self.weight = float(weight)
        self.min_future_steps = int(min_future_steps)
        self.target_tau = float(target_tau)
        self.target_critic: Optional[QuasimetricCritic] = None
        if self.weight > 0.0:
            self.target_critic = copy.deepcopy(critic)
            # Some scripted/parametrized critic tensors are non-leaf after
            # deepcopy, so requires_grad_(False) is not valid for the whole
            # module.  The target is still frozen: it is excluded from the
            # optimizer and every target forward/update runs under no_grad.
            self.target_critic.eval()

    @torch.no_grad()
    def update_target(self, critic: QuasimetricCritic) -> None:
        if self.target_critic is None:
            return
        for target, source in zip(self.target_critic.parameters(), critic.parameters()):
            target.lerp_(source.detach(), self.target_tau)
        for target, source in zip(self.target_critic.buffers(), critic.buffers()):
            target.copy_(source.detach())

    def train(self, mode: bool = True) -> "NstepGoalConsistencyLoss":
        super().train(mode)
        # The target remains deterministic/frozen when the containing loss tree
        # is toggled back to training mode.
        if self.target_critic is not None:
            self.target_critic.eval()
        return self

    def forward(self, data: BatchData, critic_batch_info: CriticBatchInfo) -> LossResult:
        zero = critic_batch_info.zx.sum() * 0.0
        infos = data.transition_infos or {}
        required = {
            "temporal_future_cost",
            "temporal_future_steps",
            "task_goal_observations",
        }
        if self.weight == 0.0 or not required.issubset(infos):
            return _zero_result(zero)
        if self.target_critic is None:  # pragma: no cover - guarded by weight
            raise RuntimeError("n-step goal target critic is unavailable")

        device = critic_batch_info.zx.device
        mask = infos["temporal_future_steps"].to(device=device) >= self.min_future_steps
        abstract_edge = infos.get("abstract_goal_edge")
        if abstract_edge is not None:
            mask &= ~abstract_edge.to(device=device, dtype=torch.bool)
        if not bool(mask.any()):
            return _zero_result(zero)

        task_goals = infos["task_goal_observations"].to(
            device=device,
            dtype=data.observations.dtype,
        )[mask]
        z_goal = critic_batch_info.critic.encoder(task_goals)
        dist = critic_batch_info.critic.quasimetric_model(
            critic_batch_info.zx[mask],
            z_goal,
        )
        with torch.no_grad():
            future_obs = data.future_observations.to(
                device=device,
                dtype=data.observations.dtype,
            )[mask]
            target_future = self.target_critic(future_obs, task_goals)
        path_cost = infos["temporal_future_cost"].to(
            device=device,
            dtype=dist.dtype,
        )[mask].reshape_as(dist).clamp_min(0.0)
        bound = path_cost + target_future.reshape_as(dist).clamp_min(0.0)
        result = _upper_bound_result(dist=dist, bound=bound, weight=self.weight)
        result.info["target_future_dist"] = target_future.mean()
        return result

    def extra_repr(self) -> str:
        return (
            f"weight={self.weight:g}, min_future_steps={self.min_future_steps}, "
            f"target_tau={self.target_tau:g}"
        )
