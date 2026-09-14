from __future__ import annotations

import copy
from typing import Optional

import attrs
import torch
import torch.nn.functional as F

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
        result.info["future_steps"] = infos["temporal_future_steps"].to(
            device=device,
            dtype=dist.dtype,
        )[mask].float().mean()
        return result

    def extra_repr(self) -> str:
        return (
            f"weight={self.weight:g}, min_future_steps={self.min_future_steps}, "
            f"target_tau={self.target_tau:g}"
        )


class MQEInspiredWaypointConsistencyLoss(CriticLossBase):
    """Two-sided multistep waypoint regression with an EMA target critic.

    Physical samples regress

        d(s_t, g) = C[t:t+k] + stop_grad(d_target(s_{t+k}, g)).

    Terminal-anchor samples use a naturally successful trajectory, set the
    waypoint to its physical terminal state, and use the task-defined exact
    edge d(s_terminal, G)=0.  Their target is therefore the observed remaining
    cost, which directly prevents abstract-goal values from becoming detached
    from the physical topology.
    """

    @attrs.define(kw_only=True)
    class Conf:
        weight: float = attrs.field(default=0.0, validator=attrs.validators.ge(0.0))
        target_tau: float = attrs.field(
            default=0.005,
            validator=attrs.validators.and_(
                attrs.validators.gt(0.0),
                attrs.validators.le(1.0),
            ),
        )
        huber_delta: float = attrs.field(
            default=1.0,
            validator=attrs.validators.gt(0.0),
        )

        def make(
            self,
            critic: QuasimetricCritic,
        ) -> "MQEInspiredWaypointConsistencyLoss":
            return MQEInspiredWaypointConsistencyLoss(
                critic=critic,
                weight=self.weight,
                target_tau=self.target_tau,
                huber_delta=self.huber_delta,
            )

    def __init__(
        self,
        *,
        critic: QuasimetricCritic,
        weight: float,
        target_tau: float,
        huber_delta: float,
    ) -> None:
        super().__init__()
        self.weight = float(weight)
        self.target_tau = float(target_tau)
        self.huber_delta = float(huber_delta)
        self.target_critic: Optional[QuasimetricCritic] = None
        if self.weight > 0.0:
            self.target_critic = copy.deepcopy(critic)
            self.target_critic.eval()

    @staticmethod
    def _empty_info(zero: torch.Tensor) -> dict[str, torch.Tensor]:
        value = zero.detach()
        return {
            "huber": value,
            "dist": value,
            "target": value,
            "target_waypoint_dist": value,
            "residual": value,
            "abs_residual": value,
            "overestimate_fraction": value,
            "underestimate_fraction": value,
            "valid_count": value,
            "physical_goal_count": value,
            "terminal_anchor_count": value,
            "terminal_anchor_fraction": value,
            "k1_ratio": value,
            "forced_k1_ratio": value,
            "waypoint_steps_mean": value,
            "waypoint_steps_p25": value,
            "waypoint_steps_p50": value,
            "waypoint_steps_p75": value,
            "waypoint_steps_p95": value,
            "goal_steps_mean": value,
            "waypoint_cost_mean": value,
            "physical_huber": value,
            "terminal_anchor_huber": value,
        }

    @torch.no_grad()
    def update_target(self, critic: QuasimetricCritic) -> None:
        if self.target_critic is None:
            return
        for target, source in zip(self.target_critic.parameters(), critic.parameters()):
            target.lerp_(source.detach(), self.target_tau)
        for target, source in zip(self.target_critic.buffers(), critic.buffers()):
            target.copy_(source.detach())

    def train(self, mode: bool = True) -> "MQEInspiredWaypointConsistencyLoss":
        super().train(mode)
        if self.target_critic is not None:
            self.target_critic.eval()
        return self

    def forward(
        self,
        data: BatchData,
        critic_batch_info: CriticBatchInfo,
    ) -> LossResult:
        zero = critic_batch_info.zx.sum() * 0.0
        infos = data.transition_infos or {}
        required = {
            "mqe_waypoint_valid",
            "mqe_waypoint_source_observations",
            "mqe_waypoint_observations",
            "mqe_waypoint_goal_observations",
            "mqe_waypoint_cost",
            "mqe_waypoint_steps",
            "mqe_waypoint_goal_steps",
            "mqe_waypoint_forced_one_step",
            "mqe_waypoint_physical_goal",
            "mqe_waypoint_terminal_anchor",
        }
        if self.weight == 0.0 or not required.issubset(infos):
            return LossResult(loss=zero, info=self._empty_info(zero))
        if self.target_critic is None:  # pragma: no cover - guarded by weight
            raise RuntimeError("MQE-inspired waypoint target critic is unavailable")

        device = critic_batch_info.zx.device
        mask = infos["mqe_waypoint_valid"].to(device=device, dtype=torch.bool)
        if not bool(mask.any()):
            return LossResult(loss=zero, info=self._empty_info(zero))

        dtype = data.observations.dtype
        source_observations = infos["mqe_waypoint_source_observations"].to(
            device=device,
            dtype=dtype,
        )[mask]
        goal_observations = infos["mqe_waypoint_goal_observations"].to(
            device=device,
            dtype=dtype,
        )[mask]
        z_source = critic_batch_info.critic.encoder(source_observations)
        z_goal = critic_batch_info.critic.encoder(goal_observations)
        dist = critic_batch_info.critic.quasimetric_model(z_source, z_goal)

        terminal_anchor = infos["mqe_waypoint_terminal_anchor"].to(
            device=device,
            dtype=torch.bool,
        )[mask].reshape(-1)
        with torch.no_grad():
            waypoint_observations = infos["mqe_waypoint_observations"].to(
                device=device,
                dtype=dtype,
            )[mask]
            target_waypoint_dist = self.target_critic(
                waypoint_observations,
                goal_observations,
            ).reshape_as(dist)
            # The terminal-to-G edge is exactly zero by the goal-set task
            # definition.  Do not let a drifting bootstrap weaken this anchor.
            target_waypoint_dist = torch.where(
                terminal_anchor.reshape_as(dist),
                torch.zeros_like(target_waypoint_dist),
                target_waypoint_dist,
            )
            waypoint_cost = infos["mqe_waypoint_cost"].to(
                device=device,
                dtype=dist.dtype,
            )[mask].reshape_as(dist)
            target = waypoint_cost + target_waypoint_dist

        per_sample_huber = F.huber_loss(
            dist,
            target,
            reduction="none",
            delta=self.huber_delta,
        )
        huber = per_sample_huber.mean()
        weighted_loss = self.weight * huber
        residual = dist - target
        steps = infos["mqe_waypoint_steps"].to(
            device=device,
            dtype=dist.dtype,
        )[mask].reshape(-1)
        goal_steps = infos["mqe_waypoint_goal_steps"].to(
            device=device,
            dtype=dist.dtype,
        )[mask].reshape(-1)
        forced = infos["mqe_waypoint_forced_one_step"].to(
            device=device,
            dtype=torch.bool,
        )[mask].reshape(-1)
        physical_goal = infos["mqe_waypoint_physical_goal"].to(
            device=device,
            dtype=torch.bool,
        )[mask].reshape(-1)
        flat_huber = per_sample_huber.reshape(-1)

        def family_mean(family_mask: torch.Tensor) -> torch.Tensor:
            if bool(family_mask.any()):
                return flat_huber[family_mask].mean()
            return zero.detach()

        count = mask.sum().to(dtype=dist.dtype)
        return LossResult(
            loss=weighted_loss,
            info={
                "huber": huber,
                "dist": dist.mean(),
                "target": target.mean(),
                "target_waypoint_dist": target_waypoint_dist.mean(),
                "residual": residual.mean(),
                "abs_residual": residual.abs().mean(),
                "overestimate_fraction": (residual > 0).to(dist.dtype).mean(),
                "underestimate_fraction": (residual < 0).to(dist.dtype).mean(),
                "valid_count": count,
                "physical_goal_count": physical_goal.sum().to(dist.dtype),
                "terminal_anchor_count": terminal_anchor.sum().to(dist.dtype),
                "terminal_anchor_fraction": terminal_anchor.to(dist.dtype).mean(),
                "k1_ratio": (steps == 1).to(dist.dtype).mean(),
                "forced_k1_ratio": forced.to(dist.dtype).mean(),
                "waypoint_steps_mean": steps.mean(),
                "waypoint_steps_p25": torch.quantile(steps, 0.25),
                "waypoint_steps_p50": torch.quantile(steps, 0.50),
                "waypoint_steps_p75": torch.quantile(steps, 0.75),
                "waypoint_steps_p95": torch.quantile(steps, 0.95),
                "goal_steps_mean": goal_steps.mean(),
                "waypoint_cost_mean": waypoint_cost.mean(),
                "physical_huber": family_mean(physical_goal),
                "terminal_anchor_huber": family_mean(terminal_anchor),
            },
        )

    def extra_repr(self) -> str:
        return (
            f"weight={self.weight:g}, target_tau={self.target_tau:g}, "
            f"huber_delta={self.huber_delta:g}"
        )
