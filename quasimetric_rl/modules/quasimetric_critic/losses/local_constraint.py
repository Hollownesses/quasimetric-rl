from typing import *

import attrs

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....data import BatchData

from ...utils import LatentTensor, LossResult, grad_mul, softplus_inv_float

from . import CriticLossBase, CriticBatchInfo



class LocalConstraintLoss(CriticLossBase):
    @attrs.define(kw_only=True)
    class Conf:
        # config / argparse uses this to specify behavior

        epsilon: float = attrs.field(default=0.25, validator=attrs.validators.gt(0))

        # Cost per step. If environment has variable costs, this can be changed
        # to load from data, and QRL will still have guarantees.
        step_cost: float = attrs.field(default=1, validator=attrs.validators.gt(0))

        cost_source: Literal["fixed", "negative_reward"] = attrs.field(
            default="fixed",
            validator=attrs.validators.in_(("fixed", "negative_reward")),
        )

        init_lagrange_multiplier: float = attrs.field(default=0.01, validator=attrs.validators.gt(0))

        constraint_mode: Literal["unified", "full_graph_stratified"] = attrs.field(
            default="unified",
            validator=attrs.validators.in_(("unified", "full_graph_stratified")),
        )
        direct_goal_epsilon: float = attrs.field(
            default=0.25,
            validator=attrs.validators.ge(0),
        )
        terminal_goal_epsilon: float = attrs.field(
            default=0.0,
            validator=attrs.validators.ge(0),
        )

        def make(self) -> 'LocalConstraintLoss':
            return LocalConstraintLoss(
                epsilon=self.epsilon,
                step_cost=self.step_cost,
                cost_source=self.cost_source,
                init_lagrange_multiplier=self.init_lagrange_multiplier,
                constraint_mode=self.constraint_mode,
                direct_goal_epsilon=self.direct_goal_epsilon,
                terminal_goal_epsilon=self.terminal_goal_epsilon,
            )

    epsilon: float
    step_cost: float
    cost_source: Literal["fixed", "negative_reward"]
    init_lagrange_multiplier: float
    constraint_mode: Literal["unified", "full_graph_stratified"]
    direct_goal_epsilon: float
    terminal_goal_epsilon: float

    raw_lagrange_multiplier: nn.Parameter  # for the QRL constrained optimization

    def __init__(
        self,
        *,
        epsilon: float,
        step_cost: float,
        cost_source: Literal["fixed", "negative_reward"],
        init_lagrange_multiplier: float,
        constraint_mode: Literal["unified", "full_graph_stratified"] = "unified",
        direct_goal_epsilon: float = 0.25,
        terminal_goal_epsilon: float = 0.0,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.step_cost = step_cost
        if cost_source not in ("fixed", "negative_reward"):
            raise ValueError(f"Unsupported local-constraint cost_source: {cost_source}")
        self.cost_source = cost_source
        self.init_lagrange_multiplier = init_lagrange_multiplier
        if constraint_mode not in ("unified", "full_graph_stratified"):
            raise ValueError(f"Unsupported constraint_mode: {constraint_mode}")
        self.constraint_mode = constraint_mode
        self.direct_goal_epsilon = float(direct_goal_epsilon)
        self.terminal_goal_epsilon = float(terminal_goal_epsilon)
        self.raw_lagrange_multiplier = nn.Parameter(
            torch.tensor(softplus_inv_float(init_lagrange_multiplier), dtype=torch.float32))
        if constraint_mode == "full_graph_stratified":
            self.raw_direct_goal_lagrange_multiplier = nn.Parameter(
                torch.tensor(
                    softplus_inv_float(init_lagrange_multiplier),
                    dtype=torch.float32,
                )
            )
            self.raw_terminal_goal_lagrange_multiplier = nn.Parameter(
                torch.tensor(
                    softplus_inv_float(init_lagrange_multiplier),
                    dtype=torch.float32,
                )
            )
        else:
            self.register_parameter("raw_direct_goal_lagrange_multiplier", None)
            self.register_parameter("raw_terminal_goal_lagrange_multiplier", None)

    def _target_cost(self, data: BatchData, dist: torch.Tensor) -> torch.Tensor:
        if self.cost_source == "fixed":
            return torch.full_like(dist, float(self.step_cost))

        if self.cost_source == "negative_reward":
            costs = -data.rewards.to(device=dist.device, dtype=dist.dtype)
            return costs.reshape_as(dist).clamp_min(0)

        raise RuntimeError(f"Unsupported local-constraint cost_source: {self.cost_source}")

    def _constraint_family(
        self,
        *,
        dist: torch.Tensor,
        target_cost: torch.Tensor,
        mask: torch.Tensor,
        epsilon: float,
        raw_lagrange_multiplier: nn.Parameter,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not bool(mask.any()):
            raise RuntimeError("stratified constraint batch is missing a required family")
        family_dist = dist[mask]
        family_target = target_cost[mask]
        excess = (family_dist - family_target).relu()
        sq_deviation = excess.square().mean()
        violation = sq_deviation - float(epsilon) ** 2
        lagrange_mult = grad_mul(
            F.softplus(raw_lagrange_multiplier),
            -1,
        )
        return violation * lagrange_mult, {
            "count": mask.sum().to(dtype=dist.dtype),
            "dist": family_dist.mean(),
            "max_dist": family_dist.max(),
            "target_cost_mean": family_target.mean(),
            "target_cost_min": family_target.min(),
            "target_cost_max": family_target.max(),
            "sq_deviation": sq_deviation,
            "violation": violation,
            "violation_fraction": (excess > 0).to(dtype=dist.dtype).mean(),
            "mean_excess": excess.mean(),
            "max_excess": excess.max(),
            "lagrange_mult": lagrange_mult,
            "epsilon": torch.as_tensor(epsilon, device=dist.device, dtype=dist.dtype),
        }

    def forward(self, data: BatchData, critic_batch_info: CriticBatchInfo) -> LossResult:

        dist = critic_batch_info.critic.quasimetric_model(critic_batch_info.zx, critic_batch_info.zy)
        target_cost = self._target_cost(data, dist)

        if self.constraint_mode == "full_graph_stratified":
            infos = data.transition_infos
            direct_goal = infos.get("full_graph_direct_goal_edge") if infos else None
            terminal_goal = infos.get("abstract_goal_edge") if infos else None
            if direct_goal is None or terminal_goal is None:
                raise RuntimeError(
                    "full_graph_stratified constraints require direct/terminal edge metadata"
                )
            terminal_mask = terminal_goal.to(device=dist.device, dtype=torch.bool)
            direct_mask = (
                direct_goal.to(device=dist.device, dtype=torch.bool)
                & ~terminal_mask
            )
            ordinary_mask = ~direct_mask & ~terminal_mask
            ordinary_loss, ordinary_info = self._constraint_family(
                dist=dist,
                target_cost=target_cost,
                mask=ordinary_mask,
                epsilon=self.epsilon,
                raw_lagrange_multiplier=self.raw_lagrange_multiplier,
            )
            direct_loss, direct_info = self._constraint_family(
                dist=dist,
                target_cost=target_cost,
                mask=direct_mask,
                epsilon=self.direct_goal_epsilon,
                raw_lagrange_multiplier=self.raw_direct_goal_lagrange_multiplier,
            )
            terminal_loss, terminal_info = self._constraint_family(
                dist=dist,
                target_cost=target_cost,
                mask=terminal_mask,
                epsilon=self.terminal_goal_epsilon,
                raw_lagrange_multiplier=self.raw_terminal_goal_lagrange_multiplier,
            )
            overall_excess = (dist - target_cost).relu()
            population_counts = infos.get(
                "full_graph_constraint_population_counts"
            )
            if population_counts is None:
                raise RuntimeError(
                    "stratified constraint batch is missing family population counts"
                )
            population_counts = population_counts.to(
                device=dist.device,
                dtype=dist.dtype,
            )
            family_fractions = torch.stack(
                [
                    ordinary_info["violation_fraction"],
                    direct_info["violation_fraction"],
                    terminal_info["violation_fraction"],
                ]
            )
            graph_weighted_violation_fraction = (
                family_fractions * population_counts
            ).sum() / population_counts.sum()
            return LossResult(
                loss=ordinary_loss + direct_loss + terminal_loss,
                info=dict(
                    dist=dist.mean(),
                    sq_deviation=overall_excess.square().mean(),
                    violation_fraction=(overall_excess > 0).to(
                        dtype=dist.dtype
                    ).mean(),
                    graph_weighted_violation_fraction=(
                        graph_weighted_violation_fraction
                    ),
                    mean_excess=overall_excess.mean(),
                    max_excess=overall_excess.max(),
                    target_cost_mean=target_cost.mean(),
                    target_cost_min=target_cost.min(),
                    target_cost_max=target_cost.max(),
                    ordinary=ordinary_info,
                    direct_goal=direct_info,
                    terminal_goal=terminal_info,
                ),
            )

        lagrange_mult = F.softplus(self.raw_lagrange_multiplier)  # make positive
        # lagrange multiplier is minimax training, so grad_mul -1
        lagrange_mult = grad_mul(lagrange_mult, -1)

        sq_deviation = (dist - target_cost).relu().square().mean()
        violation = (sq_deviation - self.epsilon ** 2)
        loss = violation * lagrange_mult

        return LossResult(
            loss=loss,
            info=dict(
                dist=dist.mean(),
                sq_deviation=sq_deviation,
                violation=violation,
                lagrange_mult=lagrange_mult,
                target_cost_mean=target_cost.mean(),
                target_cost_min=target_cost.min(),
                target_cost_max=target_cost.max(),
                violation_fraction=((dist - target_cost) > 0).to(
                    dtype=dist.dtype
                ).mean(),
                mean_excess=(dist - target_cost).relu().mean(),
                max_excess=(dist - target_cost).relu().max(),
            ),
        )

    def extra_repr(self) -> str:
        return (
            f"epsilon={self.epsilon:g}, step_cost={self.step_cost:g}, "
            f"cost_source={self.cost_source}, constraint_mode={self.constraint_mode}"
        )
