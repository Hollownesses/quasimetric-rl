from typing import *
import math

import attrs

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....data import BatchData

from ...utils import MLP, LatentTensor, LossResult, grad_mul, softplus_inv_float

from . import CriticLossBase, CriticBatchInfo



class LocalConstraintLoss(CriticLossBase):
    MODES = ("legacy_squared_hinge", "kkt_functional")

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

        mode: Literal["legacy_squared_hinge", "kkt_functional"] = attrs.field(
            default="legacy_squared_hinge",
            validator=attrs.validators.in_(
                ("legacy_squared_hinge", "kkt_functional")
            ),
        )
        augmented_lagrangian_rho: float = attrs.field(
            default=1.0,
            validator=attrs.validators.ge(0),
        )
        dual_hidden_sizes: Tuple[int, ...] = attrs.field(
            factory=lambda: (128, 128),
            converter=tuple,
        )
        dual_max: float = attrs.field(default=100000.0, validator=attrs.validators.gt(0))
        dual_steps: int = attrs.field(default=1, validator=attrs.validators.gt(0))
        dual_start_violation_fraction: float = attrs.field(
            default=0.05,
            validator=attrs.validators.and_(
                attrs.validators.ge(0),
                attrs.validators.le(1),
            ),
        )
        dual_projected_step_size: float = attrs.field(
            default=0.1,
            validator=attrs.validators.gt(0),
        )
        dual_huber_delta: float = attrs.field(
            default=1.0,
            validator=attrs.validators.gt(0),
        )
        dual_feature_scale: float = attrs.field(
            default=5.0,
            validator=attrs.validators.gt(0),
        )
        dual_raw_min: float = attrs.field(
            default=-10.0,
            validator=attrs.validators.lt(0),
        )

        def make(
            self,
            critic: Optional['QuasimetricCritic'] = None,
        ) -> 'LocalConstraintLoss':
            observation_size = None
            if critic is not None:
                observation_size = int(math.prod(critic.encoder.input_shape))
            return LocalConstraintLoss(
                epsilon=self.epsilon,
                step_cost=self.step_cost,
                cost_source=self.cost_source,
                init_lagrange_multiplier=self.init_lagrange_multiplier,
                mode=self.mode,
                augmented_lagrangian_rho=self.augmented_lagrangian_rho,
                dual_hidden_sizes=self.dual_hidden_sizes,
                dual_max=self.dual_max,
                dual_steps=self.dual_steps,
                dual_start_violation_fraction=self.dual_start_violation_fraction,
                dual_projected_step_size=self.dual_projected_step_size,
                dual_huber_delta=self.dual_huber_delta,
                dual_feature_scale=self.dual_feature_scale,
                dual_raw_min=self.dual_raw_min,
                observation_size=observation_size,
            )

    epsilon: float
    step_cost: float
    cost_source: Literal["fixed", "negative_reward"]
    init_lagrange_multiplier: float
    mode: Literal["legacy_squared_hinge", "kkt_functional"]
    augmented_lagrangian_rho: float
    dual_hidden_sizes: Tuple[int, ...]
    dual_max: float
    dual_steps: int
    dual_start_violation_fraction: float
    dual_projected_step_size: float
    dual_huber_delta: float
    dual_feature_scale: float
    dual_raw_min: float

    raw_lagrange_multiplier: Optional[nn.Parameter]
    dual_network: Optional[MLP]

    def __init__(
        self,
        *,
        epsilon: float,
        step_cost: float,
        cost_source: Literal["fixed", "negative_reward"],
        init_lagrange_multiplier: float,
        mode: Literal["legacy_squared_hinge", "kkt_functional"] = "legacy_squared_hinge",
        augmented_lagrangian_rho: float = 1.0,
        dual_hidden_sizes: Tuple[int, ...] = (128, 128),
        dual_max: float = 100000.0,
        dual_steps: int = 1,
        dual_start_violation_fraction: float = 0.05,
        dual_projected_step_size: float = 0.1,
        dual_huber_delta: float = 1.0,
        dual_feature_scale: float = 5.0,
        dual_raw_min: float = -10.0,
        observation_size: Optional[int] = None,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.step_cost = step_cost
        if cost_source not in ("fixed", "negative_reward"):
            raise ValueError(f"Unsupported local-constraint cost_source: {cost_source}")
        self.cost_source = cost_source
        self.init_lagrange_multiplier = init_lagrange_multiplier
        if mode not in self.MODES:
            raise ValueError(f"Unsupported local-constraint mode: {mode}")
        if float(augmented_lagrangian_rho) < 0.0:
            raise ValueError("augmented_lagrangian_rho must be non-negative")
        if int(dual_steps) <= 0:
            raise ValueError("dual_steps must be positive")
        if not 0.0 <= float(dual_start_violation_fraction) <= 1.0:
            raise ValueError("dual_start_violation_fraction must be in [0, 1]")
        if float(dual_projected_step_size) <= 0.0:
            raise ValueError("dual_projected_step_size must be positive")
        if float(dual_huber_delta) <= 0.0:
            raise ValueError("dual_huber_delta must be positive")
        if float(dual_feature_scale) <= 0.0:
            raise ValueError("dual_feature_scale must be positive")
        if float(dual_raw_min) >= 0.0:
            raise ValueError("dual_raw_min must be negative")
        if float(dual_max) <= float(init_lagrange_multiplier):
            raise ValueError(
                "dual_max must be greater than init_lagrange_multiplier"
            )
        self.mode = mode
        self.augmented_lagrangian_rho = float(augmented_lagrangian_rho)
        self.dual_hidden_sizes = tuple(int(size) for size in dual_hidden_sizes)
        self.dual_max = float(dual_max)
        self.dual_steps = int(dual_steps)
        self.dual_start_violation_fraction = float(
            dual_start_violation_fraction
        )
        self.dual_projected_step_size = float(dual_projected_step_size)
        self.dual_huber_delta = float(dual_huber_delta)
        self.dual_feature_scale = float(dual_feature_scale)
        self.dual_raw_min = float(dual_raw_min)

        if self.mode == "legacy_squared_hinge":
            self.raw_lagrange_multiplier = nn.Parameter(
                torch.tensor(
                    softplus_inv_float(init_lagrange_multiplier),
                    dtype=torch.float32,
                )
            )
            self.dual_network = None
            self.register_buffer("dual_raw_offset", None)
            self.register_buffer("dual_updates_started", None)
        else:
            if observation_size is None or int(observation_size) <= 0:
                raise ValueError(
                    "kkt_functional mode requires a positive observation_size"
                )
            self.register_parameter("raw_lagrange_multiplier", None)
            self.dual_network = MLP(
                2 * int(observation_size) + 1,
                1,
                hidden_sizes=self.dual_hidden_sizes,
                zero_init_last_fc=True,
            )
            initial_raw = torch.tensor(
                softplus_inv_float(float(init_lagrange_multiplier)),
                dtype=torch.float32,
            )
            self.register_buffer("dual_raw_offset", initial_raw)
            self.register_buffer(
                "dual_updates_started",
                torch.tensor(False, dtype=torch.bool),
            )

    def _target_cost(self, data: BatchData, dist: torch.Tensor) -> torch.Tensor:
        if self.cost_source == "fixed":
            return torch.full_like(dist, float(self.step_cost))

        if self.cost_source == "negative_reward":
            costs = -data.rewards.to(device=dist.device, dtype=dist.dtype)
            return costs.reshape_as(dist).clamp_min(0)

        raise RuntimeError(f"Unsupported local-constraint cost_source: {self.cost_source}")

    @property
    def uses_separate_dual_updates(self) -> bool:
        return self.mode == "kkt_functional"

    def _functional_lagrange_multiplier(
        self,
        observations: torch.Tensor,
        next_observations: torch.Tensor,
        target_cost: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.dual_network is None or self.dual_raw_offset is None:
            raise RuntimeError("functional dual requested in legacy mode")
        # Use fixed raw edge observations rather than the moving critic latent
        # coordinates.  This keeps the dual field independent of theta.
        features = torch.cat(
            [
                observations.detach().flatten(start_dim=1),
                next_observations.detach().flatten(start_dim=1),
                target_cost.detach().clamp_min(0).unsqueeze(-1),
            ],
            dim=-1,
        )
        # Raw environment coordinates span very different scales.  A fixed
        # signed-log transform keeps the dual field edge-conditioned without
        # making its output depend on the current minibatch statistics.
        features = (
            torch.sign(features)
            * torch.log1p(features.abs())
            / self.dual_feature_scale
        )
        raw_unbounded = self.dual_network(features).squeeze(-1)
        raw_unbounded = raw_unbounded + self.dual_raw_offset.to(
            device=raw_unbounded.device,
            dtype=raw_unbounded.dtype,
        )
        # Forward values cannot enter the unrecoverable lower softplus tail.
        # The straight-through form preserves a recovery gradient if the
        # unconstrained network output temporarily crosses the trust region.
        raw_effective = raw_unbounded + (
            raw_unbounded.clamp_min(self.dual_raw_min) - raw_unbounded
        ).detach()
        lagrange_mult = F.softplus(raw_effective).clamp_max(self.dual_max)
        return lagrange_mult, raw_unbounded, raw_effective

    def _functional_terms(
        self,
        data: BatchData,
        critic_batch_info: CriticBatchInfo,
        *,
        detach_constraint: bool,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if detach_constraint:
            with torch.no_grad():
                dist = critic_batch_info.critic.quasimetric_model(
                    critic_batch_info.zx.detach(),
                    critic_batch_info.zy.detach(),
                )
                target_cost = self._target_cost(data, dist)
                residual = dist - target_cost
        else:
            dist = critic_batch_info.critic.quasimetric_model(
                critic_batch_info.zx,
                critic_batch_info.zy,
            )
            target_cost = self._target_cost(data, dist)
            residual = dist - target_cost
        lagrange_mult, raw_unbounded, raw_effective = (
            self._functional_lagrange_multiplier(
                data.observations,
                data.next_observations,
                target_cost,
            )
        )
        return (
            residual,
            target_cost,
            lagrange_mult,
            raw_unbounded,
            raw_effective,
        )

    def dual_loss(
        self,
        data: BatchData,
        critic_batch_info: CriticBatchInfo,
    ) -> LossResult:
        """Fit one projected per-edge dual-ascent step in output space."""

        if not self.uses_separate_dual_updates:
            raise RuntimeError("separate dual loss is only defined in kkt_functional mode")
        (
            residual,
            _target_cost,
            lagrange_mult,
            raw_unbounded,
            raw_effective,
        ) = self._functional_terms(
            data,
            critic_batch_info,
            detach_constraint=True,
        )
        violation_mask = residual > 0
        violation_fraction = violation_mask.to(residual.dtype).mean()
        if self.dual_updates_started is None:
            raise RuntimeError("functional dual start state is unavailable")
        with torch.no_grad():
            should_start = (
                violation_fraction >= self.dual_start_violation_fraction
            )
            self.dual_updates_started.logical_or_(should_start)

        update_enabled = self.dual_updates_started
        with torch.no_grad():
            target_lagrange_mult = (
                lagrange_mult.detach()
                + self.dual_projected_step_size * residual
            ).clamp(min=0.0, max=self.dual_max)
            lagrange_floor = F.softplus(
                torch.as_tensor(
                    self.dual_raw_min,
                    device=residual.device,
                    dtype=residual.dtype,
                )
            )
            raw_target_input = target_lagrange_mult.clamp_min(lagrange_floor)
            # Stable inverse softplus: x + log(1 - exp(-x)).
            target_raw = raw_target_input + torch.log(
                -torch.expm1(-raw_target_input)
            )
            projected_delta = target_lagrange_mult - lagrange_mult.detach()

        fitted_loss = F.huber_loss(
            raw_effective,
            target_raw,
            reduction="mean",
            delta=self.dual_huber_delta,
        )
        loss = fitted_loss * update_enabled.to(residual.dtype)
        return LossResult(
            loss=loss,
            info=dict(
                fitted_loss=fitted_loss,
                update_enabled=update_enabled.to(residual.dtype),
                updates_started=self.dual_updates_started.to(residual.dtype),
                residual_mean=residual.mean(),
                residual_max=residual.max(),
                violation_fraction=violation_fraction,
                target_lagrange_mult_mean=target_lagrange_mult.mean(),
                target_lagrange_mult_min=target_lagrange_mult.min(),
                target_lagrange_mult_max=target_lagrange_mult.max(),
                projected_delta_mean=projected_delta.mean(),
                projected_delta_abs_mean=projected_delta.abs().mean(),
                projected_increase_fraction=(projected_delta > 0).to(
                    residual.dtype
                ).mean(),
                projected_decrease_fraction=(projected_delta < 0).to(
                    residual.dtype
                ).mean(),
                projected_zero_fraction=(target_lagrange_mult == 0).to(
                    residual.dtype
                ).mean(),
                lagrange_mult_mean=lagrange_mult.mean(),
                lagrange_mult_min=lagrange_mult.min(),
                lagrange_mult_max=lagrange_mult.max(),
                raw_dual_mean=raw_unbounded.mean(),
                raw_dual_min=raw_unbounded.min(),
                raw_dual_max=raw_unbounded.max(),
                raw_dual_effective_min=raw_effective.min(),
                dual_lower_saturation_fraction=(
                    raw_unbounded < self.dual_raw_min
                ).to(residual.dtype).mean(),
                dual_saturation_fraction=(lagrange_mult >= self.dual_max).to(
                    residual.dtype
                ).mean(),
            ),
        )

    def forward(self, data: BatchData, critic_batch_info: CriticBatchInfo) -> LossResult:

        if self.uses_separate_dual_updates:
            (
                residual,
                target_cost,
                lagrange_mult,
                raw_unbounded,
                raw_effective,
            ) = self._functional_terms(
                data,
                critic_batch_info,
                detach_constraint=False,
            )
            positive_residual = residual.relu()
            linear_lagrangian = (lagrange_mult.detach() * residual).mean()
            augmented_penalty = (
                0.5
                * self.augmented_lagrangian_rho
                * positive_residual.square().mean()
            )
            loss = linear_lagrangian + augmented_penalty
            dual_objective = (lagrange_mult * residual.detach()).mean()
            complementarity_abs = (
                lagrange_mult.detach() * residual.detach().abs()
            ).mean()
            return LossResult(
                loss=loss,
                info=dict(
                    dist=(residual + target_cost).mean(),
                    target_cost_mean=target_cost.mean(),
                    target_cost_min=target_cost.min(),
                    target_cost_max=target_cost.max(),
                    residual_mean=residual.mean(),
                    residual_min=residual.min(),
                    residual_max=residual.max(),
                    violation_fraction=(residual > 0).to(residual.dtype).mean(),
                    near_active_fraction=(residual.abs() <= 1e-3).to(
                        residual.dtype
                    ).mean(),
                    sq_deviation=positive_residual.square().mean(),
                    linear_lagrangian=linear_lagrangian,
                    augmented_penalty=augmented_penalty,
                    dual_objective=dual_objective,
                    complementarity_abs=complementarity_abs,
                    lagrange_mult=lagrange_mult.mean(),
                    lagrange_mult_min=lagrange_mult.min(),
                    lagrange_mult_max=lagrange_mult.max(),
                    raw_dual_mean=raw_unbounded.mean(),
                    raw_dual_min=raw_unbounded.min(),
                    raw_dual_max=raw_unbounded.max(),
                    raw_dual_effective_min=raw_effective.min(),
                    dual_lower_saturation_fraction=(
                        raw_unbounded < self.dual_raw_min
                    ).to(residual.dtype).mean(),
                    dual_saturation_fraction=(lagrange_mult >= self.dual_max).to(
                        residual.dtype
                    ).mean(),
                ),
            )

        dist = critic_batch_info.critic.quasimetric_model(critic_batch_info.zx, critic_batch_info.zy)
        target_cost = self._target_cost(data, dist)

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
            ),
        )

    def extra_repr(self) -> str:
        if self.mode == "legacy_squared_hinge":
            return (
                f"mode={self.mode}, epsilon={self.epsilon:g}, "
                f"step_cost={self.step_cost:g}, cost_source={self.cost_source}"
            )
        return (
            f"mode={self.mode}, rho={self.augmented_lagrangian_rho:g}, "
            f"dual_steps={self.dual_steps}, dual_max={self.dual_max:g}, "
            f"projected_step_size={self.dual_projected_step_size:g}, "
            f"dual_huber_delta={self.dual_huber_delta:g}, "
            f"start_violation_fraction={self.dual_start_violation_fraction:g}, "
            f"raw_min={self.dual_raw_min:g}, "
            f"step_cost={self.step_cost:g}, cost_source={self.cost_source}"
        )
