from typing import *

import attrs

import torch

from ....data import BatchData

from ...utils import LatentTensor, LossResult
from ..models import QuasimetricCritic

from . import CriticLossBase, CriticBatchInfo



class GlobalPushLoss(CriticLossBase):
    @attrs.define(kw_only=True)
    class Conf:
        # config / argparse uses this to specify behavior

        # Retained for CLI/checkpoint compatibility with earlier experiments.
        # The LP-faithful linear objective does not use either softplus field.
        softplus_beta: float = attrs.field(default=0.1, validator=attrs.validators.gt(0))

        softplus_offset: float = attrs.field(default=15, validator=attrs.validators.ge(0))
        abstract_goal_ratio: float = attrs.field(default=0.8, validator=attrs.validators.ge(0))
        state_goal_ratio: float = attrs.field(default=0.2, validator=attrs.validators.ge(0))

        def make(self) -> 'GlobalPushLoss':
            return GlobalPushLoss(
                softplus_beta=self.softplus_beta,
                softplus_offset=self.softplus_offset,
                abstract_goal_ratio=self.abstract_goal_ratio,
                state_goal_ratio=self.state_goal_ratio,
            )

    softplus_beta: float
    softplus_offset: float
    abstract_goal_ratio: float
    state_goal_ratio: float

    def __init__(
        self,
        *,
        softplus_beta: float,
        softplus_offset: float,
        abstract_goal_ratio: float,
        state_goal_ratio: float,
    ):
        super().__init__()
        self.softplus_beta = softplus_beta
        self.softplus_offset = softplus_offset
        self.abstract_goal_ratio = abstract_goal_ratio
        self.state_goal_ratio = state_goal_ratio

    def _push_loss(self, dists: torch.Tensor) -> torch.Tensor:
        # Minimizing -E[d] is exactly the unconstrained form of the theoretical
        # Global Push objective max E[d].  In particular, this has constant
        # gradient and cannot saturate at an arbitrary distance scale.
        return -dists.mean()

    def _same_context_state_goal_pairs(
        self,
        data: BatchData,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        obs = data.observations
        idxs = torch.nonzero(valid_mask, as_tuple=False).flatten()
        if idxs.numel() <= 1:
            return idxs[:0], torch.empty((0,) + obs.shape[1:], device=obs.device, dtype=obs.dtype)

        context_ids = data.transition_infos.get("context_id") if data.transition_infos else None
        if context_ids is None:
            return idxs, torch.roll(obs[idxs], 1, dims=0)

        context_ids = context_ids.to(device=obs.device, dtype=torch.int64)
        ctx = context_ids[idxs]
        order = torch.argsort(ctx, stable=True)
        sorted_idxs = idxs[order]
        sorted_ctx = ctx[order]
        n = sorted_idxs.numel()

        new_group = torch.ones(n, device=obs.device, dtype=torch.bool)
        new_group[1:] = sorted_ctx[1:] != sorted_ctx[:-1]
        group_start = torch.nonzero(new_group, as_tuple=False).flatten()
        group_end = torch.empty_like(group_start)
        if group_start.numel() > 1:
            group_end[:-1] = group_start[1:]
        group_end[-1] = n

        group_sizes = group_end - group_start
        repeated_group_sizes = torch.repeat_interleave(group_sizes, group_sizes)
        keep = repeated_group_sizes > 1
        if not bool(keep.any()):
            return idxs[:0], torch.empty((0,) + obs.shape[1:], device=obs.device, dtype=obs.dtype)

        group_ids = torch.repeat_interleave(
            torch.arange(group_start.numel(), device=obs.device),
            group_sizes,
        )
        positions = torch.arange(n, device=obs.device)
        prev_positions = positions - 1
        prev_positions[new_group] = group_end[group_ids[new_group]] - 1

        source_idxs = sorted_idxs[keep]
        partner_idxs = sorted_idxs[prev_positions[keep]]
        return source_idxs, obs[partner_idxs]

    def forward(self, data: BatchData, critic_batch_info: CriticBatchInfo) -> LossResult:
        if data.transition_infos and "task_goal_observations" in data.transition_infos:
            device = critic_batch_info.zx.device
            source_terminal = data.transition_infos.get("source_terminal_goal_state")
            abstract_edge = data.transition_infos.get("abstract_goal_edge")
            valid_task = torch.ones(data.observations.shape[0], device=device, dtype=torch.bool)
            valid_state = torch.ones_like(valid_task)
            if source_terminal is not None:
                valid_task &= ~source_terminal.to(device=device, dtype=torch.bool)
            if abstract_edge is not None:
                normal_edge = ~abstract_edge.to(device=device, dtype=torch.bool)
                valid_task &= normal_edge
                valid_state &= normal_edge

            zero = critic_batch_info.zx.sum() * 0.0
            total_loss = zero
            info: Dict[str, torch.Tensor] = {}

            if bool(valid_task.any()) and float(self.abstract_goal_ratio) > 0.0:
                task_goals = data.transition_infos["task_goal_observations"].to(
                    device=device,
                    dtype=data.observations.dtype,
                )
                task_sources = data.transition_infos.get(
                    "global_push_task_source_observations"
                )
                if task_sources is None:
                    z_task_source = critic_batch_info.zx[valid_task]
                else:
                    task_sources = task_sources.to(
                        device=device,
                        dtype=data.observations.dtype,
                    )
                    z_task_source = critic_batch_info.critic.encoder(
                        task_sources[valid_task]
                    )
                z_task_goal = critic_batch_info.critic.encoder(task_goals[valid_task])
                d_task = critic_batch_info.critic.quasimetric_model(
                    z_task_source, z_task_goal
                )
                loss_task = self._push_loss(d_task)
                total_loss = total_loss + float(self.abstract_goal_ratio) * loss_task
                info["global_push_task_set/dist"] = d_task.mean()
                info["global_push_task_set/loss"] = loss_task
            else:
                info["global_push_task_set/dist"] = zero.detach()
                info["global_push_task_set/loss"] = zero.detach()

            explicit_sources = data.transition_infos.get("global_push_source_observations")
            explicit_goals = data.transition_infos.get("global_push_goal_observations")
            explicit_mask = data.transition_infos.get("global_push_pair_mask")
            if float(self.state_goal_ratio) > 0.0 and explicit_sources is not None and explicit_goals is not None:
                pair_mask = valid_state.clone()
                if explicit_mask is not None:
                    pair_mask &= explicit_mask.to(device=device, dtype=torch.bool)
                if bool(pair_mask.any()):
                    pair_sources = explicit_sources.to(device=device, dtype=data.observations.dtype)[pair_mask]
                    pair_goals = explicit_goals.to(device=device, dtype=data.observations.dtype)[pair_mask]
                    z_state_source = critic_batch_info.critic.encoder(pair_sources)
                    z_state_goal = critic_batch_info.critic.encoder(pair_goals)
                    d_state = critic_batch_info.critic.quasimetric_model(z_state_source, z_state_goal)
                    loss_state = self._push_loss(d_state)
                    total_loss = total_loss + float(self.state_goal_ratio) * loss_state
                    info["global_push_state_state/dist"] = d_state.mean()
                    info["global_push_state_state/loss"] = loss_state
                else:
                    info["global_push_state_state/dist"] = zero.detach()
                    info["global_push_state_state/loss"] = zero.detach()
            elif bool(valid_state.any()) and int(valid_state.sum().item()) > 1 and float(self.state_goal_ratio) > 0.0:
                state_source_idxs, state_goals = self._same_context_state_goal_pairs(data, valid_state)
                if state_goals.numel() > 0:
                    z_state_goal = critic_batch_info.critic.encoder(state_goals)
                    d_state = critic_batch_info.critic.quasimetric_model(critic_batch_info.zx[state_source_idxs], z_state_goal)
                    loss_state = self._push_loss(d_state)
                    total_loss = total_loss + float(self.state_goal_ratio) * loss_state
                    info["global_push_state_state/dist"] = d_state.mean()
                    info["global_push_state_state/loss"] = loss_state
                else:
                    info["global_push_state_state/dist"] = zero.detach()
                    info["global_push_state_state/loss"] = zero.detach()
            else:
                info["global_push_state_state/dist"] = zero.detach()
                info["global_push_state_state/loss"] = zero.detach()

            info["dist"] = (
                info["global_push_task_set/dist"] * float(self.abstract_goal_ratio)
                + info["global_push_state_state/dist"] * float(self.state_goal_ratio)
            )
            info["tsfm_dist"] = total_loss.detach()
            return LossResult(loss=total_loss, info=info)

        # Fallback for non goal-set datasets.
        dists = critic_batch_info.critic.quasimetric_model(
            critic_batch_info.zx,
            torch.roll(critic_batch_info.zy, 1, dims=0),
        )
        tsfm_dist = self._push_loss(dists)
        return LossResult(loss=tsfm_dist, info=dict(dist=dists.mean(), tsfm_dist=tsfm_dist))

    def extra_repr(self) -> str:
        return (
            "objective=linear_negative_mean, "
            f"abstract_goal_ratio={self.abstract_goal_ratio:g}, "
            f"state_goal_ratio={self.state_goal_ratio:g}"
        )
