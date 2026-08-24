from typing import *

import attrs

import torch
import torch.nn.functional as F

from ....data import BatchData

from ...utils import LossResult

from . import CriticLossBase, CriticBatchInfo



class LatentDynamicsLoss(CriticLossBase):
    r"""
    Section 3.4
    """

    @attrs.define(kw_only=True)
    class Conf:
        # config / argparse uses this to specify behavior

        weight: float = attrs.field(default=0.1, validator=attrs.validators.gt(0))

        def make(self) -> 'LatentDynamicsLoss':
            return LatentDynamicsLoss(
                weight=self.weight,
            )

    weight: float

    def __init__(self, *, weight: float):
        super().__init__()
        self.weight = weight

    def forward(self, data: BatchData, critic_batch_info: CriticBatchInfo) -> LossResult:
        zx = critic_batch_info.zx
        zy = critic_batch_info.zy
        actions = data.actions
        direct_mask = data.transition_infos.get("full_graph_direct_goal_edge") if data.transition_infos else None
        population_counts = data.transition_infos.get("full_graph_constraint_population_counts") if data.transition_infos else None
        importance_weights = None
        abstract_mask = data.transition_infos.get("abstract_goal_edge") if data.transition_infos else None
        if abstract_mask is not None:
            normal_mask = ~abstract_mask.to(device=zx.device, dtype=torch.bool)
            if not bool(normal_mask.any()):
                zero = zx.sum() * 0.0
                return LossResult(
                    loss=zero,
                    info=dict(sq_dists=zero.detach(), dist_p2n=zero.detach(), dist_n2p=zero.detach()),
                )
            zx = zx[normal_mask]
            zy = zy[normal_mask]
            actions = actions[normal_mask]
            if direct_mask is not None and population_counts is not None:
                direct_normal = direct_mask.to(
                    device=zx.device,
                    dtype=torch.bool,
                )[normal_mask]
                ordinary_normal = ~direct_normal
                counts = population_counts.to(device=zx.device, dtype=zx.dtype)
                importance_weights = torch.empty(
                    len(direct_normal),
                    device=zx.device,
                    dtype=zx.dtype,
                )
                importance_weights[ordinary_normal] = (
                    counts[0] / ordinary_normal.sum().clamp_min(1)
                )
                importance_weights[direct_normal] = (
                    counts[1] / direct_normal.sum().clamp_min(1)
                )

        pred_zy = critic_batch_info.critic.latent_dynamics(zx, actions)
        dists = critic_batch_info.critic.quasimetric_model(pred_zy, zy, bidirectional=True)
        dist_p2n, dist_n2p = dists.unbind(-1)
        if importance_weights is None:
            sq_dists = dists.square().mean()
            dist_p2n_mean = dist_p2n.mean()
            dist_n2p_mean = dist_n2p.mean()
        else:
            normalized_weights = importance_weights / importance_weights.sum()
            sq_dists = (
                dists.square().mean(dim=-1) * normalized_weights
            ).sum()
            dist_p2n_mean = (dist_p2n * normalized_weights).sum()
            dist_n2p_mean = (dist_n2p * normalized_weights).sum()
        return LossResult(
            loss=sq_dists * self.weight,
            info=dict(
                sq_dists=sq_dists,
                dist_p2n=dist_p2n_mean,
                dist_n2p=dist_n2p_mean,
            ),
        )

    def extra_repr(self) -> str:
        return f"weight={self.weight:g}"
