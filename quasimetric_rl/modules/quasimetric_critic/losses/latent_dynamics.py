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

        pred_zy = critic_batch_info.critic.latent_dynamics(zx, actions)
        dists = critic_batch_info.critic.quasimetric_model(pred_zy, zy, bidirectional=True)
        sq_dists = dists.square().mean()

        dist_p2n, dist_n2p = dists.unbind(-1)
        return LossResult(
            loss=sq_dists * self.weight,
            info=dict(sq_dists=sq_dists, dist_p2n=dist_p2n.mean(), dist_n2p=dist_n2p.mean()),
        )

    def extra_repr(self) -> str:
        return f"weight={self.weight:g}"
