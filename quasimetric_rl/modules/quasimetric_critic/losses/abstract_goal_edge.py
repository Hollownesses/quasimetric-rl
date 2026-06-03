from typing import *

import attrs

import torch

from ....data import BatchData
from ...utils import LossResult

from . import CriticLossBase, CriticBatchInfo


class AbstractGoalEdgeLoss(CriticLossBase):
    @attrs.define(kw_only=True)
    class Conf:
        weight: float = attrs.field(default=1.0, validator=attrs.validators.ge(0))

        def make(self) -> 'AbstractGoalEdgeLoss':
            return AbstractGoalEdgeLoss(weight=self.weight)

    weight: float

    def __init__(self, *, weight: float):
        super().__init__()
        self.weight = float(weight)

    def forward(self, data: BatchData, critic_batch_info: CriticBatchInfo) -> LossResult:
        zero = critic_batch_info.zx.sum() * 0.0
        if self.weight == 0.0 or not data.transition_infos or "abstract_goal_edge" not in data.transition_infos:
            return LossResult(loss=zero, info=dict(dist=zero.detach(), sq_dist=zero.detach(), count=zero.detach()))

        mask = data.transition_infos["abstract_goal_edge"].to(device=critic_batch_info.zx.device, dtype=torch.bool)
        if not bool(mask.any()):
            return LossResult(loss=zero, info=dict(dist=zero.detach(), sq_dist=zero.detach(), count=zero.detach()))

        task_goals = data.transition_infos.get("task_goal_observations", data.next_observations)
        task_goals = task_goals.to(device=critic_batch_info.zx.device, dtype=data.observations.dtype)
        z_goal = critic_batch_info.critic.encoder(task_goals[mask])
        dists = critic_batch_info.critic.quasimetric_model(critic_batch_info.zx[mask], z_goal)
        sq_dist = dists.square().mean()
        return LossResult(
            loss=sq_dist * float(self.weight),
            info=dict(
                dist=dists.mean(),
                sq_dist=sq_dist,
                count=torch.as_tensor(float(mask.sum().item()), device=critic_batch_info.zx.device),
            ),
        )

    def extra_repr(self) -> str:
        return f"weight={self.weight:g}"
