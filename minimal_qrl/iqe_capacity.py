"""Shared IQE capacity configuration for training and checkpoint loading."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from quasimetric_rl.modules import QRLConf
from quasimetric_rl.modules.quasimetric_critic import QuasimetricCriticConf
from quasimetric_rl.modules.quasimetric_critic.models import QuasimetricCritic
from quasimetric_rl.modules.quasimetric_critic.models.quasimetric_model import (
    QuasimetricModel,
)


@dataclass(frozen=True)
class IQECapacity:
    """The only model dimensions varied by the capacity experiment."""

    dim: int = 2048
    components: int = 64

    def __post_init__(self) -> None:
        if int(self.dim) <= 0 or int(self.components) <= 0:
            raise ValueError("IQE dimension and component count must be positive")
        if int(self.dim) % int(self.components) != 0:
            raise ValueError("IQE dimension must be divisible by component count")

    @property
    def spec(self) -> str:
        return f"iqe(dim={int(self.dim)},components={int(self.components)})"

    def to_dict(self) -> dict[str, int]:
        return {key: int(value) for key, value in asdict(self).items()}


DEFAULT_IQE_CAPACITY = IQECapacity()


def qrl_conf_for_iqe_capacity(
    *,
    num_critics: int,
    capacity: IQECapacity,
) -> QRLConf:
    """Build the standard critic while changing only its IQE head capacity."""

    return QRLConf(
        actor=None,
        num_critics=int(num_critics),
        quasimetric_critic=QuasimetricCriticConf(
            model=QuasimetricCritic.Conf(
                quasimetric_model=QuasimetricModel.Conf(
                    projector_arch=(512,),
                    quasimetric_head_spec=capacity.spec,
                )
            )
        ),
    )


def iqe_capacity_from_checkpoint(checkpoint: Any) -> IQECapacity:
    """Recover capacity metadata, with old checkpoints mapping to 1x defaults."""

    if not isinstance(checkpoint, Mapping):
        return DEFAULT_IQE_CAPACITY
    explicit = checkpoint.get("model_capacity")
    if isinstance(explicit, Mapping):
        return IQECapacity(
            dim=int(explicit.get("dim", DEFAULT_IQE_CAPACITY.dim)),
            components=int(
                explicit.get("components", DEFAULT_IQE_CAPACITY.components)
            ),
        )
    config = checkpoint.get("config")
    if isinstance(config, Mapping):
        return IQECapacity(
            dim=int(config.get("iqe_dim", DEFAULT_IQE_CAPACITY.dim)),
            components=int(
                config.get("iqe_components", DEFAULT_IQE_CAPACITY.components)
            ),
        )
    return DEFAULT_IQE_CAPACITY
