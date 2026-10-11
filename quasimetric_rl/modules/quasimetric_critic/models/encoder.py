from typing import *

import attrs

import torch
import torch.nn as nn

from ...utils import MLP, LatentTensor

from ....data import EnvSpec
from ....data.env_spec.input_encoding import InputEncoding


class Encoder(nn.Module):
    r"""
    (*, *input_shape)                      Input
           |
     [input_encoding]                      e.g., AtariTorso network to map image input to a flat vector
           |
        (*, d)                             Encoded 1-D input
           |
    [MLP specified by arch]                ENCODER
           |
        (*, latent_size)                   1-D Latent
    """

    @attrs.define(kw_only=True)
    class Conf:
        # config / argparse uses this to specify behavior

        arch: Tuple[int, ...] = (512, 512)
        latent_size: int = 128

        def make(self, *, env_spec: EnvSpec) -> 'Encoder':
            return Encoder(
                env_spec=env_spec,
                arch=self.arch,
                latent_size=self.latent_size,
            )

    input_shape: torch.Size
    input_encoding: InputEncoding
    encoder: MLP
    latent_size: int

    def __init__(self, *, env_spec: EnvSpec,
                 arch: Tuple[int, ...], latent_size: int, **kwargs):
        super().__init__(**kwargs)
        self.input_shape = env_spec.observation_shape
        self.input_encoding = env_spec.make_observation_input()
        encoder_input_size = self.input_encoding.output_size
        self.encoder = MLP(encoder_input_size, latent_size, hidden_sizes=arch)
        self.latent_size = latent_size
        # Persist this switch so evaluation reconstructs the same quotient map
        # when loading a feasible-direction checkpoint into a plain QRL agent.
        self.register_buffer("comm_goal_set_quotient", torch.tensor(False))

    def enable_comm_goal_set_quotient(self) -> None:
        if tuple(self.input_shape) != (42,):
            raise ValueError("comm goal-set quotient requires 42-dimensional task observations")
        self.comm_goal_set_quotient.fill_(True)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Checkpoints created before the quotient switch retain their old behavior.
        key = prefix + "comm_goal_set_quotient"
        if key not in state_dict:
            state_dict[key] = self.comm_goal_set_quotient.detach().clone()
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def _canonicalize_comm_goal_set(self, x: torch.Tensor) -> torch.Tensor:
        # Field positions follow CommInspectionDubinsUAV2D.TASK_CONTEXT_FIELDS.
        # Its abstract goal has these task fields, heading cos=1, and all other
        # state fields zero. Every feasible terminal of that task maps to it.
        terminal_or_goal = (x[..., 40] > 0.5) | (x[..., 41] > 0.5)
        canonical = torch.zeros_like(x)
        canonical[..., 2] = 1.0
        canonical[..., 4:10] = x[..., 4:10]
        canonical[..., 20:26] = x[..., 20:26]
        canonical[..., 41] = 1.0
        return torch.where(terminal_or_goal.unsqueeze(-1), canonical, x)

    def forward(self, x: torch.Tensor) -> LatentTensor:
        if bool(self.comm_goal_set_quotient):
            x = self._canonicalize_comm_goal_set(x)
        return self.encoder(self.input_encoding(x))

    # for type hint
    def __call__(self, x: torch.Tensor) -> LatentTensor:
        return super().__call__(x)

    def extra_repr(self) -> str:
        return f"input_shape={self.input_shape}, latent_size={self.latent_size}"
