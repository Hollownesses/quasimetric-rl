from __future__ import annotations
from typing import *

import attrs

import numpy as np
import torch
import torch.utils.data
import gym

from omegaconf import MISSING

from .utils import TensorCollectionAttrsMixin
from .env_spec import EnvSpec



#-----------------------------------------------------------------------------#
#-------------------------------- Batch data ---------------------------------#
#-----------------------------------------------------------------------------#

# What should be in a batch


@attrs.define(kw_only=True)
class BatchData(TensorCollectionAttrsMixin):  # TensorCollectionAttrsMixin has some util methods
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    rewards: torch.Tensor
    terminals: torch.Tensor
    timeouts: torch.Tensor

    future_observations: torch.Tensor  # sampled!
    transition_infos: Mapping[str, torch.Tensor] = attrs.Factory(dict)

    @property
    def device(self) -> torch.device:
        return self.observations.device

    @property
    def batch_shape(self) -> torch.Size:
        return self.terminals.shape

    @property
    def num_transitions(self) -> int:
        return self.terminals.numel()



#-----------------------------------------------------------------------------#
#------------------------------- Episode data --------------------------------#
#-----------------------------------------------------------------------------#


@attrs.define(kw_only=True)
class MultiEpisodeData(TensorCollectionAttrsMixin):
    r"""
    The DATASET of MULTIPLE episodes
    """


    # For each episode, L: number of (s, a, s', r, d, to) pairs, so number of transitions (not observations)
    episode_lengths: torch.Tensor
    # cat all states from all episodes, where the last s' is added. I.e., each episode has L+1 states
    all_observations: torch.Tensor
    # cat all actions from all episodes. Each episode has L actions.
    actions: torch.Tensor
    # cat all rewards from all episodes. Each episode has L rewards.
    rewards: torch.Tensor
    # cat all terminals from all episodes. Each episode has L terminals.
    terminals: torch.Tensor
    # cat all timeouts from all episodes. Each episode has L timeouts.
    timeouts: torch.Tensor
    # cat all observation infos from all episodes. Each episode has L + 1 elements.
    observation_infos: Mapping[str, torch.Tensor] = attrs.Factory(dict)
    # cat all transition infos from all episodes. Each episode has L elements.
    transition_infos: Mapping[str, torch.Tensor] = attrs.Factory(dict)

    @property
    def num_episodes(self) -> int:
        return self.episode_lengths.shape[0]

    @property
    def num_transitions(self) -> int:
        return self.rewards.shape[0]

    def __attrs_post_init__(self):
        assert self.episode_lengths.ndim == 1
        N = self.num_transitions
        assert N > 0
        assert self.all_observations.ndim >= 1 and self.all_observations.shape[0] == (N + self.num_episodes), self.all_observations.shape
        assert self.actions.ndim >= 1 and self.actions.shape[0] == N
        assert self.rewards.ndim == 1 and self.rewards.shape[0] == N
        assert self.terminals.ndim == 1 and self.terminals.shape[0] == N
        assert self.timeouts.ndim == 1 and self.timeouts.shape[0] == N
        for k, v in self.observation_infos.items():
            assert v.shape[0] == N + self.num_episodes, k
        for k, v in self.transition_infos.items():
            assert v.shape[0] == N, k



@attrs.define(kw_only=True)
class EpisodeData(MultiEpisodeData):
    r"""
    A SINGLE episode
    """

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        assert self.num_episodes == 1

    @classmethod
    def from_simple_trajectory(cls,
                               observations: Union[np.ndarray, torch.Tensor],
                               actions: Union[np.ndarray, torch.Tensor],
                               next_observations: Union[np.ndarray, torch.Tensor],
                               rewards: Union[np.ndarray, torch.Tensor],
                               terminals: Union[np.ndarray, torch.Tensor],
                               timeouts: Union[np.ndarray, torch.Tensor],
                               *,
                               observation_infos: Optional[Mapping[str, Union[np.ndarray, torch.Tensor]]] = None,
                               transition_infos: Optional[Mapping[str, Union[np.ndarray, torch.Tensor]]] = None):
        observations = torch.tensor(observations)
        next_observations=torch.tensor(next_observations)
        all_observations = torch.cat([observations, next_observations[-1:]], dim=0)
        return cls(
            episode_lengths=torch.tensor([observations.shape[0]]),
            all_observations=all_observations,
            actions=torch.tensor(actions),
            rewards=torch.tensor(rewards),
            terminals=torch.tensor(terminals),
            timeouts=torch.tensor(timeouts),
            observation_infos={
                k: torch.tensor(v)
                for k, v in (observation_infos or {}).items()
            },
            transition_infos={
                k: torch.tensor(v)
                for k, v in (transition_infos or {}).items()
            },
        )


#-----------------------------------------------------------------------------#
#--------------------------------- dataset -----------------------------------#
#-----------------------------------------------------------------------------#


# Each env is specified with two strings:
#   + kind  # d4rl, gcrl, etc.
#   + spec  # maze2d-umaze-v1, FetchPushImage, etc.


LOAD_EPISODES_REGISTRY: Mapping[Tuple[str, str], Callable[[], Iterator[EpisodeData]]] = {}
CREATE_ENV_REGISTRY: Mapping[Tuple[str, str], Callable[[], gym.Env]] = {}


def register_offline_env(kind: str, spec: str, *, load_episodes_fn, create_env_fn):
    r"""
    Each specific env (e.g., an offline env from d4rl) just needs to register

        1. how to load the episodes
        (this is optional in online settings. see ReplayBuffer)

        load_episodes_fn() -> Iterator[EpisodeData]

        2. how to create an env

        create_env_fn() -> gym.Env

     See d4rl/maze2d.py for example
    """
    assert (kind, spec) not in LOAD_EPISODES_REGISTRY
    LOAD_EPISODES_REGISTRY[(kind, spec)] = load_episodes_fn
    CREATE_ENV_REGISTRY[(kind, spec)] = create_env_fn


class Dataset:
    @attrs.define(frozen=True, kw_only=True)
    class MQEInspiredWaypointSampling:
        """Sampling-only subset adapted from MQE's multistep relabeling.

        Physical goals and waypoints stay inside one episode.  A fixed fraction
        of every MQE batch is independently resampled from naturally successful
        episodes and connected to the abstract task goal through that episode's
        physical terminal state.  This makes the anchor rate independent of the
        (usually very small) successful-transition rate in the main dataloader.
        """

        goal_discount: float = attrs.field(
            default=0.995,
            validator=attrs.validators.and_(
                attrs.validators.ge(0.0), attrs.validators.lt(1.0)
            ),
        )
        waypoint_lambda: float = attrs.field(
            default=0.95,
            validator=attrs.validators.and_(
                attrs.validators.ge(0.0), attrs.validators.lt(1.0)
            ),
        )
        next_state_probability: float = attrs.field(
            default=0.2,
            validator=attrs.validators.and_(
                attrs.validators.ge(0.0), attrs.validators.le(1.0)
            ),
        )
        terminal_anchor_fraction: float = attrs.field(
            default=0.1,
            validator=attrs.validators.and_(
                attrs.validators.ge(0.0), attrs.validators.le(1.0)
            ),
        )

    @attrs.define(kw_only=True)
    class Conf:
        # config / argparse uses this to specify behavior

        kind: str = MISSING  # d4rl, gcrl, etc.
        name: str = MISSING  # maze2d-umaze-v1, etc.

        # Defines how to fetch the future observation. smaller -> more recent
        future_observation_discount: float = attrs.field(default=0.99, validator=attrs.validators.and_(
            attrs.validators.ge(0.0),
            attrs.validators.le(1.0),
        ))

        def make(self, *, dummy: bool = False) -> 'Dataset':
            return Dataset(self.kind, self.name,
                           future_observation_discount=self.future_observation_discount,
                           dummy=dummy)

    kind: str
    name: str
    future_observation_discount: float

    # Computed Attributes::

    # Data
    raw_data: MultiEpisodeData  # will contain all episodes

    # Env info
    env_spec: EnvSpec

    # Defines how to fetch the future observation. smaller -> more recent
    future_observation_discount: float

    # Auxiliary structures that helps fetching transitions of specific kinds
    # -----
    obs_indices_to_obs_index_in_episode: torch.Tensor
    indices_to_episode_indices: torch.Tensor  # episode indices refers to indices in this split
    indices_to_episode_timesteps: torch.Tensor
    obs_indices_to_cumulative_cost: torch.Tensor
    max_episode_length: int
    # -----

    def create_env(self) -> gym.Env:
        return CREATE_ENV_REGISTRY[self.kind, self.name]()

    def load_episodes(self) -> Iterator[EpisodeData]:
        return LOAD_EPISODES_REGISTRY[self.kind, self.name]()

    def __init__(self, kind: str, name: str, *,
                 future_observation_discount: float,
                 dummy: bool = False,  # when you don't want to load data, e.g., in analysis
                 ) -> None:
        self.kind = kind
        self.name = name
        self.future_observation_discount = future_observation_discount

        self.env_spec = EnvSpec.from_env(self.create_env())

        assert 0 <= future_observation_discount
        self.future_observation_discount = future_observation_discount

        if not dummy:
            episodes = tuple(self.load_episodes())
        else:
            from .online.utils import get_empty_episode
            episodes = (get_empty_episode(self.env_spec, episode_length=1),)

        obs_indices_to_obs_index_in_episode = []
        indices_to_episode_indices = []
        indices_to_episode_timesteps = []
        obs_indices_to_cumulative_cost = []
        legal_anchor_transition_indices = []
        transition_offset = 0
        for eidx, episode in enumerate(episodes):
            l = episode.num_transitions
            obs_indices_to_obs_index_in_episode.append(torch.arange(l + 1, dtype=torch.int64))
            indices_to_episode_indices.append(torch.full([l], eidx, dtype=torch.int64))
            indices_to_episode_timesteps.append(torch.arange(l, dtype=torch.int64))
            costs = (-episode.rewards.to(dtype=torch.float32)).clamp_min(0.0)
            obs_indices_to_cumulative_cost.append(torch.cat([
                torch.zeros(1, dtype=costs.dtype),
                torch.cumsum(costs, dim=0),
            ]))
            abstract_edges = episode.transition_infos.get("abstract_goal_edge")
            task_success = episode.transition_infos.get("task_success_episode")
            task_goals = episode.transition_infos.get("task_goal_observations")
            has_legal_task_terminal = (
                bool(episode.terminals[-1])
                and task_success is not None
                and bool(task_success[-1])
                and task_goals is not None
                and not (
                    abstract_edges is not None
                    and bool(abstract_edges[-1])
                )
            )
            if has_legal_task_terminal:
                legal_anchor_transition_indices.append(
                    torch.arange(
                        transition_offset,
                        transition_offset + l,
                        dtype=torch.int64,
                    )
                )
            transition_offset += l

        assert len(episodes) > 0, "must have at least one episode"
        self.raw_data = MultiEpisodeData.cat(episodes)

        self.obs_indices_to_obs_index_in_episode = torch.cat(obs_indices_to_obs_index_in_episode, dim=0)
        self.indices_to_episode_indices = torch.cat(indices_to_episode_indices, dim=0)
        self.indices_to_episode_timesteps = torch.cat(indices_to_episode_timesteps, dim=0)
        self.obs_indices_to_cumulative_cost = torch.cat(obs_indices_to_cumulative_cost, dim=0)
        self.legal_anchor_transition_indices = (
            torch.cat(legal_anchor_transition_indices)
            if legal_anchor_transition_indices
            else torch.empty(0, dtype=torch.int64)
        )
        self.max_episode_length = self.raw_data.episode_lengths.max().item()
        self.mqe_inspired_waypoint_sampling: Optional[
            Dataset.MQEInspiredWaypointSampling
        ] = None

    def configure_mqe_inspired_waypoint_sampling(
        self,
        config: Optional[MQEInspiredWaypointSampling],
    ) -> None:
        """Enable isolated MQE samples without changing the main replay batch."""

        if (
            config is not None
            and config.terminal_anchor_fraction > 0.0
            and self.legal_anchor_transition_indices.numel() == 0
        ):
            raise ValueError(
                "MQE terminal anchors require at least one complete, naturally "
                "successful episode with task_goal_observations"
            )
        self.mqe_inspired_waypoint_sampling = config

    @staticmethod
    def _geometric_steps(
        continuation_probability: float,
        shape: torch.Size,
    ) -> torch.Tensor:
        """Sample Geom(1-continuation_probability) on {1, 2, ...}."""

        continuation_probability = float(continuation_probability)
        if continuation_probability == 0.0:
            return torch.ones(shape, dtype=torch.int64)
        failures = torch.distributions.Geometric(
            probs=torch.full(shape, 1.0 - continuation_probability)
        ).sample()
        return failures.to(dtype=torch.int64) + 1

    def _mqe_inspired_waypoint_infos(
        self,
        *,
        indices: torch.Tensor,
        episode_indices: torch.Tensor,
        observation_indices: torch.Tensor,
        timesteps: torch.Tensor,
        episode_lengths: torch.Tensor,
    ) -> Mapping[str, torch.Tensor]:
        config = self.mqe_inspired_waypoint_sampling
        if config is None:
            return {}

        # Start from the unchanged replay batch.  MQE-only anchor slots replace
        # their source with a transition sampled from the legal success pool;
        # no other loss sees this replacement.
        source_indices = indices.clone()
        source_episode_indices = episode_indices.clone()
        source_observation_indices = observation_indices.clone()
        source_timesteps = timesteps.clone()
        source_episode_lengths = episode_lengths.clone()
        terminal_anchor = torch.zeros(indices.shape, dtype=torch.bool)

        batch_size = int(indices.numel())
        if config.terminal_anchor_fraction > 0.0 and batch_size > 0:
            anchor_count = min(
                batch_size,
                max(
                    1,
                    int(np.ceil(batch_size * config.terminal_anchor_fraction)),
                ),
            )
            anchor_slots = torch.randperm(batch_size)[:anchor_count]
            anchor_pool_slots = torch.randint(
                self.legal_anchor_transition_indices.numel(),
                (anchor_count,),
            )
            anchor_indices = self.legal_anchor_transition_indices[anchor_pool_slots]
            anchor_episode_indices = self.indices_to_episode_indices[anchor_indices]
            source_indices[anchor_slots] = anchor_indices
            source_episode_indices[anchor_slots] = anchor_episode_indices
            source_observation_indices[anchor_slots] = (
                anchor_indices + anchor_episode_indices
            )
            source_timesteps[anchor_slots] = self.indices_to_episode_timesteps[
                anchor_indices
            ]
            source_episode_lengths[anchor_slots] = self.raw_data.episode_lengths[
                anchor_episode_indices
            ]
            terminal_anchor[anchor_slots] = True

        shape = source_indices.shape
        remaining_steps = source_episode_lengths - source_timesteps
        abstract_edges = self.raw_data.transition_infos.get("abstract_goal_edge")
        if abstract_edges is None:
            valid = torch.ones(shape, dtype=torch.bool)
        else:
            valid = ~abstract_edges[source_indices].to(dtype=torch.bool)
        valid &= remaining_steps >= 1

        goal_steps = torch.minimum(
            self._geometric_steps(config.goal_discount, shape),
            remaining_steps,
        )
        geometric_waypoint_steps = self._geometric_steps(
            config.waypoint_lambda,
            shape,
        )
        forced_one_step = (
            torch.rand(shape) < config.next_state_probability
        ) & valid & ~terminal_anchor
        waypoint_steps = torch.where(
            forced_one_step,
            torch.ones_like(geometric_waypoint_steps),
            geometric_waypoint_steps,
        )
        waypoint_steps = torch.minimum(waypoint_steps, goal_steps)

        # Abstract G is admitted only through the physical terminal state of a
        # complete natural success.  It is never used as a waypoint.
        goal_steps = torch.where(terminal_anchor, remaining_steps, goal_steps)
        waypoint_steps = torch.where(
            terminal_anchor,
            remaining_steps,
            waypoint_steps,
        )

        safe_waypoint_steps = torch.where(
            valid, waypoint_steps, torch.zeros_like(waypoint_steps)
        )
        safe_goal_steps = torch.where(
            valid, goal_steps, torch.zeros_like(goal_steps)
        )
        waypoint_observation_indices = (
            source_observation_indices + safe_waypoint_steps
        )
        goal_observation_indices = source_observation_indices + safe_goal_steps
        source_observations = self.get_observations(source_observation_indices)
        waypoint_observations = self.get_observations(
            waypoint_observation_indices
        )
        physical_goals = self.get_observations(goal_observation_indices)
        task_goals = self.raw_data.transition_infos.get("task_goal_observations")
        if task_goals is None:
            goal_observations = physical_goals
        else:
            task_goal_observations = task_goals[source_indices]
            goal_mask = terminal_anchor.reshape(
                terminal_anchor.shape + (1,) * (physical_goals.ndim - 1)
            )
            goal_observations = torch.where(
                goal_mask,
                task_goal_observations,
                physical_goals,
            )
        waypoint_cost = (
            self.obs_indices_to_cumulative_cost[waypoint_observation_indices]
            - self.obs_indices_to_cumulative_cost[source_observation_indices]
        )

        return {
            "mqe_waypoint_valid": valid,
            "mqe_waypoint_source_observations": source_observations,
            "mqe_waypoint_observations": waypoint_observations,
            "mqe_waypoint_goal_observations": goal_observations,
            "mqe_waypoint_cost": waypoint_cost,
            "mqe_waypoint_steps": waypoint_steps,
            "mqe_waypoint_goal_steps": goal_steps,
            "mqe_waypoint_forced_one_step": forced_one_step,
            "mqe_waypoint_physical_goal": valid & ~terminal_anchor,
            "mqe_waypoint_terminal_anchor": terminal_anchor & valid,
            "mqe_waypoint_episode_index": source_episode_indices,
            "mqe_waypoint_source_transition_index": source_indices,
        }

    def get_observations(self, obs_indices: torch.Tensor):
        return self.raw_data.all_observations[obs_indices]

    def __getitem__(self, indices: torch.Tensor) -> BatchData:
        indices = torch.as_tensor(indices)
        eindices = self.indices_to_episode_indices[indices]
        obs_indices = indices + eindices  # index for `observation`: skip the s_last from previous episodes
        obs = self.get_observations(obs_indices)
        nobs = self.get_observations(obs_indices + 1)

        terminals = self.raw_data.terminals[indices]

        tindices = self.indices_to_episode_timesteps[indices]
        epilengths = self.raw_data.episode_lengths[eindices]  # max idx is this
        deltas = torch.arange(self.max_episode_length)
        pdeltas = torch.where(
            # test tidx + 1 + delta <= max_idx = epi_length
            (tindices[:, None] + deltas) < epilengths[:, None],
            self.future_observation_discount ** deltas,
            0,
        )
        deltas = torch.distributions.Categorical(
            probs=pdeltas,
        ).sample()
        future_obs_indices = obs_indices + 1 + deltas
        future_observations = self.get_observations(future_obs_indices)
        future_costs = (
            self.obs_indices_to_cumulative_cost[future_obs_indices]
            - self.obs_indices_to_cumulative_cost[obs_indices]
        )
        transition_infos = {
            k: v[indices]
            for k, v in self.raw_data.transition_infos.items()
        }
        transition_infos.update({
            "temporal_future_cost": future_costs,
            "temporal_future_steps": deltas + 1,
        })
        transition_infos.update(self._mqe_inspired_waypoint_infos(
            indices=indices,
            episode_indices=eindices,
            observation_indices=obs_indices,
            timesteps=tindices,
            episode_lengths=epilengths,
        ))

        return BatchData(
            observations=obs,
            actions=self.raw_data.actions[indices],
            next_observations=nobs,
            future_observations=future_observations,
            rewards=self.raw_data.rewards[indices],
            terminals=terminals,
            timeouts=self.raw_data.timeouts[indices],
            transition_infos=transition_infos,
        )

    def __len__(self):
        return self.raw_data.num_transitions

    def __repr__(self):
        return rf"""
{self.__class__.__name__}(
    kind={self.kind!r},
    name={self.name!r},
    future_observation_discount={self.future_observation_discount!r},
    env_spec={self.env_spec!r},
)""".lstrip('\n')

    def get_dataloader(self, *,
                       batch_size: int, shuffle: bool = False,
                       drop_last: bool = False,
                       pin_memory: bool = False,
                       num_workers: int = 0, persistent_workers: bool = False,
                       successful_transition_weight: float = 1.0,
                       **kwargs) -> torch.utils.data.DataLoader:
        successful_transition_weight = float(successful_transition_weight)
        if successful_transition_weight <= 0.0:
            raise ValueError("successful_transition_weight must be positive")
        success_mask = self.raw_data.transition_infos.get("task_success_episode")
        if success_mask is not None and successful_transition_weight != 1.0:
            sample_weights = torch.ones(len(self), dtype=torch.float64)
            success_mask = success_mask.to(dtype=torch.bool)
            sample_weights[success_mask] = successful_transition_weight
            base_sampler = torch.utils.data.WeightedRandomSampler(
                sample_weights,
                num_samples=len(self),
                replacement=True,
            )
        else:
            base_sampler = torch.utils.data.RandomSampler(self)
        sampler = torch.utils.data.BatchSampler(
            base_sampler,
            batch_size=batch_size,
            drop_last=drop_last,
        )
        return torch.utils.data.DataLoader(
            self,
            batch_size=None,
            sampler=sampler,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            **kwargs,
        )


def seed_worker(_):
    worker_seed = torch.utils.data.get_worker_info().seed % (2 ** 32)
    np.random.seed(worker_seed)


from . import d4rl  # register
