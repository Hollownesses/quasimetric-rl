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

        assert len(episodes) > 0, "must have at least one episode"
        self.raw_data = MultiEpisodeData.cat(episodes)

        self.obs_indices_to_obs_index_in_episode = torch.cat(obs_indices_to_obs_index_in_episode, dim=0)
        self.indices_to_episode_indices = torch.cat(indices_to_episode_indices, dim=0)
        self.indices_to_episode_timesteps = torch.cat(indices_to_episode_timesteps, dim=0)
        self.obs_indices_to_cumulative_cost = torch.cat(obs_indices_to_cumulative_cost, dim=0)
        self.max_episode_length = self.raw_data.episode_lengths.max().item()

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
                       dense_transition_fraction: Optional[float] = None,
                       full_graph_stratified_constraints: bool = False,
                       **kwargs) -> torch.utils.data.DataLoader:
        successful_transition_weight = float(successful_transition_weight)
        if successful_transition_weight <= 0.0:
            raise ValueError("successful_transition_weight must be positive")
        success_mask = self.raw_data.transition_infos.get("task_success_episode")
        if full_graph_stratified_constraints:
            if successful_transition_weight != 1.0:
                raise ValueError(
                    "full-graph stratified batches cannot use success reweighting"
                )
            if dense_transition_fraction is not None:
                raise ValueError(
                    "full-graph and dense-transition batch samplers are mutually exclusive"
                )
            sampler = FullGraphConstraintBatchSampler(
                direct_goal_mask=self.raw_data.transition_infos.get(
                    "full_graph_direct_goal_edge"
                ),
                terminal_goal_mask=self.raw_data.transition_infos.get(
                    "abstract_goal_edge"
                ),
                ordinary_batch_size=int(batch_size),
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
        if dense_transition_fraction is not None:
            if successful_transition_weight != 1.0:
                raise ValueError(
                    "dense-transition batches cannot also use success reweighting"
                )
            dense_mask = self.raw_data.transition_infos.get(
                "dense_u_trap_transition"
            )
            dense_strata = self.raw_data.transition_infos.get(
                "dense_u_trap_stratum"
            )
            if dense_mask is None or dense_strata is None:
                raise ValueError("dense-transition metadata is missing from dataset")
            sampler = DenseTransitionBatchSampler(
                dense_mask=dense_mask,
                dense_strata=dense_strata,
                batch_size=batch_size,
                local_fraction=float(dense_transition_fraction),
                num_batches=(len(self) // batch_size if drop_last else int(np.ceil(len(self) / batch_size))),
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


class IndependentTransitionDataset(Dataset):
    """Dataset for graph edges that are independent one-step transitions.

    Unlike ``EpisodeData``, this representation does not require one edge's
    destination to equal the next edge's source.  It is therefore the exact
    storage model for a finite directed graph enumerated as ``(s, a, s', c)``.
    """

    def __init__(
        self,
        *,
        env: gym.Env,
        observations: Union[np.ndarray, torch.Tensor],
        actions: Union[np.ndarray, torch.Tensor],
        next_observations: Union[np.ndarray, torch.Tensor],
        rewards: Union[np.ndarray, torch.Tensor],
        terminals: Union[np.ndarray, torch.Tensor],
        timeouts: Union[np.ndarray, torch.Tensor],
        transition_infos: Optional[
            Mapping[str, Union[np.ndarray, torch.Tensor]]
        ] = None,
        uniform_task_source_observation_pool: Optional[
            Union[np.ndarray, torch.Tensor]
        ] = None,
        name: str = "independent_transitions",
    ) -> None:
        self.kind = "independent_transitions"
        self.name = str(name)
        self.future_observation_discount = 0.0
        self.env_spec = EnvSpec.from_env(env)
        observations = torch.as_tensor(observations)
        actions = torch.as_tensor(actions)
        next_observations = torch.as_tensor(next_observations)
        rewards = torch.as_tensor(rewards)
        terminals = torch.as_tensor(terminals)
        timeouts = torch.as_tensor(timeouts)
        count = int(rewards.shape[0])
        if count <= 0:
            raise ValueError("independent transition dataset must be non-empty")
        for label, value in (
            ("observations", observations),
            ("actions", actions),
            ("next_observations", next_observations),
            ("terminals", terminals),
            ("timeouts", timeouts),
        ):
            if int(value.shape[0]) != count:
                raise ValueError(f"{label} must contain {count} transitions")
        infos = {
            key: torch.as_tensor(value)
            for key, value in (transition_infos or {}).items()
        }
        for key, value in infos.items():
            if int(value.shape[0]) != count:
                raise ValueError(
                    f"transition info {key!r} must contain {count} transitions"
                )
        self.raw_data = BatchData(
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            rewards=rewards,
            terminals=terminals,
            timeouts=timeouts,
            future_observations=next_observations,
            transition_infos=infos,
        )
        self.uniform_task_source_observation_pool = (
            torch.as_tensor(uniform_task_source_observation_pool)
            if uniform_task_source_observation_pool is not None
            else None
        )
        if (
            self.uniform_task_source_observation_pool is not None
            and len(self.uniform_task_source_observation_pool) <= 0
        ):
            raise ValueError("uniform task-source observation pool must be non-empty")
        self.full_graph_constraint_population_counts = None
        if self.uniform_task_source_observation_pool is not None:
            terminal = infos.get("abstract_goal_edge")
            direct = infos.get("full_graph_direct_goal_edge")
            if terminal is None or direct is None:
                raise ValueError(
                    "stratified full-graph dataset requires constraint-family masks"
                )
            terminal = terminal.to(dtype=torch.bool)
            direct = direct.to(dtype=torch.bool) & ~terminal
            ordinary = ~direct & ~terminal
            self.full_graph_constraint_population_counts = torch.stack(
                [ordinary.sum(), direct.sum(), terminal.sum()]
            ).to(dtype=torch.float32)

    def __getitem__(self, indices: torch.Tensor) -> BatchData:
        indices = torch.as_tensor(indices)
        transition_infos = {
            key: value[indices]
            for key, value in self.raw_data.transition_infos.items()
        }
        if self.uniform_task_source_observation_pool is not None:
            pool_indices = torch.randint(
                len(self.uniform_task_source_observation_pool),
                (int(indices.numel()),),
            )
            transition_infos["global_push_task_source_observations"] = (
                self.uniform_task_source_observation_pool[pool_indices]
            )
            transition_infos["full_graph_constraint_population_counts"] = (
                self.full_graph_constraint_population_counts
            )
        return BatchData(
            observations=self.raw_data.observations[indices],
            actions=self.raw_data.actions[indices],
            next_observations=self.raw_data.next_observations[indices],
            rewards=self.raw_data.rewards[indices],
            terminals=self.raw_data.terminals[indices],
            timeouts=self.raw_data.timeouts[indices],
            future_observations=self.raw_data.next_observations[indices],
            transition_infos=transition_infos,
        )


class FullGraphConstraintBatchSampler(torch.utils.data.Sampler):
    """Sample ordinary edges while including every goal-bound edge per batch."""

    def __init__(
        self,
        *,
        direct_goal_mask: Optional[torch.Tensor],
        terminal_goal_mask: Optional[torch.Tensor],
        ordinary_batch_size: int,
    ) -> None:
        super().__init__()
        if direct_goal_mask is None or terminal_goal_mask is None:
            raise ValueError("full-graph constraint-family metadata is missing")
        if int(ordinary_batch_size) <= 0:
            raise ValueError("ordinary_batch_size must be positive")
        terminal = torch.as_tensor(terminal_goal_mask, dtype=torch.bool)
        direct = torch.as_tensor(direct_goal_mask, dtype=torch.bool) & ~terminal
        ordinary = ~direct & ~terminal
        self.ordinary_indices = torch.nonzero(ordinary, as_tuple=False).flatten()
        self.direct_goal_indices = torch.nonzero(direct, as_tuple=False).flatten()
        self.terminal_goal_indices = torch.nonzero(
            terminal,
            as_tuple=False,
        ).flatten()
        self.ordinary_batch_size = int(ordinary_batch_size)
        if not len(self.ordinary_indices):
            raise ValueError("full-graph dataset has no ordinary edges")
        if not len(self.direct_goal_indices):
            raise ValueError("full-graph dataset has no direct-to-G edges")
        if not len(self.terminal_goal_indices):
            raise ValueError("full-graph dataset has no terminal-to-G edges")

    @property
    def total_batch_size(self) -> int:
        return (
            self.ordinary_batch_size
            + len(self.direct_goal_indices)
            + len(self.terminal_goal_indices)
        )

    def __len__(self) -> int:
        return int(np.ceil(len(self.ordinary_indices) / self.ordinary_batch_size))

    def __iter__(self):
        permutation = self.ordinary_indices[
            torch.randperm(len(self.ordinary_indices))
        ]
        for begin in range(0, len(permutation), self.ordinary_batch_size):
            ordinary = permutation[begin : begin + self.ordinary_batch_size]
            if len(ordinary) < self.ordinary_batch_size:
                padding = self.ordinary_indices[
                    torch.randint(
                        len(self.ordinary_indices),
                        (self.ordinary_batch_size - len(ordinary),),
                    )
                ]
                ordinary = torch.cat([ordinary, padding])
            batch = torch.cat(
                [ordinary, self.direct_goal_indices, self.terminal_goal_indices]
            )
            yield batch[torch.randperm(len(batch))]


class DenseTransitionBatchSampler(torch.utils.data.Sampler):
    """Exact global/local batches with equal coverage of non-empty local strata."""

    def __init__(
        self,
        *,
        dense_mask: torch.Tensor,
        dense_strata: torch.Tensor,
        batch_size: int,
        local_fraction: float,
        num_batches: int,
    ) -> None:
        super().__init__()
        if not 0.0 < float(local_fraction) < 1.0:
            raise ValueError("local_fraction must lie strictly between 0 and 1")
        self.batch_size = int(batch_size)
        self.num_batches = int(num_batches)
        mask = torch.as_tensor(dense_mask, dtype=torch.bool).cpu()
        strata = torch.as_tensor(dense_strata, dtype=torch.int64).cpu()
        self.global_pool = torch.nonzero(~mask, as_tuple=False).flatten()
        local_ids = torch.unique(strata[mask], sorted=True)
        self.local_pools = [
            torch.nonzero(mask & (strata == stratum), as_tuple=False).flatten()
            for stratum in local_ids
        ]
        self.local_pools = [pool for pool in self.local_pools if pool.numel()]
        if not self.global_pool.numel() or not self.local_pools:
            raise ValueError("dense-transition sampler requires global and local data")
        self.local_count = int(round(self.batch_size * float(local_fraction)))
        self.global_count = self.batch_size - self.local_count

    @staticmethod
    def _draw(pool: torch.Tensor, count: int) -> torch.Tensor:
        return pool[torch.randint(0, pool.numel(), (int(count),))]

    def __iter__(self):
        stratum_counts = np.asarray(
            [
                self.local_count // len(self.local_pools)
                + int(index < self.local_count % len(self.local_pools))
                for index in range(len(self.local_pools))
            ],
            dtype=np.int64,
        )
        for _ in range(self.num_batches):
            chunks = [self._draw(self.global_pool, self.global_count)]
            chunks.extend(
                self._draw(pool, int(count))
                for pool, count in zip(self.local_pools, stratum_counts)
                if int(count) > 0
            )
            batch = torch.cat(chunks)
            yield batch[torch.randperm(batch.numel())]

    def __len__(self) -> int:
        return self.num_batches


def seed_worker(_):
    worker_seed = torch.utils.data.get_worker_info().seed % (2 ** 32)
    np.random.seed(worker_seed)


from . import d4rl  # register
