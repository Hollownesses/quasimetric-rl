from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from minimal_qrl.envs import CommInspectionDubinsUAV2D


@dataclass
class RawGoalSetEpisode:
    states: np.ndarray
    actions: np.ndarray
    device_id: str
    collisions: np.ndarray
    out_of_bounds: np.ndarray
    truncated: np.ndarray
    source: str = "policy"
    feasible_tasks: Optional[List[tuple[str, ...]]] = None

    def __post_init__(self) -> None:
        self.states = np.asarray(self.states, dtype=np.float32)
        self.actions = np.asarray(self.actions, dtype=np.float32)
        self.collisions = np.asarray(self.collisions, dtype=bool)
        self.out_of_bounds = np.asarray(self.out_of_bounds, dtype=bool)
        self.truncated = np.asarray(self.truncated, dtype=bool)
        if self.states.ndim != 2 or self.states.shape[1] != 3:
            raise ValueError("states must have shape [T + 1, 3]")
        if self.actions.ndim == 1:
            self.actions = self.actions[:, None]
        transition_count = len(self.states) - 1
        if len(self.actions) != transition_count:
            raise ValueError("actions must contain one entry per state transition")
        for name, value in (
            ("collisions", self.collisions),
            ("out_of_bounds", self.out_of_bounds),
            ("truncated", self.truncated),
        ):
            if len(value) != transition_count:
                raise ValueError(f"{name} must contain one entry per transition")

    @property
    def length(self) -> int:
        return int(len(self.actions))


class ContextHERReplayBuffer:
    """Episode replay with on-sample catalog-task hindsight relabeling."""

    def __init__(
        self,
        env: CommInspectionDubinsUAV2D,
        capacity: int,
        device: torch.device,
        *,
        her_k: int = 4,
        seed: int = 0,
    ) -> None:
        self.env = env
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.her_k = max(0, int(her_k))
        self.rng = np.random.default_rng(int(seed))
        self.episodes: List[RawGoalSetEpisode] = []
        self._cumulative = np.zeros((0,), dtype=np.int64)
        self.size = 0
        self.total_added = 0
        self.policy_transitions = 0
        self.teacher_transitions = 0
        self.sampled_transitions = 0
        self.relabel_count = 0
        self.cross_context_relabel_count = 0
        self.eligible_future_count = 0
        self.positive_relabel_count = 0
        self.relabel_by_device: Dict[str, int] = {}

    def _compute_feasible_tasks(self, states: np.ndarray) -> List[tuple[str, ...]]:
        result: List[tuple[str, ...]] = []
        for state in states:
            result.append(tuple(
                device_id
                for device_id in self.env.device_ids
                if self.env.is_task_feasible_for_task(state, device_id)
            ))
        return result

    def add_episode(self, episode: RawGoalSetEpisode) -> None:
        if episode.length <= 0:
            return
        if episode.feasible_tasks is None:
            episode.feasible_tasks = self._compute_feasible_tasks(episode.states)
        if len(episode.feasible_tasks) != len(episode.states):
            raise ValueError("feasible_tasks must align with states")
        self.episodes.append(episode)
        self.size += episode.length
        self.total_added += episode.length
        if episode.source == "teacher":
            self.teacher_transitions += episode.length
        else:
            self.policy_transitions += episode.length
        while self.size > self.capacity and len(self.episodes) > 1:
            removed = self.episodes.pop(0)
            self.size -= removed.length
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        self._cumulative = np.cumsum(
            np.asarray([episode.length for episode in self.episodes], dtype=np.int64)
        )

    def _locate(self, flat_index: int) -> tuple[RawGoalSetEpisode, int]:
        episode_index = int(np.searchsorted(self._cumulative, int(flat_index), side="right"))
        previous = 0 if episode_index == 0 else int(self._cumulative[episode_index - 1])
        return self.episodes[episode_index], int(flat_index) - previous

    def _future_tasks(self, episode: RawGoalSetEpisode, t: int) -> List[str]:
        assert episode.feasible_tasks is not None
        future_indices = np.arange(t + 1, episode.length + 1, dtype=np.int64)
        self.rng.shuffle(future_indices)
        selected: List[str] = []
        for future_index in future_indices:
            tasks = list(episode.feasible_tasks[int(future_index)])
            if not tasks:
                continue
            self.rng.shuffle(tasks)
            for task in tasks:
                task = str(task)
                if task not in selected:
                    selected.append(task)
                    if len(selected) >= max(1, self.her_k):
                        return selected
        return selected

    def _encode(
        self,
        episode: RawGoalSetEpisode,
        t: int,
        device_id: str,
    ) -> Dict[str, object]:
        state = episode.states[t]
        next_state = episode.states[t + 1]
        outcome = self.env.transition_outcome_for_task(
            next_state,
            device_id,
            collision=bool(episode.collisions[t]),
            out_of_bounds=bool(episode.out_of_bounds[t]),
        )
        return {
            "obs": self.env.observation_for_task(state, device_id),
            "goal": np.asarray(outcome["goal_observation"], dtype=np.float32),
            "action": episode.actions[t],
            "reward": float(outcome["reward"]),
            "next_obs": np.asarray(outcome["next_observation"], dtype=np.float32),
            "done": float(bool(outcome["terminated"])),
            "device_id": str(device_id),
            "success": float(bool(outcome["success"])),
            "source": episode.source,
        }

    def sample_numpy(
        self,
        batch_size: int,
        *,
        positive_only: bool = False,
    ) -> Dict[str, object]:
        if self.size <= 0:
            raise ValueError("cannot sample an empty replay buffer")
        rows: List[Dict[str, object]] = []
        max_attempts = max(100, int(batch_size) * 50)
        attempts = 0
        while len(rows) < int(batch_size) and attempts < max_attempts:
            attempts += 1
            flat_index = int(self.rng.integers(0, self.size))
            episode, t = self._locate(flat_index)
            future_tasks = self._future_tasks(episode, t) if self.her_k > 0 else []
            if positive_only:
                if not future_tasks:
                    continue
                targets = [(task, True) for task in future_tasks]
            else:
                targets = [(episode.device_id, False)]
                targets.extend((task, True) for task in future_tasks)
            for target_task, hindsight in targets:
                if len(rows) >= int(batch_size):
                    break
                row = self._encode(episode, t, target_task)
                row["relabeled"] = float(hindsight)
                rows.append(row)
                self.sampled_transitions += 1
                if hindsight:
                    self.relabel_count += 1
                    self.eligible_future_count += 1
                    self.positive_relabel_count += 1
                    if target_task != episode.device_id:
                        self.cross_context_relabel_count += 1
                    self.relabel_by_device[target_task] = self.relabel_by_device.get(target_task, 0) + 1
        if len(rows) < int(batch_size):
            raise RuntimeError(
                f"only found {len(rows)} positive goal-set samples after {attempts} attempts"
            )
        array_keys = ("obs", "goal", "action", "reward", "next_obs", "done", "success", "relabeled")
        batch: Dict[str, object] = {}
        for key in array_keys:
            batch[key] = np.asarray([row[key] for row in rows], dtype=np.float32)
        batch["device_ids"] = [str(row["device_id"]) for row in rows]
        batch["sources"] = [str(row["source"]) for row in rows]
        return batch

    def sample(self, batch_size: int, *, positive_only: bool = False) -> Dict[str, Tensor | List[str]]:
        numpy_batch = self.sample_numpy(batch_size, positive_only=positive_only)
        result: Dict[str, Tensor | List[str]] = {}
        for key, value in numpy_batch.items():
            if key in {"device_ids", "sources"}:
                result[key] = value  # type: ignore[assignment]
            else:
                result[key] = torch.as_tensor(value, device=self.device, dtype=torch.float32)
        return result

    def diagnostics(self) -> Dict[str, float]:
        sampled = max(1, self.sampled_transitions)
        diagnostics = {
            "replay_size": float(self.size),
            "policy_transitions": float(self.policy_transitions),
            "teacher_transitions": float(self.teacher_transitions),
            "sampled_transitions": float(self.sampled_transitions),
            "relabel_count": float(self.relabel_count),
            "cross_context_relabel_count": float(self.cross_context_relabel_count),
            "eligible_future_ratio": float(self.eligible_future_count / sampled),
            "positive_relabel_ratio": float(self.positive_relabel_count / sampled),
        }
        diagnostics.update({
            f"relabel_device::{device_id}": float(count)
            for device_id, count in sorted(self.relabel_by_device.items())
        })
        return diagnostics


def episode_from_observations(
    env: CommInspectionDubinsUAV2D,
    observations: Sequence[np.ndarray],
    actions: Sequence[np.ndarray],
    device_id: str,
    *,
    collisions: Optional[Sequence[bool]] = None,
    out_of_bounds: Optional[Sequence[bool]] = None,
    truncated: Optional[Sequence[bool]] = None,
    source: str = "policy",
) -> RawGoalSetEpisode:
    states = np.stack([env.observation_to_state(obs) for obs in observations], axis=0)
    n = len(actions)
    return RawGoalSetEpisode(
        states=states,
        actions=np.asarray(actions, dtype=np.float32),
        device_id=str(device_id),
        collisions=np.zeros((n,), dtype=bool) if collisions is None else np.asarray(collisions, dtype=bool),
        out_of_bounds=np.zeros((n,), dtype=bool) if out_of_bounds is None else np.asarray(out_of_bounds, dtype=bool),
        truncated=np.zeros((n,), dtype=bool) if truncated is None else np.asarray(truncated, dtype=bool),
        source=str(source),
    )
