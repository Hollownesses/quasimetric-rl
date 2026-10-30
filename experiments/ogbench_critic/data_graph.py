"""Training-transition-only coarse graph critic; no simulator/geometry imports."""
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

SPACING = .2


def valid_transitions(dataset):
    """Delegate successor and valid-boundary semantics to the official Dataset."""
    indices = np.asarray(dataset.valid_idxs)
    assert len(indices) and len(np.unique(indices)) == len(indices)
    assert np.all(dataset['valids'][indices] > 0)
    if 'next_observations' not in dataset:
        assert indices.max() < dataset.size - 1
        # Compact data marks the final state invalid; a penultimate transition
        # may itself carry a terminal flag, so never filter terminal flags here.
        ends = np.flatnonzero(dataset['valids'] <= 0)
        assert not np.isin(indices, ends).any()
    subset = dataset.get_subset(indices)
    observations, successors = subset['observations'], subset['next_observations']
    assert observations.shape == successors.shape == (len(indices), 2)
    assert np.isfinite(observations).all() and np.isfinite(successors).all()
    return observations, successors


@dataclass
class DataGraph:
    nodes: np.ndarray
    source: np.ndarray
    target: np.ndarray
    spacing: float = SPACING
    _cache: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        n = len(self.nodes)
        assert n > 0 and self.nodes.shape == (n, 2)
        assert self.source.shape == self.target.shape
        assert self.source.min() >= 0 and self.source.max() < n
        assert self.target.min() >= 0 and self.target.max() < n
        self.graph = csr_matrix((np.ones(len(self.source), dtype=np.float32),
                                 (self.source, self.target)), shape=(n, n))
        assert self.graph.nnz == len(self.source) and np.all(self.graph.data == 1)
        self.reverse = self.graph.T.tocsr()
        self.tree = cKDTree(self.nodes)
        grid = np.rint(self.nodes / self.spacing).astype(np.int64)
        self.lower = grid.min(axis=0)
        self.lookup = np.full(grid.max(axis=0) - self.lower + 1, -1, dtype=np.int32)
        self.lookup[tuple((grid - self.lower).T)] = np.arange(n)

    @classmethod
    def build(cls, observations, successors, spacing=SPACING):
        assert observations.shape == successors.shape and observations.shape[1] == 2
        quantized = np.rint(np.concatenate([observations, successors]) / spacing).astype(np.int32)
        grid, ids = np.unique(quantized, axis=0, return_inverse=True)
        count = len(observations)
        edges = np.unique(np.column_stack([ids[:count], ids[count:]]), axis=0)
        return cls(grid.astype(np.float64) * spacing, edges[:, 0], edges[:, 1], spacing)

    def indices(self, observations):
        observations = np.asarray(observations)
        assert observations.ndim == 2 and observations.shape[1] == 2
        grid = np.rint(observations / self.spacing).astype(np.int64) - self.lower
        inside = ((grid >= 0) & (grid < self.lookup.shape)).all(axis=1)
        indices = np.full(len(observations), -1, dtype=np.int64)
        indices[inside] = self.lookup[tuple(grid[inside].T)]
        missing = indices < 0
        if missing.any(): indices[missing] = self.tree.query(observations[missing])[1]
        return indices

    def goal_distances(self, goal):
        goal = np.asarray(goal)
        assert goal.shape == (2,)
        index = int(self.indices(goal[None])[0])
        if index not in self._cache:
            distance = dijkstra(self.reverse, directed=True, indices=index, unweighted=True)
            distance = np.minimum(distance, len(self.nodes))
            assert np.isfinite(distance).all() and distance[index] == 0
            assert np.all(distance[self.source] <= 1 + distance[self.target])
            self._cache[index] = distance
        return self._cache[index]

    def distances(self, observations, goal):
        return self.goal_distances(goal)[self.indices(observations)]

    def save(self, path):
        np.savez_compressed(path, nodes=self.nodes, source=self.source, target=self.target,
                            spacing=np.asarray(self.spacing))

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            return cls(data['nodes'].copy(), data['source'].copy(), data['target'].copy(), float(data['spacing']))
