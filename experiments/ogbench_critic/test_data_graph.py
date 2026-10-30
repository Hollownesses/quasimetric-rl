"""Graph direction, repeated edges, unreachable distances and official compact ends."""
import tempfile
import unittest
from pathlib import Path

from common import Dataset
from data_graph import DataGraph, valid_transitions
import numpy as np


class DataGraphTests(unittest.TestCase):
    def test_compact_ends_use_valid_not_terminal_filter(self):
        observations = np.array([[0., 0.], [.2, 0.], [.4, 0.], [10., 0.], [10.2, 0.]])
        ds = Dataset.create(observations=observations, valids=np.array([1, 1, 0, 1, 0]),
                            terminals=np.array([0, 1, 1, 1, 1]))
        source, target = valid_transitions(ds)
        np.testing.assert_array_equal(source, observations[[0, 1, 3]])
        np.testing.assert_array_equal(target, observations[[1, 2, 4]])
        graph = DataGraph.build(source, target)
        assert graph.distances(observations[[0]], observations[2])[0] == 2
        assert graph.distances(observations[[0]], observations[3])[0] == len(graph.nodes)

    def test_unit_dedup_direction_triangle_and_save_load(self):
        a = np.array([[0., 0.], [0., 0.], [.2, 0.], [1., 0.]])
        b = np.array([[.2, 0.], [.2, 0.], [.4, 0.], [1., 0.]])
        graph = DataGraph.build(a, b)
        self.assertEqual(graph.graph.nnz, 3)
        self.assertTrue(np.all(graph.graph.data == 1))
        distance = np.stack([graph.distances(graph.nodes, goal) for goal in graph.nodes], axis=1)
        self.assertEqual(distance[0, 2], 2)
        self.assertEqual(distance[2, 0], len(graph.nodes))
        for middle in range(len(distance)):
            self.assertTrue(np.all(distance <= distance[:, middle, None] + distance[None, middle, :]))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'graph.npz'; graph.save(path); other = DataGraph.load(path)
            np.testing.assert_array_equal(graph.nodes, other.nodes)
            for goal in graph.nodes:
                np.testing.assert_array_equal(graph.distances(graph.nodes, goal), other.distances(other.nodes, goal))


if __name__ == '__main__':
    unittest.main()
