"""Exact float32-key semantics, first-hit audit, output ownership and eviction."""
import unittest
from unittest.mock import patch

from common import QRLAgent, distances
from exact_score_cache import ExactScoreCache
import numpy as np


class CacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from agents.qrl import get_config
        cfg = get_config().to_dict(); cfg.update(actor_hidden_dims=(16,16), value_hidden_dims=(16,16), latent_dim=16)
        cls.agent = QRLAgent.create(0, np.zeros((1,2),np.float32), np.zeros((1,2),np.float32), cfg)

    def test_effective_float32_hit_and_audit(self):
        ob = np.array([[1., 2.], [3., 4.]], np.float64); goal = ob + 10
        cache = ExactScoreCache(self.agent)
        with patch('exact_score_cache.distances', wraps=distances) as scorer:
            first = cache(ob, goal); near = ob + 1e-10
            np.testing.assert_array_equal(ob.astype(np.float32), near.astype(np.float32))
            np.testing.assert_array_equal(first, cache(near, goal))
            np.testing.assert_array_equal(first, cache(ob, goal))
            self.assertEqual(scorer.call_count, 2)
            self.assertEqual(cache.statistics()['first_hit_audits'], 1)
            changed = ob.copy(); changed[0,0] += .01
            cache(changed, goal); self.assertEqual(scorer.call_count, 3)

    def test_eviction_shape_goal_and_returned_array_are_independent(self):
        ob = np.ones((2,2), np.float32); goal = ob + 2; cache = ExactScoreCache(self.agent, capacity=1)
        value = cache(ob, goal); expected = value.copy(); value[:] = -999
        np.testing.assert_array_equal(cache(ob, goal), expected)
        cache(ob, goal + .1); self.assertEqual(cache.statistics()['entries'], 1)
        cache(ob, goal); self.assertEqual(cache.statistics()['hits'], 1)
        cache(ob[:1], goal[:1]); self.assertEqual(cache.statistics()['entries'], 1)


if __name__ == '__main__': unittest.main()
