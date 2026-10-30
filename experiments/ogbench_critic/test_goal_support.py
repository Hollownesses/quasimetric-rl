"""Support population balance, source identity, deterministic no-RNG queries."""
import unittest
import numpy as np
from goal_support import GoalSupport
from query_augmentation import QueryAugmentation


class GoalSupportTests(unittest.TestCase):
    def setUp(self):
        self.nodes=np.arange(14,dtype=np.float32).reshape(7,2);self.sampler=GoalSupport(self.nodes)
        self.batch={k:np.arange(32,dtype=np.float32).reshape(16,2)+i for i,k in enumerate(['observations','next_observations','value_goals'])}

    def test_uniform_cycle_and_actual_population_counts(self):
        np.testing.assert_array_equal(np.sort(self.sampler.indices(0,7)),np.arange(7))
        np.testing.assert_array_equal(self.sampler.indices(7,7),self.sampler.indices(0,7))
        sampler=GoalSupport(np.zeros((6302,2),np.float32));counts=np.bincount(sampler.indices(0,128000),minlength=6302)
        self.assertEqual(int(counts.sum()),128000);self.assertEqual(int(counts.min()),20);self.assertEqual(int(counts.max()),21)

    def test_source_original_goals_inputs_unchanged(self):
        b=self.batch;before={k:v.copy() for k,v in b.items()};new=self.sampler(b,0);old=QueryAugmentation()(b,0)
        np.testing.assert_array_equal(new['observations'],old['observations']);np.testing.assert_array_equal(new['next_observations'],old['next_observations'])
        np.testing.assert_array_equal(new['value_goals'][::2],b['value_goals'][::2])
        np.testing.assert_array_equal(new['value_goals'][1::2],self.nodes[self.sampler.indices(0,8)]+self.sampler.offsets[:8,2:])
        for k in b:np.testing.assert_array_equal(b[k],before[k])

    def test_chunk_identity_and_numpy_rng_preserved(self):
        state=np.random.get_state();full=self.sampler(self.batch,128000);after=np.random.get_state()
        self.assertEqual(state[0],after[0]);np.testing.assert_array_equal(state[1],after[1]);self.assertEqual(state[2:],after[2:])
        pieces=[self.sampler({k:v[:8] for k,v in self.batch.items()},128000),self.sampler({k:v[8:] for k,v in self.batch.items()},128004)]
        for k in full:np.testing.assert_array_equal(full[k],np.concatenate([p[k] for p in pieces]))


if __name__=='__main__':unittest.main()
