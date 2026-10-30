"""Sampler invariants, effective offsets and sequence identity."""
import unittest
import numpy as np
from query_augmentation import QueryAugmentation


class QueryTests(unittest.TestCase):
    def setUp(self):
        self.sampler=QueryAugmentation()
        self.batch={'observations':np.arange(32,dtype=np.float32).reshape(16,2),
                    'value_goals':np.arange(32,64,dtype=np.float32).reshape(16,2),
                    'next_observations':np.zeros((16,2),np.float32)}

    def test_preserve_original_rows_inputs_and_full_coordinates(self):
        before={k:v.copy() for k,v in self.batch.items()};aug=self.sampler(self.batch,0)
        for k in before:np.testing.assert_array_equal(self.batch[k],before[k])
        for k,cols in [('observations',slice(0,2)),('value_goals',slice(2,4))]:
            np.testing.assert_array_equal(aug[k][::2],before[k][::2])
            np.testing.assert_array_equal(aug[k][1::2],before[k][1::2]+self.sampler.offsets[:8,cols])
            self.assertTrue(np.all(np.abs(aug[k]-before[k])<=1))
            self.assertTrue((np.ptp(aug[k][1::2]-before[k][1::2],axis=0)>0).all())
        np.testing.assert_array_equal(aug['next_observations'],before['next_observations'])

    def test_chunk_identity_no_numpy_rng_effect(self):
        state=np.random.get_state();full=self.sampler(self.batch,128000)
        after=np.random.get_state()
        self.assertEqual(state[0],after[0]);np.testing.assert_array_equal(state[1],after[1]);self.assertEqual(state[2:],after[2:])
        pieces=[self.sampler({k:v[:8] for k,v in self.batch.items()},128000),
                self.sampler({k:v[8:] for k,v in self.batch.items()},128004)]
        for k in full:np.testing.assert_array_equal(full[k],np.concatenate([x[k] for x in pieces]))
        for k in full:np.testing.assert_array_equal(full[k],QueryAugmentation()(self.batch,128000)[k])


if __name__=='__main__':unittest.main()
