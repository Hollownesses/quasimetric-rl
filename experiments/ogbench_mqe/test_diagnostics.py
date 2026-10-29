"""Check native V inference, fixed max aggregation and endpoint safeguards."""
from pathlib import Path
import tempfile
import unittest

from diagnostics import MQEAgent,value_components,aggregate,run
import flax
import numpy as np


class NativeDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from agents.mqe import get_config
        cfg=get_config().to_dict();cfg.update(actor_hidden_dims=(16,16),value_hidden_dims=(16,16),latent_dim=16,batch_size=8)
        cls.agent=MQEAgent.create(7,np.zeros((1,2),np.float32),np.zeros((1,2),np.float32),cfg)

    def test_state_value_matches_native_action_free_distance(self):
        rng=np.random.RandomState(81);obs=rng.normal(size=(8,2)).astype(np.float32);goals=rng.normal(size=(8,2)).astype(np.float32)
        view=self.agent.replace(config=flax.core.FrozenDict({**self.agent.config,'use_action_for_distance':False}))
        expected=np.asarray(view.get_distance(obs,goals,np.zeros_like(obs)))
        actual=np.asarray(value_components(self.agent,obs,goals))
        self.assertEqual(actual.shape,(2,8));np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-6)

    def test_pessimistic_aggregation_is_fixed_and_rejects_nan(self):
        values=np.array([[1.,4.,3.],[2.,1.,5.]])
        np.testing.assert_array_equal(aggregate(values,'max'),[2.,4.,5.])
        np.testing.assert_array_equal(aggregate(values,'ensemble0'),values[0])
        np.testing.assert_array_equal(aggregate(values,'ensemble1'),values[1])
        with self.assertRaises(AssertionError):aggregate(np.array([[np.nan],[1.]]),'max')
        with self.assertRaises(ValueError):aggregate(values,'best_looking_critic')

    def test_intermediate_epoch_cannot_be_formal_endpoint(self):
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)/'not_created'
            for epoch,smoke in [(10000,False),(2,False),(100000,True)]:
                with self.assertRaises(AssertionError):run('unused',epoch,'unused',output,smoke)
            self.assertFalse(output.exists())


if __name__=='__main__':unittest.main()
