"""Verify complete official loss decomposition, including dynamics to value."""
import unittest

from common import QRLAgent
from critic import subset
from full_gradient import gradient,difference,assert_gradient_equal
import jax
import numpy as np


class FullGradientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from agents.qrl import get_config
        cfg=get_config().to_dict();cfg.update(actor_hidden_dims=(16,16),value_hidden_dims=(16,16),latent_dim=16)
        cls.agent=QRLAgent.create(0,np.zeros((1,2),np.float32),np.zeros((1,2),np.float32),cfg)
        rng=np.random.RandomState(89)
        cls.batch={k:rng.normal(size=(8,2)).astype(np.float32) for k in
                   ['observations','next_observations','value_goals','actor_goals','actions']}

    def test_sum_matches_native_full_parameter_value_gradient(self):
        a=self.agent;p=subset(a.network.params);b=self.batch
        parts=[gradient(a,p,b,k) for k in ['value','dynamics','actor']]
        expected=jax.tree_util.tree_map(lambda x,y,z:x+y+z,*parts)
        actual=jax.grad(lambda x:a.total_loss(b,x)[0])(a.network.params)
        self.assertLess(difference(expected['modules_value'],actual['modules_value']),5e-5)
        self.assertLess(difference(expected,gradient(a,p,b,'total')),5e-5)
        assert_gradient_equal(expected,gradient(a,p,b,'total'))

    def test_numeric_comparison_rejects_material_gradient_changes(self):
        with self.assertRaises(AssertionError):
            assert_gradient_equal({'x':np.array([1.,0.])},{'x':np.array([1.01,0.])})

    def test_dynamics_reaches_value_but_actor_does_not(self):
        a=self.agent;p=subset(a.network.params)
        dyn=gradient(a,p,self.batch,'dynamics')['modules_value']
        actor=gradient(a,p,self.batch,'actor')['modules_value']
        self.assertTrue(any(np.any(np.asarray(x)!=0) for x in jax.tree_util.tree_leaves(dyn)))
        self.assertTrue(all(np.all(np.asarray(x)==0) for x in jax.tree_util.tree_leaves(actor)))


if __name__=='__main__':unittest.main()
