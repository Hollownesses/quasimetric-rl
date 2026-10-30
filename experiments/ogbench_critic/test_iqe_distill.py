"""Paired MSE definition, gradient isolation and nonzero-momentum freezing."""
import unittest
import jax
import jax.numpy as jnp
import numpy as np

from common import QRLAgent,tree_equal
from iqe_distill import loss,update,init_optimizer,frozen_proof


class DistillTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from agents.qrl import get_config
        cfg=get_config().to_dict();cfg.update(actor_hidden_dims=(16,16),value_hidden_dims=(16,16),latent_dim=16)
        cls.agent=QRLAgent.create(0,np.zeros((1,2),np.float32),np.zeros((1,2),np.float32),cfg)
        rng=np.random.RandomState(13)
        cls.batch={k:rng.normal(size=(16,2)).astype(np.float32) for k in ['observations','value_goals']}
        cls.targets=np.linspace(1,4,16,dtype=np.float32)

    def test_paired_original_function_loss(self):
        a=self.agent;b=self.batch;f=a.network.select('value')(b['observations'],b['value_goals'])
        r=a.network.select('value')(b['value_goals'],b['observations'])
        expected=.5*(jnp.square(f-self.targets)+jnp.square(r-self.targets)).mean()
        np.testing.assert_array_equal(loss(a,b,a.network.params,self.targets)[0],expected)

    def test_no_actor_dynamics_or_dual_gradient(self):
        a=self.agent;grads=jax.grad(lambda p:loss(a,self.batch,p,self.targets)[0])(a.network.params)
        for key in ['modules_actor','modules_dynamics','modules_lam']:
            self.assertTrue(all(np.count_nonzero(x)==0 for x in jax.tree_util.tree_leaves(grads[key])))
        self.assertTrue(any(np.count_nonzero(x)>0 for x in jax.tree_util.tree_leaves(grads['modules_value'])))

    def test_optimizer_keeps_all_frozen_nonzero_moments(self):
        a=self.agent.replace(network=self.agent.network.apply_gradients(jax.tree_util.tree_map(jnp.ones_like,self.agent.network.params)))
        trained,_,_=update(a,init_optimizer(a),self.batch,self.targets)
        self.assertEqual(frozen_proof(a),frozen_proof(trained))
        self.assertEqual(int(trained.network.opt_state[0].count),int(a.network.opt_state[0].count)+1)
        self.assertFalse(tree_equal(a.network.params['modules_value'],trained.network.params['modules_value']))


if __name__=='__main__':unittest.main()
