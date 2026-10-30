"""Signed shape definition, zero successor degeneration and freezing."""
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from common import QRLAgent,tree_equal
from iqe_distill import loss as absolute_loss
from local_shape import loss,update,init_optimizer,frozen_proof
from shape_queries import ShapeQueries
from query_augmentation import QueryAugmentation


class ShapeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from agents.qrl import get_config
        config=get_config().to_dict();config.update(actor_hidden_dims=(16,16),value_hidden_dims=(16,16),latent_dim=16)
        cls.agent=QRLAgent.create(0,np.zeros((1,2),np.float32),np.zeros((1,2),np.float32),config)
        rng=np.random.RandomState(13);cls.batch={k:rng.normal(size=(16,2)).astype(np.float32) for k in ['observations','next_observations','value_goals']}
        cls.targets=np.column_stack([np.linspace(1,4,16),np.linspace(4,1,16)]).astype(np.float32)

    def test_signed_formula_and_isolated_gradient(self):
        a=self.agent;b=self.batch;target=self.targets
        def reference(p):
            v=a.network.select('value');f=v(b['observations'],b['value_goals'],params=p);r=v(b['value_goals'],b['observations'],params=p)
            fn=v(b['next_observations'],b['value_goals'],params=p);rn=v(b['value_goals'],b['next_observations'],params=p)
            delta=target[:,0]-target[:,1]
            return absolute_loss(a,b,p,target[:,0])[0]+.5*(jnp.square((f-fn-delta)/.2)+jnp.square((r-rn-delta)/.2)).mean()
        self.assertTrue((target[:,0]-target[:,1]<0).any());self.assertTrue((target[:,0]-target[:,1]>0).any())
        np.testing.assert_array_equal(loss(a,b,a.network.params,target)[0],reference(a.network.params))
        actual=jax.grad(lambda p:loss(a,b,p,target)[0])(a.network.params);expected=jax.grad(reference)(a.network.params)
        # Shared encoder paths versus separate calls sum FP32 gradients in a different order.
        for x,y in zip(jax.tree_util.tree_leaves(actual),jax.tree_util.tree_leaves(expected)):np.testing.assert_allclose(x,y,rtol=2e-5,atol=3e-5)
        for name in ['modules_actor','modules_dynamics','modules_lam']:self.assertTrue(all(np.count_nonzero(x)==0 for x in jax.tree_util.tree_leaves(actual[name])))

    def test_same_successor_zero_shape(self):
        b={**self.batch,'next_observations':self.batch['observations']};t=np.column_stack([self.targets[:,0],self.targets[:,0]])
        value,info=loss(self.agent,b,self.agent.network.params,t)
        self.assertEqual(float(info['shape_loss']),0)
        np.testing.assert_array_equal(value,absolute_loss(self.agent,b,self.agent.network.params,t[:,0])[0])

    def test_nonzero_momentum_frozen_and_count(self):
        a=self.agent.replace(network=self.agent.network.apply_gradients(jax.tree_util.tree_map(jnp.ones_like,self.agent.network.params)))
        trained,_,_=update(a,init_optimizer(a),self.batch,self.targets)
        self.assertEqual(frozen_proof(a),frozen_proof(trained));self.assertEqual(int(trained.network.opt_state[0].count),int(a.network.opt_state[0].count)+1)
        self.assertFalse(tree_equal(a.network.params['modules_value'],trained.network.params['modules_value']))

    def test_current_queries_identical_and_next_translation(self):
        sampler=ShapeQueries();b=self.batch;before={k:v.copy() for k,v in b.items()};a=sampler(b,128000);old=QueryAugmentation()(b,128000)
        for k in ['observations','value_goals']:np.testing.assert_array_equal(a[k],old[k])
        np.testing.assert_array_equal(a['next_observations'][::2],b['next_observations'][::2])
        np.testing.assert_array_equal(a['next_observations'][1::2],b['next_observations'][1::2]+sampler.offsets[128000:128008,:2])
        for k in b:np.testing.assert_array_equal(b[k],before[k])


if __name__=='__main__':unittest.main()
