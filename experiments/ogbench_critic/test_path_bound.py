"""Original sampler stream, compact ends, one-step degeneration and frozen Adam."""
import unittest

from common import QRLAgent,Dataset,GCDataset,tree_equal
from critic import init_optimizer
from path_bound import PathBounds,sample_with_indices,requested_horizons,loss,update
from train import freeze_proof
import jax
import jax.numpy as jnp
import numpy as np


class PathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from agents.qrl import get_config
        cfg=get_config().to_dict();cfg.update(actor_hidden_dims=(16,16),value_hidden_dims=(16,16),latent_dim=16)
        cls.agent=QRLAgent.create(0,np.zeros((1,2),np.float32),np.zeros((1,2),np.float32),cfg)
        ob=np.column_stack([np.arange(7,dtype=np.float32),np.zeros(7,np.float32)])
        cls.dataset=GCDataset(Dataset.create(observations=ob,actions=np.zeros_like(ob),
            valids=np.array([1,1,1,0,1,1,0],np.float32),terminals=np.array([0,0,1,1,0,1,1],np.float32)),cfg)
        rng=np.random.RandomState(13)
        cls.batch={k:rng.normal(size=(16,2)).astype(np.float32) for k in ['observations','next_observations','value_goals','actor_goals','actions']}

    def test_sampler_batch_and_subsequent_rng_unchanged(self):
        np.random.seed(2001);expected=self.dataset.sample(16);after=np.random.get_state()
        np.random.seed(2001);actual,idxs=sample_with_indices(self.dataset,16);state=np.random.get_state()
        self.assertTrue(tree_equal(expected,actual));self.assertTrue(tree_equal(after,state))
        np.testing.assert_array_equal(actual['observations'],self.dataset.get_observations(idxs))

    def test_compact_double_ends_zero_length_and_valid_span(self):
        bounds=PathBounds(self.dataset);idx=np.array([0,1,2,4,5])
        observations,cost,ends=bounds.endpoints(idx,np.full(5,64))
        np.testing.assert_array_equal(ends,[2,2,2,5,5]);np.testing.assert_array_equal(cost,[2,1,0,1,0])
        np.testing.assert_array_equal(observations,self.dataset.get_observations(ends))
        for source,end in zip(idx,ends):self.assertTrue((self.dataset.dataset['valids'][source:end]>0).all())
        with self.assertRaises(AssertionError):bounds.endpoints(np.array([3]),np.array([4]))
        np.testing.assert_array_equal(np.sort(requested_horizons(1,6)),[2,4,8,16,32,64])

    def test_one_step_loss_and_gradient_degenerate_to_official(self):
        a=self.agent;b=self.batch;p=a.network.params
        value,info=loss(a,b,p,b['next_observations'],np.ones(16,np.float32))
        np.testing.assert_array_equal(value,a.value_loss(b,p)[0])
        actual=jax.grad(lambda q:loss(a,b,q,b['next_observations'],np.ones(16,np.float32))[0])(p)
        expected=jax.grad(lambda q:a.value_loss(b,q)[0])(p)
        for x,y in zip(jax.tree_util.tree_leaves(actual),jax.tree_util.tree_leaves(expected)):
            np.testing.assert_allclose(x,y,rtol=1e-6,atol=2e-6)
        # Separate compiled reductions can differ by a float32 ULP.
        self.assertGreaterEqual(float(info['d_pos_loss'])+2e-6,float(info['pair_loss']))

    def test_frozen_nonzero_moments_and_actual_count(self):
        a=self.agent.replace(network=self.agent.network.apply_gradients(jax.tree_util.tree_map(jnp.ones_like,self.agent.network.params)))
        trained,_,info=update(a,init_optimizer(a),self.batch,self.batch['value_goals'],np.ones(16,np.float32)*4)
        self.assertEqual(freeze_proof(a),freeze_proof(trained))
        self.assertEqual(int(trained.network.opt_state[0].count),int(a.network.opt_state[0].count)+1)
        self.assertGreaterEqual(float(info['d_pos_loss'])+2e-6,float(info['pair_loss']))


if __name__=='__main__':unittest.main()
