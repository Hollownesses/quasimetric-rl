"""Weighting identity, cell probability mass and safe critic freezing."""
import tempfile
import unittest
from pathlib import Path

from common import QRLAgent, tree_equal
from critic import init_optimizer
from source_balance import SourceCells, loss, update
from train import freeze_proof
import jax
import jax.numpy as jnp
import numpy as np


class SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from agents.qrl import get_config
        cfg = get_config().to_dict(); cfg.update(actor_hidden_dims=(16,16), value_hidden_dims=(16,16), latent_dim=16)
        cls.agent = QRLAgent.create(0, np.zeros((1,2),np.float32), np.zeros((1,2),np.float32), cfg)
        rng = np.random.RandomState(13)
        cls.batch = {k: rng.normal(size=(16,2)).astype(np.float32) for k in
            ['observations','next_observations','value_goals','actor_goals','actions']}

    def test_ones_are_official_loss_and_gradient(self):
        agent = self.agent; batch = self.batch; params = agent.network.params; weights = jnp.ones(16)
        self.assertTrue(tree_equal(loss(agent,batch,params,weights)[0],agent.value_loss(batch,params)[0]))
        a = jax.grad(lambda p: loss(agent,batch,p,weights)[0])(params)
        b = jax.grad(lambda p: agent.value_loss(batch,p)[0])(params)
        self.assertTrue(tree_equal(a,b))

    def test_population_weights_and_full_observation_distinction(self):
        observations = np.array([[0,0],[.01,.01],[.02,.02],[0,1]],np.float32)
        cells = SourceCells.build(observations); w = cells.weights(observations)
        self.assertAlmostEqual(float(w.mean()),1.)
        self.assertAlmostEqual(float(w[:3].sum()/4),.5)
        self.assertAlmostEqual(float(w[3]/4),.5)
        self.assertEqual(cells.occupied,2)  # Both observation dimensions participate.
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'cells.npz';cells.save(path)
            np.testing.assert_array_equal(w,SourceCells.load(path).weights(observations))

    def test_frozen_nonzero_moments_and_actual_count(self):
        agent = self.agent.replace(network=self.agent.network.apply_gradients(jax.tree_util.tree_map(jnp.ones_like,self.agent.network.params)))
        trained,_,_ = update(agent,init_optimizer(agent),self.batch,jnp.linspace(.1,2.,16))
        self.assertEqual(freeze_proof(agent),freeze_proof(trained))
        self.assertEqual(int(trained.network.opt_state[0].count),int(agent.network.opt_state[0].count)+1)
        self.assertFalse(tree_equal(agent.network.params['modules_value'],trained.network.params['modules_value']))


if __name__ == '__main__': unittest.main()
