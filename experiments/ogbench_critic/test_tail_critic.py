"""Official-loss identity, tail selection and optimizer-safe freezing."""
import unittest
import jax
import jax.numpy as jnp
import numpy as np

from common import QRLAgent, tree_equal
from critic import init_optimizer
from tail_critic import loss, update, upper_tail_mean
from train import freeze_proof


class TailTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from agents.qrl import get_config
        config = get_config().to_dict()
        config.update(actor_hidden_dims=(16, 16), value_hidden_dims=(16, 16), latent_dim=16)
        cls.agent = QRLAgent.create(0, np.zeros((1, 2), np.float32), np.zeros((1, 2), np.float32), config)
        rng = np.random.RandomState(13)
        cls.batch = {k: rng.normal(size=(16, 2)).astype(np.float32) for k in
            ['observations', 'next_observations', 'value_goals', 'actor_goals', 'actions']}

    def test_mean_and_full_tail_equal_official_value_and_gradient(self):
        p = self.agent.network.params
        reference = self.agent.value_loss(self.batch, p)[0]
        gradient = jax.grad(lambda q: self.agent.value_loss(self.batch, q)[0])(p)
        for mode, fraction in [('mean', .1), ('tail', 1.)]:
            actual = loss(self.agent, self.batch, p, mode, fraction)[0]
            grad = jax.grad(lambda q: loss(self.agent, self.batch, q, mode, fraction)[0])(p)
            self.assertTrue(tree_equal(actual, reference))
            self.assertTrue(tree_equal(grad, gradient))

    def test_tail_is_worst_ceil_fraction_with_correct_gradient(self):
        values = jnp.arange(16, dtype=jnp.float32)
        self.assertEqual(float(upper_tail_mean(values)), 14.5)
        grad = jax.grad(upper_tail_mean)(values)
        np.testing.assert_array_equal(grad, [0.] * 14 + [.5, .5])
        self.assertGreaterEqual(float(upper_tail_mean(values)), float(values.mean()))

    def test_frozen_nonzero_moments_and_actual_count(self):
        agent = self.agent.replace(network=self.agent.network.apply_gradients(
            jax.tree_util.tree_map(jnp.ones_like, self.agent.network.params)))
        for mode in ['mean', 'tail']:
            updated, _, _ = update(agent, init_optimizer(agent), self.batch, mode)
            self.assertEqual(freeze_proof(updated), freeze_proof(agent))
            self.assertEqual(int(updated.network.opt_state[0].count), int(agent.network.opt_state[0].count) + 1)
            self.assertFalse(tree_equal(agent.network.params['modules_value'], updated.network.params['modules_value']))


if __name__ == '__main__': unittest.main()
