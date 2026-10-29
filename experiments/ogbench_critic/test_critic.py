"""Mechanism tests: gradients, exact A control and optimizer-safe freezing."""
import unittest
import jax
import jax.numpy as jnp
import numpy as np
import optax
from common import QRLAgent, tree_equal
from critic import critic_loss, diagnostics, init_optimizer, subset, update


class CriticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from agents.qrl import get_config
        config=get_config().to_dict()
        config.update(actor_hidden_dims=(16,16),value_hidden_dims=(16,16),latent_dim=16)
        cls.agent=QRLAgent.create(0,np.zeros((1,2),np.float32),np.zeros((1,2),np.float32),config)
        rng=np.random.RandomState(13)
        cls.batch={k:rng.normal(size=(8,2)).astype(np.float32) for k in
                   ['observations','next_observations','value_goals','actor_goals','actions']}

    def test_A_is_exact_official_value_loss_and_gradient(self):
        p=self.agent.network.params
        a,ia=critic_loss(self.agent,self.batch,p,'pair')
        b,ib=self.agent.value_loss(self.batch,p)
        self.assertTrue(tree_equal((a,ia),(b,ib)))
        ga=jax.grad(lambda q:critic_loss(self.agent,self.batch,q,'pair')[0])(p)
        gb=jax.grad(lambda q:self.agent.value_loss(self.batch,q)[0])(p)
        self.assertTrue(tree_equal(ga,gb))

    def test_freeze_with_nonzero_pretrained_momentum(self):
        # First make ALL Adam moments nonzero, exposing the zero-gradient trap.
        a=self.agent
        grads=jax.tree_util.tree_map(jnp.ones_like,a.network.params)
        a=a.replace(network=a.network.apply_gradients(grads))
        original=a
        opt=init_optimizer(a)
        for mode in ['pair','goal']:
            b,_,_=update(a,opt,self.batch,mode)
            for key in ['modules_actor','modules_dynamics']:
                self.assertTrue(tree_equal(original.network.params[key],b.network.params[key]))
                self.assertTrue(tree_equal(original.network.opt_state[0].mu[key],b.network.opt_state[0].mu[key]))
                self.assertTrue(tree_equal(original.network.opt_state[0].nu[key],b.network.opt_state[0].nu[key]))
            self.assertFalse(tree_equal(a.network.params['modules_value'],b.network.params['modules_value']))
            self.assertEqual(int(b.network.opt_state[0].count),int(a.network.opt_state[0].count)+1)

    def test_triangle_implies_goal_violation_no_larger_than_pair(self):
        info=diagnostics(self.agent,self.batch)
        self.assertLessEqual(float(info['goal_squared_positive']),float(info['pair_squared_positive'])+1e-5)

    def test_goal_loss_gradients_reach_both_endpoint_distances(self):
        class Network:
            def select(self,name):
                if name=='lam': return lambda params=None:jnp.array(1.)
                return lambda s,g,params=None:jnp.abs(s[:,0]-g[:,0])
        class Agent:
            network=Network()
            config={'eps':.05}
        def penalty(s,ns):
            batch=dict(observations=jnp.array([[s,0.]]),
                       next_observations=jnp.array([[ns,0.]]),
                       value_goals=jnp.array([[10.,0.]]))
            return critic_loss(Agent(),batch,None,'goal')[1]['d_pos_loss']
        # d(0,10)-d(2,10)-1=1; differentiate the implemented B penalty.
        self.assertEqual(float(jax.grad(penalty,0)(0.,2.)),-2.)
        self.assertEqual(float(jax.grad(penalty,1)(0.,2.)),2.)
        # Reverse transition increases distance, yet is not penalized.
        self.assertEqual(float(penalty(2.,0.)),0.)



if __name__=='__main__':
    unittest.main()
