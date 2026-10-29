"""Check that the mechanism audit decomposes the implemented critic gradients."""
import copy
import unittest

from common import QRLAgent,tree_equal
from critic import subset,init_optimizer,critic_loss
from mechanism import part_gradient,probe,zero_momentum_motion
import jax
import jax.numpy as jnp
import numpy as np


class MechanismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from agents.qrl import get_config
        config=get_config().to_dict()
        config.update(actor_hidden_dims=(16,16),value_hidden_dims=(16,16),latent_dim=16)
        cls.agent=QRLAgent.create(0,np.zeros((1,2),np.float32),np.zeros((1,2),np.float32),config)
        rng=np.random.RandomState(88)
        cls.batch={k:rng.normal(size=(8,2)).astype(np.float32) for k in
                   ['observations','next_observations','value_goals']}

    def test_decomposition_matches_actual_value_gradients(self):
        agent=self.agent;p=subset(agent.network.params)
        push=part_gradient(agent,p,self.batch,'push')['modules_value']
        for arm in ['pair','goal']:
            constraint=part_gradient(agent,p,self.batch,arm)['modules_value']
            def loss(params):
                return critic_loss(agent,self.batch,{**agent.network.params,**params},arm)[0]
            actual=jax.grad(loss)(p)['modules_value']
            expected=jax.tree_util.tree_map(lambda x,y:x+y,push,constraint)
            for a,b in zip(jax.tree_util.tree_leaves(actual),jax.tree_util.tree_leaves(expected)):
                np.testing.assert_allclose(a,b,rtol=1e-5,atol=1e-5)

    def test_shape_removes_positive_distance_scaling(self):
        agent=self.agent;p=subset(agent.network.params)
        scaled=copy.deepcopy(p)
        phi=scaled['modules_value']['phi']
        key=max((k for k in phi if k.startswith('Dense_')),key=lambda k:int(k.split('_')[-1]))
        phi[key]=jax.tree_util.tree_map(lambda x:3*x,phi[key])
        points=np.array([[0.,0.],[1.,2.]],np.float32)
        goals=np.array([[4.,3.],[4.,3.]],np.float32)
        before=np.asarray(probe(agent,p,points,goals))
        after=np.asarray(probe(agent,scaled,points,goals))
        np.testing.assert_allclose(after[0],3*before[0],atol=2e-5,rtol=2e-5)
        np.testing.assert_allclose(after[1],before[1],atol=2e-5,rtol=2e-5)

    def test_zero_new_gradient_is_not_zero_adam_motion(self):
        agent=self.agent
        network=agent.network.apply_gradients(jax.tree_util.tree_map(jnp.ones_like,agent.network.params))
        agent=agent.replace(network=network)
        after=zero_momentum_motion(agent,init_optimizer(agent))
        self.assertFalse(tree_equal(agent.network.params['modules_value'],after.network.params['modules_value']))
        self.assertTrue(tree_equal(agent.network.params['modules_actor'],after.network.params['modules_actor']))
        self.assertTrue(tree_equal(agent.network.params['modules_dynamics'],after.network.params['modules_dynamics']))


if __name__=='__main__':unittest.main()
