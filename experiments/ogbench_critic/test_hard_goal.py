"""Proposal lineage, bidirectional tie semantics and max-objective gradient."""
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from common import QRLAgent
from iqe_distill import loss
from goal_support import GoalSupport
from hard_goal import HardGoal,paired_errors,select_worst


class HardGoalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from agents.qrl import get_config
        cfg=get_config().to_dict();cfg.update(actor_hidden_dims=(16,16),value_hidden_dims=(16,16),latent_dim=16)
        cls.agent=QRLAgent.create(0,np.zeros((1,2),np.float32),np.zeros((1,2),np.float32),cfg)
        rng=np.random.RandomState(13);cls.batch={k:rng.normal(size=(16,2)).astype(np.float32) for k in ['observations','value_goals','next_observations']}
        cls.nodes=rng.normal(size=(32,2)).astype(np.float32)

    def test_proposal_zero_source_population_rng_and_inputs(self):
        sampler=HardGoal(self.nodes);state=np.random.get_state();before={k:v.copy() for k,v in self.batch.items()}
        uniform,sources,goals,ids=sampler.proposals(self.batch,0);after=np.random.get_state()
        expected=GoalSupport(self.nodes)(self.batch,0)
        for k in expected:np.testing.assert_array_equal(uniform[k],expected[k]);np.testing.assert_array_equal(self.batch[k],before[k])
        np.testing.assert_array_equal(sources,np.repeat(expected['observations'][1::2],8,axis=0))
        np.testing.assert_array_equal(goals[:,0],expected['value_goals'][1::2]);self.assertEqual(ids.shape,(8,8))
        for k in range(8):
            phase=k*(len(self.nodes)//8);indices=((np.arange(32)+phase)*sampler.stride)%32
            np.testing.assert_array_equal(np.sort(indices),np.arange(32))
        self.assertEqual(state[0],after[0]);np.testing.assert_array_equal(state[1],after[1]);self.assertEqual(state[2:],after[2:])

    def test_bidirectional_error_and_first_tie_selection(self):
        s=np.array([[.2,.4],[.5,-.3]],np.float32);g=-s;targets=np.array([1,2],np.float32);a=self.agent
        f=a.network.select('value')(s,g);r=a.network.select('value')(g,s)
        np.testing.assert_allclose(paired_errors(a,s,g,targets),.5*((f-targets)**2+(r-targets)**2),rtol=2e-5,atol=3e-5)
        goals=np.zeros((8,3,2),np.float32);labels=np.arange(24,dtype=np.float32).reshape(8,3)
        errors=np.tile(np.array([4,9,9]),(8,1));selected,t,idx=select_worst(self.batch,goals,labels,errors)
        np.testing.assert_array_equal(idx,np.ones(8,dtype=int));np.testing.assert_array_equal(t,labels[:,1])
        np.testing.assert_array_equal(selected['value_goals'][::2],self.batch['value_goals'][::2])

    def test_selected_gradient_matches_explicit_max_objective(self):
        a=self.agent;b=self.batch;sampler=HardGoal(self.nodes,candidates=3);uniform,sources,goals,_=sampler.proposals(b,0)
        labels=np.arange(24,dtype=np.float32).reshape(8,3)/4+1
        errors=np.asarray(paired_errors(a,sources,goals.reshape(-1,2),labels.reshape(-1))).reshape(8,3)
        selected,chosen,_=select_worst(uniform,goals,labels,errors);targets=np.ones(16,np.float32);targets[1::2]=chosen
        def reference(params):
            value=a.network.select('value');obs=uniform['observations'][::2];g=uniform['value_goals'][::2]
            f=value(obs,g,params=params);r=value(g,obs,params=params);even=.5*((f-1)**2+(r-1)**2).mean()
            flat=goals.reshape(-1,2);f=value(sources,flat,params=params);r=value(flat,sources,params=params)
            error=.5*((f-labels.reshape(-1))**2+(r-labels.reshape(-1))**2)
            return .5*even+.5*error.reshape(8,3).max(axis=1).mean()
        actual=lambda p:loss(a,selected,p,targets)[0]
        np.testing.assert_allclose(actual(a.network.params),reference(a.network.params),rtol=2e-5,atol=3e-5)
        ga=jax.grad(actual)(a.network.params);gr=jax.grad(reference)(a.network.params)
        for x,y in zip(jax.tree_util.tree_leaves(ga),jax.tree_util.tree_leaves(gr)):np.testing.assert_allclose(x,y,rtol=2e-5,atol=3e-5)


if __name__=='__main__':unittest.main()
