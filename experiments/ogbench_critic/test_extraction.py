"""Official gradient isolation, nonzero Adam freezes, count and I/O semantics."""
import unittest
import jax
import jax.numpy as jnp
import numpy as np
import optax
from common import QRLAgent,tree_equal
from extraction import update,init_optimizer,proof,official_loss,module_key
from actor_recording import EpisodeRecorder


class ExtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from agents.qrl import get_config
        cfg=get_config().to_dict();cfg.update(actor_hidden_dims=(16,16),value_hidden_dims=(16,16),latent_dim=16)
        cls.agent=QRLAgent.create(0,np.zeros((1,2),np.float32),np.zeros((1,2),np.float32),cfg)
        rng=np.random.RandomState(13);cls.batch={k:rng.normal(size=(16,2)).astype(np.float32) for k in ['observations','next_observations','actions','actor_goals']}

    def test_module_gradient_matches_official_and_frozen_nonzero_moments(self):
        a=self.agent.replace(network=self.agent.network.apply_gradients(jax.tree_util.tree_map(jnp.ones_like,self.agent.network.params)))
        for stage in ['dynamics','actor']:
            key=module_key(stage);keys=[k for k in a.network.params if k!=key];p=proof(a,keys)
            _,rng=jax.random.split(a.rng)
            full=jax.grad(lambda params:official_loss(a,self.batch,params,stage,rng)[0])(a.network.params)[key]
            restricted=jax.grad(lambda params:official_loss(a,self.batch,{**a.network.params,key:params},stage,rng)[0])(a.network.params[key])
            self.assertTrue(tree_equal(full,restricted))
            trained,_,_=update(a,init_optimizer(a,stage),self.batch,stage)
            self.assertEqual(p,proof(trained,keys));self.assertFalse(tree_equal(a.network.params[key],trained.network.params[key]))
            self.assertEqual(int(trained.network.opt_state[0].count),int(a.network.opt_state[0].count)+1)
            np.testing.assert_array_equal(trained.rng,jax.random.split(a.rng)[0])

    def test_large_count_gap_has_identical_float32_adam_update(self):
        tx=optax.adam(3e-4);params={'a':jnp.array([1.,2.],jnp.float32)};state=tx.init(params)
        state=(state[0]._replace(mu={'a':jnp.array([.2,.4])},nu={'a':jnp.array([.1,.3])}),state[1]);grads={'a':jnp.array([.3,.6])}
        for count in [100000,100999,101000,101999,102000]:
            for beta in [.9,.999]:self.assertEqual(float(1-jnp.asarray(beta,jnp.float32)**count),1)
            one=tx.update(grads,(state[0]._replace(count=jnp.asarray(count,jnp.int32)),state[1]),params)[0]
            two=tx.update(grads,(state[0]._replace(count=jnp.asarray(count+1000,jnp.int32)),state[1]),params)[0]
            self.assertTrue(tree_equal(one,two))

    def test_recording_adapter_forwards_identical_results(self):
        class Env:
            def reset(self,*args,**kwargs):self.reset_args=(args,kwargs);return reset_result
            def step(self,action):self.action=action;return step_result
        reset_result=(np.array([1.,2.]),{'goal':np.array([3.,4.])});step_result=(np.array([2.,3.]),1,False,True,{'success':True})
        env=Env();calls=[];wrapper=EpisodeRecorder(env,check=lambda:calls.append('check'),integration=lambda e:np.array([5.]),on_end=lambda n:calls.append(n))
        self.assertIs(wrapper.reset(seed=0,options={'task_id':1}),reset_result);action=np.array([1.,0.]);self.assertIs(wrapper.step(action),step_result)
        self.assertIs(env.action,action);self.assertEqual(env.reset_args,((),{'seed':0,'options':{'task_id':1}}));self.assertEqual(calls,['check','check',1])
        np.testing.assert_array_equal(wrapper.starts[0]['goal'],reset_result[1]['goal']);np.testing.assert_array_equal(wrapper.ends[0],[5.])


if __name__=='__main__':unittest.main()
