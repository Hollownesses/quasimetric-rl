import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from common import QRLAgent,tree_equal
from multistep import initialize,encoder,embeddings,iqe,loss,update,sample,UNIT,load_state,save
import jax
import jax.numpy as jnp
import numpy as np
from utils.datasets import Dataset,GCDataset


class BackupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from agents.qrl import get_config
        c=get_config().to_dict();c.update(actor_hidden_dims=(16,16),value_hidden_dims=(16,16),latent_dim=16)
        cls.base=QRLAgent.create(0,np.zeros((1,2),np.float32),np.zeros((1,2),np.float32),c)
        rng=np.random.RandomState(3)
        cls.batch={k:rng.normal(size=(4,2)).astype(np.float32) for k in
                   ['observations','next_observations','value_goals','actions']}
        cls.batch.update(waypoint=cls.batch['next_observations'],offsets=np.ones(4,np.float32),
            goal_choices=np.column_stack([np.arange(4),np.tile(np.arange(4),(4,1))]).astype(np.int32))

    def test_initial_distance_scaling_and_action_encoder_equality(self):
        state=initialize(self.base);b=self.batch
        old=self.base.network.select('value')(b['observations'],b['value_goals'])
        d,ps,_=embeddings(self.base,state.params,b['observations'],b['value_goals'])
        pq=encoder(self.base).apply({'params':state.params['q']},np.concatenate([b['observations'],b['actions']],axis=-1))
        np.testing.assert_allclose(d,UNIT*np.asarray(old),atol=1e-6,rtol=2e-5)
        np.testing.assert_allclose(ps,pq,atol=1e-7)

    def test_one_and_multi_are_identical_at_one_step(self):
        s=initialize(self.base)
        a,ia=loss(self.base,s.params,self.batch,'one');b,ib=loss(self.base,s.params,self.batch,'multi')
        self.assertTrue(tree_equal((a,ia),(b,ib)))
        ga=jax.grad(lambda p:loss(self.base,p,self.batch,'one')[0])(s.params)
        gb=jax.grad(lambda p:loss(self.base,p,self.batch,'multi')[0])(s.params)
        self.assertTrue(tree_equal(ga,gb))

    def test_full_enumeration_matches_diagonal_allpair_objective(self):
        from multistep import GAMMA,DIAG_WEIGHT,CLIP
        s=initialize(self.base);b=self.batch
        _,ps,pg=embeddings(self.base,s.params,b['observations'],b['value_goals'])
        _,pw,_=embeddings(self.base,s.params,b['next_observations'],b['value_goals'])
        pq=encoder(self.base).apply({'params':s.params['q']},np.concatenate([b['observations'],b['actions']],axis=-1))
        d=iqe(self.base,s.params,pq[:,None,:],pg[None,:,:])
        target=iqe(self.base,s.params,pw[:,None,:],pg[None,:,:])
        delta=d-target
        divergence=jnp.where(delta>CLIP,delta,GAMMA*jnp.exp(jnp.minimum(delta,CLIP))-d)
        expected=DIAG_WEIGHT*jnp.diag(divergence).mean()+(1-DIAG_WEIGHT)*divergence.mean()
        actual=loss(self.base,s.params,b,'one')[1]['backup']
        np.testing.assert_allclose(actual,expected,atol=1e-6)

    def test_sampler_respects_official_boundaries_and_real_next(self):
        ds=Dataset.create(observations=np.arange(12,dtype=np.float32).reshape(6,2),
            actions=np.zeros((6,2),np.float32),terminals=np.array([0,1,1,0,1,1],np.float32),
            valids=np.array([1,1,0,1,1,0],np.float32))
        dataset=GCDataset(ds,self.base.config)
        b=sample(dataset,np.random.RandomState(22),1024)
        i,g,w,f=[b[k] for k in ['source_indices','goal_indices','waypoint_indices','final_indices']]
        self.assertTrue(np.all(ds['valids'][i]>0))
        self.assertTrue(np.all((g>=i)&(g<=f)))
        self.assertTrue(np.all((w>=i)&(w<=np.maximum(f,i+1))))
        self.assertTrue(np.array_equal(b['offsets'],w-i))
        self.assertTrue(np.array_equal(b['next_observations'],ds['observations'][i+1]))
        self.assertTrue(np.array_equal(b['goal_choices'][:,0],np.arange(1024)))

    def test_optimizer_count_and_full_roundtrip(self):
        from multistep import as_agent
        from train import freeze_proof
        s=initialize(self.base);new,info=update(self.base,s,self.batch,'multi')
        self.assertEqual(int(new.step),1);self.assertEqual(int(new.opt_state[0].count),1)
        self.assertFalse(tree_equal(s.params,new.params))
        self.assertTrue(all(np.isfinite(np.asarray(v)).all() for v in jax.tree_util.tree_leaves(new)))
        self.assertEqual(freeze_proof(as_agent(self.base,new)),freeze_proof(self.base))
        with tempfile.TemporaryDirectory() as d:
            save(self.base,new,d,'multi',np.random.RandomState(44))
            again,bundle=load_state(self.base,d,1)
            self.assertTrue(tree_equal(new,again));self.assertEqual(bundle['mode'],'multi')


if __name__=='__main__':unittest.main()
