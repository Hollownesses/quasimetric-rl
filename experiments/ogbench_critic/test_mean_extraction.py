"""Changed E6 lambda must not affect frozen-critic official extraction losses."""
import unittest
import jax
import numpy as np
from common import QRLAgent,tree_equal
from extraction import official_loss,module_key


class MeanExtractionTests(unittest.TestCase):
    def test_lambda_has_no_effect_on_loss_or_active_gradient(self):
        from agents.qrl import get_config
        cfg=get_config().to_dict();cfg.update(actor_hidden_dims=(16,16),value_hidden_dims=(16,16),latent_dim=16)
        a=QRLAgent.create(0,np.zeros((1,2),np.float32),np.zeros((1,2),np.float32),cfg)
        params={**a.network.params,'modules_lam':jax.tree_util.tree_map(lambda x:x+7,a.network.params['modules_lam'])}
        self.assertFalse(tree_equal(params['modules_lam'],a.network.params['modules_lam']))
        r=np.random.RandomState(13);batch={k:r.normal(size=(16,2)).astype(np.float32) for k in ['observations','next_observations','actions','actor_goals']}
        for stage in ['dynamics','actor']:
            key=module_key(stage)
            def calc(p):
                def fn(active):return official_loss(a,batch,{**p,key:active},stage,a.rng)
                return jax.value_and_grad(fn,has_aux=True)(p[key])
            self.assertTrue(tree_equal(calc(a.network.params),calc(params)))


if __name__=='__main__':unittest.main()
