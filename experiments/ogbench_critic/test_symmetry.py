"""Projection identities, metric structure and the exact frozen scoring path."""
import unittest
import numpy as np

from common import QRLAgent,distances
from symmetry import components,paired_statistics,SymmetricScore


class SymmetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from agents.qrl import get_config
        cfg=get_config().to_dict();cfg.update(actor_hidden_dims=(16,16),value_hidden_dims=(16,16),latent_dim=16)
        cls.agent=QRLAgent.create(0,np.zeros((1,2),np.float32),np.zeros((1,2),np.float32),cfg)

    def test_exact_definition_symmetry_triangle_and_zero(self):
        ob=np.array([[0.,0.],[1.,2.],[3.,4.],[-1.,2.]],np.float32)
        source=np.repeat(ob,len(ob),axis=0);goal=np.tile(ob,(len(ob),1))
        f=np.asarray(distances(self.agent,source,goal));r=np.asarray(distances(self.agent,goal,source));c=components(f,r)
        np.testing.assert_allclose(c['forward'],c['symmetric']+c['antisymmetric'],atol=1e-6,rtol=1e-6)
        metric=c['symmetric'].reshape(4,4);np.testing.assert_array_equal(metric,metric.T)
        np.testing.assert_array_equal(np.diag(metric),np.zeros(4));self.assertTrue((metric>=-1e-6).all())
        for k in range(4):self.assertTrue(np.all(metric<=metric[:,k,None]+metric[None,k,:]+1e-5))

    def test_jensen_paired_objective_and_pair_risk(self):
        f=np.array([1.,20.,100.,200.]);r=np.array([10.,1.,50.,400.])
        result=paired_statistics(f,r,f/100,r/100)
        self.assertLessEqual(result['symmetric_push'],result['paired_push']+1e-12)
        self.assertLessEqual(result['symmetric_pair_risk'],result['paired_pair_risk']+1e-12)
        self.assertEqual(result['symmetric_mean'],.5*(result['forward_mean']+result['reverse_mean']))

    def test_cached_score_is_original_two_directions(self):
        ob=np.arange(16,dtype=np.float32).reshape(8,2);goal=ob+3;score=SymmetricScore(self.agent)
        expected=.5*(np.asarray(distances(self.agent,ob,goal))+np.asarray(distances(self.agent,goal,ob)))
        np.testing.assert_array_equal(score(ob,goal),expected)
        np.testing.assert_array_equal(score(goal,ob),expected)
        self.assertGreater(score.cache.statistics()['first_hit_audits'],0)


if __name__=='__main__':unittest.main()
