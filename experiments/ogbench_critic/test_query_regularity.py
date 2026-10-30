"""Constructive diagonal-discontinuity and effective-input alias examples."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from roadmap_metric import RoadmapMetric


class RegularityTests(unittest.TestCase):
    def test_positive_diagonal_limit_and_same_point_zero(self):
        source=np.array([[0,0],[.2,0],[.4,0],[.4,.2]],np.float32);target=np.array([[.2,0],[.4,0],[.4,.2],[.4,.4]],np.float32)
        teacher=RoadmapMetric.build(source,target,connections=5)
        with tempfile.TemporaryDirectory() as temp:
            teacher.precompute(Path(temp)/'d.npy');x=np.array([[.11,.13]],np.float64)
            _,b,_=teacher.attachments(x);limit=2*b.min()
            self.assertGreater(limit,0);self.assertEqual(float(teacher(x,x)[0]),0)
            for eps in [1e-5,1e-6,1e-7]:
                y=x+np.array([eps,eps/np.sqrt(2)])
                self.assertAlmostEqual(float(teacher(x,y)[0]),limit,delta=20*eps)

    def test_distinct_teacher_queries_can_have_identical_float32_input(self):
        teacher=RoadmapMetric(np.array([[0,0],[2,0]],np.float64),np.array([[0,1]]),np.array([0,1]),connections=1)
        with tempfile.TemporaryDirectory() as temp:
            teacher.precompute(Path(temp)/'d.npy');x=np.array([[.5,.1]],np.float64);y=x+1e-10
            self.assertFalse(np.array_equal(x,y));np.testing.assert_array_equal(x.astype(np.float32),y.astype(np.float32))
            self.assertEqual(float(teacher(x,x)[0]),0);self.assertGreater(float(teacher(x,y)[0]),0)


if __name__=='__main__':unittest.main()
