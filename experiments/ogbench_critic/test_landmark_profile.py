"""Direct teacher equivalence, deterministic selection and pseudometric bounds."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from roadmap_metric import RoadmapMetric
from landmark_profile import LandmarkProfile


class ProfileTests(unittest.TestCase):
    def test_profiles_direct_teacher_selection_and_roundtrip(self):
        source=np.array([[0,0],[.2,0],[.4,0],[.4,.2]],np.float32);successor=np.array([[.2,0],[.4,0],[.4,.2],[.4,.4]],np.float32)
        with tempfile.TemporaryDirectory() as temp:
            temp=Path(temp);teacher=RoadmapMetric.build(source,successor,connections=5);teacher.precompute(temp/'d.npy')
            metric=LandmarkProfile.build(teacher,count=3);np.testing.assert_array_equal(metric.indices,[0,4,2])
            points=np.concatenate([teacher.nodes,np.array([[.12,.07],[.32,.13],[.6,-.1]])])
            expected=np.column_stack([teacher(points,np.broadcast_to(l,points.shape)) for l in metric.landmarks])
            np.testing.assert_allclose(metric.profile(points),expected,rtol=1e-10,atol=1e-10)
            metric.save(temp/'profile.npz');restored=LandmarkProfile.load(teacher,temp/'profile.npz')
            np.testing.assert_array_equal(metric.profile(points),restored.profile(points))
            a=np.repeat(points,len(points),axis=0);b=np.tile(points,(len(points),1))
            d=metric(a,b).reshape(len(points),len(points));td=teacher(a,b).reshape(d.shape)
            np.testing.assert_array_equal(np.diag(d),np.zeros(len(points)));np.testing.assert_array_equal(d,d.T)
            self.assertTrue(np.all(d<=td+1e-9))
            for k in range(len(points)):self.assertTrue(np.all(d<=d[:,k,None]+d[None,k,:]+1e-9))

    def test_single_landmark_not_assumed_injective(self):
        teacher=RoadmapMetric(np.array([[0,0],[1,0]],np.float64),np.array([[0,1]]),np.array([0,1]),connections=1)
        with tempfile.TemporaryDirectory() as temp:
            teacher.precompute(Path(temp)/'d.npy');metric=LandmarkProfile.build(teacher,count=1)
            a=np.array([[0,.1]]);b=np.array([[0,-.1]])
            self.assertGreater(float(teacher(a,b)[0]),0);self.assertEqual(float(metric(a,b)[0]),0)


if __name__=='__main__':unittest.main()
