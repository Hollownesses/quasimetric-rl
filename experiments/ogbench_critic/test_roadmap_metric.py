"""Query metric consistency, actual prototypes, disconnected refusal and I/O."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from roadmap_metric import RoadmapMetric


class RoadmapTests(unittest.TestCase):
    def make(self):
        nodes=np.array([[0,0],[.2,0],[.4,0],[.4,.2],[.4,.4]],np.float32)
        return RoadmapMetric.build(nodes[:-1],nodes[1:],connections=5),nodes

    def test_actual_prototypes_edges_and_saved_query_equality(self):
        graph,nodes=self.make();points=np.concatenate([nodes[:-1],nodes[1:]])
        np.testing.assert_array_equal(graph.nodes,points[graph.prototype_indices])
        self.assertEqual(len(graph.edges),4)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);graph.save(path/'graph.npz');graph.precompute(path/'distance.npy')
            restored=RoadmapMetric.load(path/'graph.npz',path/'distance.npy')
            a=np.array([[.1,.1],[.25,.1]]);b=np.array([[.4,.4],[0,0]])
            np.testing.assert_array_equal(graph(a,b),restored(a,b))

    def test_correction_conditions_and_query_metric_triangle(self):
        graph,_=self.make()
        with tempfile.TemporaryDirectory() as temp:
            graph.precompute(Path(temp)/'dist.npy')
            points=np.array([[0,0],[.1,.1],[.2,.2],[.4,.4],[.5,.3],[.25,-.1]],np.float64)
            _,_,correction=graph.attachments(points)
            self.assertGreater(float(correction.max()),0)
            a=np.repeat(points,len(points),axis=0);b=np.tile(points,(len(points),1))
            distance=graph(a,b).reshape(len(points),len(points))
            np.testing.assert_allclose(distance,distance.T,atol=1e-10,rtol=1e-10)
            np.testing.assert_array_equal(np.diag(distance),np.zeros(len(points)))
            for k in range(len(points)):
                self.assertTrue(np.all(distance<=distance[:,k,None]+distance[None,k,:]+1e-9))

    def test_disconnected_graph_refused_without_filtering(self):
        points=np.array([[0,0],[10,10]],np.float32)
        with self.assertRaisesRegex(AssertionError,'Disconnected'):
            RoadmapMetric.build(points,points,connections=2)


if __name__=='__main__':unittest.main()
