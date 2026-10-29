import unittest
import numpy as np
from field import point_clear,segment_clear,graph_metrics
from scipy.sparse import csr_matrix


class GraphTests(unittest.TestCase):
    def test_swept_edges_cannot_cut_corner(self):
        wall=np.array([[0.,0.,1.,1.]])
        a=np.array([[1.8,1.],[1.8,1.8],[2.,2.]])
        b=np.array([[1.,1.8],[2.,1.8],[3.,3.]])
        self.assertTrue(point_clear(a,wall,.7).all())
        self.assertTrue(point_clear(b,wall,.7).all())
        self.assertEqual(segment_clear(a,b,wall,.7).tolist(),[False,True,True])

    def test_minimax_checks_all_routes_and_success_region(self):
        # A direct route rises to 8; a detour only rises to 4.
        nodes=np.array([[0.,0.],[2.,0.],[4.,0.],[0.,2.],[2.,2.],[4.,2.]])
        edges=[(0,1),(1,2),(0,3),(3,4),(4,5),(5,2)]
        rows=[];cols=[]
        for a,b in edges:rows.extend([a,b]);cols.extend([b,a])
        g=csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(6,6))
        stats,data=graph_metrics(nodes,g,np.array([3.,8.,0.,4.,2.,1.]),goal=np.array([4.,0.]),anchor=np.array([0.,0.]))
        self.assertEqual(stats['minimum_peak'],4.)
        self.assertEqual(stats['shortest_route_peak'],8.)
        self.assertFalse(stats['strict_descent_from_anchor'])
        self.assertEqual(stats['reference_steps'],2.)


if __name__=='__main__':unittest.main()
