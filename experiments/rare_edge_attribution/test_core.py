import numpy as np
from experiments.rare_edge_attribution.core import select_edges,counterfactual

def test_shared_suffix_must_cancel_and_negative_h_is_not_clipped():
    h=np.array([.8,-.4,9.])
    assert counterfactual(-.2,[0,2],[1,2],h,[2])==(-.2,0.)
    np.testing.assert_allclose(counterfactual(-.2,[0,2],[1,2],h,[0]),[.6,.8])
    np.testing.assert_allclose(counterfactual(-.2,[0,2],[1,2],h,[1]),[.2,.4])
    assert counterfactual(-.2,[0,2],[1,2],h,[1],True)==(-.2,0.)

def test_reference_selection_and_missing_cost_matched_control():
    c=np.array([1.,1.04,4.,1.01]);counts=np.array([1,10,100,2])
    assert select_edges([0,1,2,3],counts,c)==(0,3)
    assert select_edges([0,2],counts,c)==(0,None)
    assert select_edges([],counts,c)==(None,None)
