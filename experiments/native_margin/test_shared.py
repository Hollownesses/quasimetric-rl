import numpy as np
from experiments.native_margin.shared import select_pair,source_weights

def test_pair_is_nearest_strict_and_independent_of_prediction():
    assert select_pair([4,2,2,3,9])==(1,3)
    assert select_pair([1,1,1]) is None
    assert select_pair([2,1,3],[0,2,4])==(1,2)
    assert select_pair([1,2],[5,4]) is None

def test_global_source_weights_equalize_mass_without_batch_renormalization():
    weights,inverse,count=source_weights(np.array([[1,2],[1,2],[3,4],[5,6],[5,6],[5,6]]))
    np.testing.assert_allclose(weights.mean(),1)
    np.testing.assert_allclose(np.bincount(inverse,weights),[2,2,2])
    assert weights[[0,2]].mean()!=1
