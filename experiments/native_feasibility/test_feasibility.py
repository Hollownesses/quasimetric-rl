"""Native hard/mean certificates need different aggregation semantics."""
import numpy as np
import pytest
from experiments.native_feasibility.core import hard_scale,mean_scale,risk

def test_hard_bound_uses_every_edge_and_handles_zero_distance():
    d=np.array([0.,2.,8.]);c=np.array([.1,1.,1.])
    upper,actual=hard_scale(d,c)
    assert upper==.125 and actual<upper
    assert np.max(actual*d-c)<0
    assert hard_scale([0.,.2],[1.,1.])==(1.,1.)

def test_population_mean_and_hard_constraint_are_different():
    d=np.array([0.,0.,2.]);lo,hi,a=mean_scale(d,.01)
    np.testing.assert_allclose(lo,(1+np.sqrt(.03))/2,rtol=0,atol=1e-14)
    assert hi-lo<1e-14 and risk(d,a)<.01
    assert np.max(a*d-1)>0
    assert mean_scale([0.,1.],.05)==(1.,1.,1.)

def test_invalid_inputs_do_not_create_silent_certificates():
    with pytest.raises(ValueError):hard_scale([2.],[0.])
    with pytest.raises(ValueError):mean_scale([float('nan')],.05)
