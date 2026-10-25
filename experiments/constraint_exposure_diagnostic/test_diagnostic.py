import numpy as np
import pytest
from .core import (edge_index, branch_neighborhood, shortest_exposure, rollout_exposure,
                   dual_window, score_error_identity, residual_stats)


def test_edge_identity_keeps_actions_costs_and_anchor_separate():
    src=np.array([0,0,0,0,1]);dst=np.array([1,1,1,1,1])
    actions=np.array([[0.],[0.],[1.],[0.],[0.]],np.float32)
    reps,mapping=edge_index(src,dst,actions,np.array([2.,2.,2.,3.,0.]),np.array([1,1,1,1,0],bool))
    np.testing.assert_array_equal(reps,[0,2,3]);np.testing.assert_array_equal(mapping,[0,0,1,2,-1])


def test_branch_neighborhood_is_hops_and_includes_incoming_direction():
    src=np.array([0,1,1,2,3,4]);dst=np.array([1,2,3,4,4,5])
    degree,distance=branch_neighborhood(src,dst,np.arange(6),6,radius=1)
    assert degree[1]==2
    np.testing.assert_array_equal(distance,[1,0,1,1,-1,-1])


def test_far_source_reference_paths_include_shared_near_suffix():
    src=np.array([0,1,2]);dst=np.array([1,2,3]);mapping=np.arange(3)
    ex=shortest_exposure([0,1],np.array([0,1,2,-1]),src,dst,mapping,np.array([3.,2.,1.,0.]))
    np.testing.assert_array_equal(ex,[1,2,2])
    with pytest.raises(ValueError):
        shortest_exposure([0],np.array([1,1,2,-1]),src,dst,mapping,np.array([3.,2.,1.,0.]))


def fixture_rollouts():
    # At 0 the chosen direct path costs 5 vs optimal 3; common suffix costs 1.
    src=np.array([0,0,1,2]);dst=np.array([1,2,3,3]);cost=np.array([2.,4.,1.,1.])
    reference=np.array([3.,1.,1.,0.]);mapping=np.arange(4)
    eps=[dict(start=0,status='success',rows=[1,3],cost=5.,regret=2.),
         dict(start=0,status='success',rows=[1,3],cost=5.,regret=2.),
         dict(start=2,status='success',rows=[3],cost=1.,regret=0.)]
    return eps,src,dst,cost,mapping,reference


def test_rollout_unique_start_exposure_and_regret_telescope():
    args=fixture_rollouts()
    exposure,contribution,regrets=rollout_exposure(*args,goal=3,unique=True)
    np.testing.assert_array_equal(exposure,[0,1,0,2]);np.testing.assert_allclose(contribution,[0,1,0,0])
    np.testing.assert_allclose(regrets,[2,0])
    e,c,r=rollout_exposure(*args,goal=3,unique=False)
    np.testing.assert_array_equal(e,[0,2,0,3]);assert c.sum()==pytest.approx(4/3)


def test_rollout_rejects_fake_connection_or_wrong_regret():
    args=list(fixture_rollouts());args[0][0]['rows']=[0,3]
    with pytest.raises(ValueError):rollout_exposure(*args,goal=3)
    args=list(fixture_rollouts());args[0][0]['regret']=0.
    with pytest.raises(ValueError):rollout_exposure(*args,goal=3)


def test_dual_window_preserves_record_measure_and_endpoint_limit():
    h0=np.array([1.,1.,-2.]);h1=np.array([2.,.5,-1.])
    result=dual_window(h0,h1,np.array([1.,4.,3.]),np.array([3.,5.,0.]),
                       np.array([2,2,2]),np.ones(3,bool))
    assert result['persistent_positive']==2
    assert result['persistent_growing_dual']==2
    assert result['persistent_growing_dual_worsening']==1
    assert result['persistent']['lambda50_mean']==4.
    assert result['persistent']['effective_phr_coefficient_mean']==129.
    assert result['persistent']['projected_dual_residual_rms']==pytest.approx(np.sqrt((4+.25)/2))
    # Projected decrease may clamp, so delta/visit is deliberately a net change only.
    assert result['all']['delta_per_visit_mean']==pytest.approx(0.)
    with pytest.raises(ValueError):
        dual_window(h0,h1,np.ones(3),np.ones(3),np.zeros(3),np.ones(3,bool))


def test_branch_error_contrast_explains_wrong_choice_not_local_violation():
    _,src,dst,cost,_,reference=fixture_rollouts()
    prediction=np.array([3.,4.,0.,0.])
    result=score_error_identity(cost,prediction,reference,dst,np.array([0,1]),1)
    assert result['regret']==2.
    assert result['successor_error_contrast']==-4.
    assert result['predicted_choice_minus_reference_best']==-2.


def test_positive_residual_statistics_exclude_slack_and_allow_empty():
    h=np.array([-10.,0.,1e-6,2.]);v=residual_stats(h,np.ones(4,bool))
    assert v['violations']==1 and v['mean_positive']==pytest.approx(.50000025)
    assert residual_stats(h,np.zeros(4,bool))=={'count':0}
