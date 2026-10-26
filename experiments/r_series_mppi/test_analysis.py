import numpy as np
from .analyze import Coverage,describe,pair_details

def test_periodic_joint_coverage_and_empty():
    c=Coverage([[0,0,-np.pi+.01],[2,0,0]])
    a=c.query([[0,0,np.pi-.01],[2,0,0]])
    np.testing.assert_allclose(a[0],[0,0,.02,.01,.02],atol=1e-12)
    np.testing.assert_allclose(a[1],0,atol=1e-12)
    assert c.query([]).shape==(0,5)

def test_paired_costs_exclude_failures_and_keep_ids():
    a=[dict(state_id='a',planner_seed=1,success=False,cost=1,steps=1),dict(state_id='b',planner_seed=1,success=True,cost=9,steps=7)]
    b=[dict(a[0],success=True,cost=10),dict(a[1],cost=11,steps=8)]
    s=pair_details(a,b)
    assert s['new_success']==[['a',1]] and s['lost_success']==[] and s['joint_success_cost_delta']==2
    assert s['common_success_differences'][0]['state_id']=='b'

def test_empty_distribution_is_missing_not_zero_cost():
    assert describe([])['mean'] is None
    assert describe([0,2])['mean']==1 and describe([0,2])['n']==2
