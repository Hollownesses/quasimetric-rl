from types import SimpleNamespace
import numpy as np
import torch
from .metrics import copy_groups,path_positive,successor_diagnostics,common_pair_changes


def test_copy_groups_keep_record_and_edge_denominators():
    h=np.array([1.,2.,2.,-1.]);rows=np.arange(4);reps=np.array([0,1,3]);ids=np.array([0,1,1,2]);counts=np.array([1,2,1])
    groups=copy_groups(h,rows,reps,ids,counts,[dict(label='1',min=1,max=1),dict(label='2+',min=2,max=None)])
    assert groups['1']['unique_edge_violations']['mean_positive']==.5
    assert groups['2+']['unique_edges']==1 and groups['2+']['records']==2
    assert groups['2+']['record_mass']==.5 and groups['2+']['unique_edge_mass']==1/3
    assert groups['2+']['unique_edge_violations']['max_positive']==2


def test_fixed_paths_not_changed_by_model_residuals():
    episodes=[dict(start=0,rows=[0,1]),dict(start=0,rows=[0,1]),dict(start=1,rows=[1,2])]
    h=np.array([2.,-3.,4.])
    unique=path_positive(h,episodes,True);records=path_positive(h,episodes,False)
    assert unique['count']==2 and unique['mean']==3
    assert records['count']==3 and records['mean']==8/3
    assert unique['paths'][0]['sum_signed']==-1 and unique['paths'][0]['sum_positive']==2


def test_successor_error_difference_can_flip_branch():
    # Two outgoing actions, true totals 2 and 5. Bad successor error makes worse action look preferable.
    data=SimpleNamespace(cost=torch.tensor([1.,1.]),dst=torch.tensor([1,2]))
    graph=SimpleNamespace(outgoing=[[0,1],[],[]]);ref=np.array([2.,1.,4.])
    bad=np.array([0.,1.,0.]);good=ref.copy()
    wrong=dict(branches=[dict(source=0,chosen_row=1,regret=3.)],summary=dict(pairwise_order_accuracy=0.))
    right=dict(branches=[dict(source=0,chosen_row=0,regret=0.)],summary=dict(pairwise_order_accuracy=1.))
    a=successor_diagnostics(data,graph,ref,bad,wrong);b=successor_diagnostics(data,graph,ref,good,right)
    assert a['branches'][0]['successor_error_contrast']==-4.
    assert a['branches'][0]['predicted_choice_minus_reference_best']==-1.
    changes=common_pair_changes(a,b)
    assert changes['wrong_to_correct']==1 and changes['correct_to_wrong']==0
