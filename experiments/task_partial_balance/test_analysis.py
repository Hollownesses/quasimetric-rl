from types import SimpleNamespace
import numpy as np
import torch
from .analyze import repeat_groups


def test_multiplicity_groups_and_weighted_denominators():
    counts=np.array([1,2,5,17,65]);source=np.repeat(np.arange(5),counts)
    data=SimpleNamespace(original=torch.arange(len(source)),src=torch.tensor(source))
    ref=np.array([10.,20.,30.,40.,50.])
    pred=dict(R0=ref-2,Rhalf=ref+1,R1=ref+3)
    bins=[dict(label=str(i),min=int(c),max=int(c)) for i,c in enumerate(counts)]
    groups,unique,actual=repeat_groups(data,ref,pred,bins)
    assert np.array_equal(counts,actual) and np.array_equal(unique,np.arange(5))
    for i,c in enumerate(counts):
        g=groups[str(i)]
        assert g['n_sources']==1 and g['n_records']==c
        assert abs(g['task_mass']['Rhalf']-(.5*c/counts.sum()+.5/5))<1e-12
        for arm,error in [('R0',-2),('Rhalf',1),('R1',3)]:
            for mode in ('records','unique_sources'):
                d=g['metrics'][arm][mode];assert d['bias']==error and d['mae']==abs(error)
        assert g['changes']['Rhalf_minus_R0']['improved_sources']==1
        assert g['changes']['Rhalf_minus_R1']['improved_sources']==1


def test_group_mixed_counts_weights_and_deterioration():
    src=torch.tensor([0,1,1,2,2,2]);data=SimpleNamespace(original=torch.arange(6),src=src)
    ref=np.array([10.,20.,30.]);pred=dict(R0=ref,Rhalf=ref+np.array([1.,2.,3.]),R1=ref+4)
    groups,_,_=repeat_groups(data,ref,pred,[dict(label='all',min=1,max=None)])
    g=groups['all'];h=g['metrics']['Rhalf']
    assert h['unique_sources']['mae']==2
    assert h['records']['mae']==14/6
    assert g['changes']['Rhalf_minus_R0']['worsened_sources']==3
    assert g['changes']['Rhalf_minus_R1']['improved_sources']==3
