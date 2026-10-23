"""Signed path accounting and full-dataset gradient normalization."""
import numpy as np
import pytest
import torch
from experiments.expert_utrap_network.test_network import dataset
from experiments.connected_expert_utrap.test_connected import connected
from experiments.connected_expert_utrap.data import load_training_data
from experiments.four_path_factorial.run import IndexedIQE
from .diagnostic import decompose
from .objective import full_gradients


@pytest.mark.parametrize('edge,pred,expected',[
    ([.5,.3],[.7,.4,0],[1.2,.1,1.3]),
    ([1.5,1],[2.3,1,0],[-.5,.2,-.3])])
def test_signed_decomposition(edge,pred,expected):
    p=decompose(np.array([2.,1.,0.]),np.array([0,1,-1]),np.array([1,2]),
                np.ones(2),np.array(pred),np.array(edge))
    np.testing.assert_allclose([p[k][0] for k in ('edge_slack','triangle_slack','gap')],expected)


def test_invalid_reference_path_rejected():
    with pytest.raises(AssertionError):
        decompose(np.array([2.,1.,0.]),np.array([0,1,-1]),np.array([1,2]),
                  np.array([.5,1]),np.zeros(3),np.zeros(2))


def test_full_gradient_chunking_and_readonly(connected):
    torch.set_num_threads(1)
    d=load_training_data(connected[0]);m=IndexedIQE(d.small,'encoder',0,initialization_seed=606)
    before={k:v.clone() for k,v in m.state_dict().items()}
    dual=torch.linspace(0,3,len(d.src));dual_before=dual.clone()
    a,va,ha=full_gradients(m,d,dual,batch=len(d.original))
    b,vb,hb=full_gradients(m,d,dual,batch=17)
    for k in va:torch.testing.assert_close(va[k],vb[k],rtol=2e-4,atol=2e-5)
    np.testing.assert_allclose(ha,hb,atol=2e-5)
    assert a['augmented_loss']==pytest.approx(b['augmented_loss'],rel=1e-5,abs=1e-4)
    assert not torch.equal(va['phr'],va['lagrange'])
    assert all(p.grad is None for p in m.parameters())
    assert torch.equal(dual,dual_before)
    for k,v in m.state_dict().items():assert torch.equal(v,before[k])
