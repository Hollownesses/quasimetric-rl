"""A cancellation-safe factor must preserve the true Jacobian's quadratic form."""
import torch
from experiments.four_path_refinement.repair import stable_jacobian_gram, factor_jacobian_gram


def test_block_gram_and_fallback_match_explicit_jacobian():
    rng=torch.Generator().manual_seed(19)
    gx=torch.randn(7,11,generator=rng,dtype=torch.float64)
    gy=-gx+1e-7*torch.randn(7,11,generator=rng,dtype=torch.float64)
    fs=1e5*torch.randn(7,5,generator=rng,dtype=torch.float64)
    fd=fs+1e-5*torch.randn(7,5,generator=rng,dtype=torch.float64)
    pg=torch.randn(11,5,generator=rng,dtype=torch.float64)
    jac=torch.cat(((gx[:,:,None]*fs[:,None,:]+gy[:,:,None]*fd[:,None,:]).flatten(1),-pg.flatten()[None,:]))
    expected=jac@jac.T
    stable=stable_jacobian_gram(gx,gy,fs,fd,pg,block_size=3)
    torch.testing.assert_close(stable,expected,rtol=1e-13,atol=1e-13)
    # Force the failing path; the factor must come from the actual derivatives,
    # not clipping an indefinite matrix or loosening the PSD tolerance.
    broken=expected.clone();broken[0,0]=-1.
    diagnostics={}
    factor=factor_jacobian_gram(broken,gx,gy,fs,fd,pg,diagnostics)
    assert diagnostics['stable_block_fallback']
    torch.testing.assert_close(factor@factor.T,expected,rtol=1e-10,atol=1e-12)
    v=torch.randn(8,generator=rng,dtype=torch.float64)
    torch.testing.assert_close((v@factor).square().sum(),(v@jac).square().sum(),rtol=1e-12,atol=1e-12)


def test_well_conditioned_factor_does_not_use_fallback():
    gx=torch.eye(3,dtype=torch.float64);gy=torch.zeros_like(gx)
    fs=torch.ones(3,2,dtype=torch.float64);pg=torch.zeros(3,2,dtype=torch.float64)
    gram=stable_jacobian_gram(gx,gy,fs,fs,pg)
    diag={};a=factor_jacobian_gram(gram,gx,gy,fs,fs,pg,diag)
    assert not diag['stable_block_fallback']
    torch.testing.assert_close(a@a.T,gram)
