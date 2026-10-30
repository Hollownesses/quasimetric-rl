"""Frozen equal-weight forward/reverse QRL projection; no learning."""
import numpy as np

from exact_score_cache import ExactScoreCache


def components(forward,reverse):
    forward,reverse=np.asarray(forward),np.asarray(reverse)
    assert forward.shape==reverse.shape and np.isfinite(forward).all() and np.isfinite(reverse).all()
    return dict(forward=forward,reverse=reverse,symmetric=.5*(forward+reverse),antisymmetric=.5*(forward-reverse))


def push(distance):
    return 100*np.logaddexp(0,5-np.asarray(distance,dtype=np.float64)/100)


class SymmetricScore:
    def __init__(self,agent):self.cache=ExactScoreCache(agent,capacity=4096,audit=True)

    def __call__(self,observations,goals):
        forward=self.cache(observations,goals);reverse=self.cache(goals,observations)
        return components(forward,reverse)['symmetric']


def paired_statistics(forward,reverse,pair_forward,pair_reverse):
    f,r,pf,pr=[np.asarray(x,dtype=np.float64) for x in [forward,reverse,pair_forward,pair_reverse]]
    sym=.5*(f+r);ps=.5*(pf+pr)
    return dict(forward_mean=float(f.mean()),reverse_mean=float(r.mean()),symmetric_mean=float(sym.mean()),
        mean_absolute_asymmetry=float(np.abs(f-r).mean()),
        mean_relative_asymmetry=float((np.abs(f-r)/np.maximum(f+r,1e-12)).mean()),
        unpaired_forward_push=float(push(f).mean()),unpaired_reverse_push=float(push(r).mean()),
        paired_push=float(.5*(push(f)+push(r)).mean()),symmetric_push=float(push(sym).mean()),
        forward_pair_risk=float(np.square(np.maximum(pf-1,0)).mean()),
        reverse_pair_risk=float(np.square(np.maximum(pr-1,0)).mean()),
        symmetric_pair_risk=float(np.square(np.maximum(ps-1,0)).mean()),
        paired_pair_risk=float(.5*(np.square(np.maximum(pf-1,0))+np.square(np.maximum(pr-1,0))).mean()))
