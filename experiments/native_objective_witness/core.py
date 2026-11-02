"""Feasible ray bounds and bounded native-query comparison, no reference labels."""
import copy
import numpy as np

def risk(d,a=1.):return float(np.mean(np.maximum(a*np.asarray(d,np.float64)-1,0)**2))

def hard_boundary(d,c):
    d,c=np.broadcast_arrays(np.asarray(d,np.float64),np.asarray(c,np.float64))
    if not d.size or np.any(d<0) or np.any(c<=0) or not np.isfinite([d,c]).all() or not np.any(d>0):raise ValueError('invalid or unbounded hard ray')
    upper=float(np.min(c[d>0]/d[d>0]))
    return upper,upper*(1-1e-4)

def mean_boundary(d,eps):
    d=np.asarray(d,np.float64)
    if not d.size or np.any(d<0) or not np.isfinite(d).all() or eps<=0:raise ValueError('invalid mean ray')
    lo=0.;hi=1.
    while risk(d,hi)<=eps and hi<64:lo=hi;hi*=2
    if risk(d,hi)<=eps:raise ValueError('mean ray not bracketed by fixed 64 cap')
    for _ in range(60):
        mid=(lo+hi)/2
        if risk(d,mid)<=eps:lo=mid
        else:hi=mid
    return lo,hi,lo*(1-1e-4)

def push_loss(distance):return 100*np.logaddexp(0,5-np.asarray(distance,np.float64)/100)

def hoeffding_halfwidth(n,comparisons=4,error=.05):
    if n<=0:raise ValueError('empty query audit')
    bound=float(push_loss(0))
    return bound*np.sqrt(2*np.log(2*comparisons/error)/n)

def last_dense(module):
    return max((k for k in module if k.startswith('Dense_')),key=lambda k:int(k.split('_')[-1]))

def coordinate_scale(params,alpha,latent_size):
    """Preserve latent dynamics in scaled coordinates; supports NumPy and JAX arrays."""
    if alpha<=0:raise ValueError('positive coordinate scale required')
    out=copy.deepcopy(params);phi=out['modules_value']['phi'];last=last_dense(phi)
    phi[last]={k:v*alpha for k,v in phi[last].items()}
    dyn=out['modules_dynamics'];last=last_dense(dyn);assert last!='Dense_0'
    kernel=dyn['Dense_0']['kernel'];assert kernel.shape[0]>latent_size
    multiplier=np.ones(kernel.shape[0],np.float32);multiplier[:latent_size]=1/alpha
    dyn['Dense_0']={**dyn['Dense_0'],'kernel':kernel*multiplier[:,None]}
    dyn[last]={k:v*alpha for k,v in dyn[last].items()}
    return out
