"""Population feasibility; uses no goals, references, or evaluation labels."""
import numpy as np

def risk(distance,alpha=1.):
    return float(np.mean(np.maximum(alpha*np.asarray(distance,np.float64)-1,0)**2))

def hard_scale(distance,cost):
    d,c=np.broadcast_arrays(np.asarray(distance,np.float64),np.asarray(cost,np.float64))
    if not d.size or np.any(d<0) or np.any(c<=0) or not np.isfinite([d,c]).all():raise ValueError('invalid local distances/costs')
    if np.all(d<=c):return 1.,1.
    upper=min(1.,float(np.min(c[d>0]/d[d>0])))
    return upper,upper*(1-1e-4)

def mean_scale(distance,eps):
    d=np.asarray(distance,np.float64)
    if not d.size or not np.isfinite(d).all() or np.any(d<0) or eps<=0:raise ValueError('invalid mean risk input')
    if risk(d)<=eps:return 1.,1.,1.
    lo,hi=0.,1.
    for _ in range(60):
        mid=(lo+hi)/2
        if risk(d,mid)<=eps:lo=mid
        else:hi=mid
    return lo,hi,lo*(1-1e-4)
