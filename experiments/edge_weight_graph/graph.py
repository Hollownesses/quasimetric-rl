"""Frozen six-node graph, scoring, and shared signed path diagnostics."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import dijkstra

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.unified_mechanism.metrics import path_metrics, write_rows

NAMES = ('s','a','u','b','v','g')
S,A,U,B,V,G = range(6)
SRC = np.array([S,A,U,S,B,V,G])
DST = np.array([A,U,G,B,V,G,S])
Q = (1/7,.01,.001,.0003,.0001,.00003,.00001,.000001)
ETA = 1e-4
NEAR = 1e-6
TIE = 1e-6
HASH = 'e89405668fb157e68617264eebadc7a35340b95bacdcd125ab880468980c84ff'


def cases():
    for delta in (1,2):
        for q in (None,*Q):
            yield delta,q


def tag(delta,q):
    weight = 'hard' if q is None else 'soft_'+format(q,'.9g')
    return f'd{delta}_{weight}'


def graph(delta,q):
    cost=np.ones(7);cost[5]+=delta
    weights=np.full(7,1/7 if q is None else (1-q)/6)
    if q is not None: weights[2]=q
    assert np.isclose(weights.sum(),1) and weights.min()>0
    return cost,weights


def shortest(cost):
    adj=np.full((6,6),np.inf)
    adj[SRC,DST]=cost
    return dijkstra(adj,directed=True)


def freeze(out):
    data=(HERE/'PROTOCOL.md').read_bytes()
    assert hashlib.sha256(data).hexdigest()==HASH
    out.mkdir(parents=True,exist_ok=False)
    (out/'PROTOCOL.md').write_bytes(data)


def save(path, obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def evaluate(d,delta,q):
    cost,w=graph(delta,q)
    assert d.shape==(6,6) and np.isfinite(d).all()
    truth=shortest(cost)[:,G]
    np.testing.assert_array_equal(truth,[3,2,1,2+delta,1+delta,0])
    j=d[:,G];h=d[SRC,DST]-cost
    margin=j[B]-j[A]
    action='a' if margin>TIE else 'b' if margin < -TIE else 'tie'
    low=delta if action=='b' else 0
    high=0 if action=='a' else delta
    suffix=[]
    for nodes,idx in [([A,U,G],[1,2]),([B,V,G],[4,5])]:
        suffix.append(path_metrics(cost[idx],d[SRC[idx],DST[idx]],j[nodes]))
    ha,hb=[s['sum_h'] for s in suffix];ta,tb=[s['sum_tau'] for s in suffix]
    identity=abs(margin-(delta+hb-ha-tb+ta))
    assert identity<=1e-8
    triangle=max(0.,(d[:,None,:]-d[:,:,None]-d[None,:,:]).max())
    return dict(delta=delta,constraint='hard' if q is None else 'mean_square',q=q,
                F=j[:G].mean(),m=margin,action=action,regret_low=low,regret_high=high,
                max_h_plus=max(0.,h.max()),risk=w@np.maximum(h,0)**2,
                diag_residual=np.abs(d.diagonal()).max(),nonneg_residual=max(0.,-d.min()),
                triangle_residual=triangle,H_a=ha,H_b=hb,T_a=ta,T_b=tb,identity_error=identity)


def feasible(row, *, table):
    struct=max(row[k] for k in ('diag_residual','nonneg_residual','triangle_residual'))
    if row['constraint']=='hard':
        local=row['max_h_plus'] <= (1e-8 if table else 1e-5)
    else:
        local=row['risk'] <= ETA+(1e-11 if table else 1e-8)
    return bool(local and struct<=1e-8)
