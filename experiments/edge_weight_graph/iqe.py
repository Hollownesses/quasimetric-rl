"""Finite-budget full-batch optimization. No table or truth enters the primal loss."""
import argparse
import hashlib
import math
from pathlib import Path

import numpy as np
import torch
import torchqmet
from scipy.optimize import minimize
from graph import A,B,G,SRC,DST,ETA,NEAR,cases,evaluate,feasible,graph,save,tag,write_rows

STEPS=12000
SRC_T=torch.tensor(SRC);DST_T=torch.tensor(DST)


class NodeIQE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(606)
        self.z=torch.nn.Parameter(torch.randn(6,64,dtype=torch.float64)*.1)
        self.head=torchqmet.IQE(64,8).double()

    def full(self):
        return self.head(self.z[:,None,:],self.z[None,:,:])


def rate(i):
    return .01 if i<8000 else .01*math.exp(math.log(.001)*(i-8000)/3999)


def digest(model):
    return hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in model.state_dict().values())).hexdigest()


def train(delta,q,folder,target=None):
    model=NodeIQE()
    init=digest(model)
    optimizer=torch.optim.Adam(model.parameters(),lr=.01,foreach=False)
    cost,w=map(torch.from_numpy,graph(delta,q))
    dual=torch.zeros(7 if q is None else 1,dtype=torch.float64)
    rho=20 if q is None else 1
    steps=[]
    def residual(d):
        h=d[SRC_T,DST_T]-cost
        if q is None: return h
        return (torch.linalg.vector_norm(w.sqrt()*h.relu())/math.sqrt(ETA)-1).reshape(1)
    for i in range(STEPS):
        optimizer.param_groups[0]['lr']=rate(i)
        optimizer.zero_grad()
        d=model.full()
        if target is None:
            r=residual(d)
            loss=-d[:G,G].mean()+(((dual+rho*r).relu().square()-dual.square())/(2*rho)).sum()
        else:
            loss=(d-target).square().mean()
        loss.backward()
        optimizer.step()
        if (i+1)%100==0:
            with torch.no_grad():
                d=model.full()
                if target is None:
                    dual=(dual+rho*residual(d)).relu()
                steps.append(dict(step=i+1,F=d[:G,G].mean().item(),m=(d[B,G]-d[A,G]).item(),
                                  max_r=residual(d).max().item(),dual=dual.tolist()))
    fit={}
    if target is not None:
        params=list(model.parameters())
        def unpack(x):
            with torch.no_grad():
                torch.nn.utils.vector_to_parameters(torch.tensor(x,dtype=torch.float64),params)
        def mse(x):
            unpack(x)
            loss=(model.full()-target).square().mean()
            grad=torch.autograd.grad(loss,params)
            return loss.item(),torch.cat([g.flatten() for g in grad]).numpy()
        res=minimize(mse,torch.nn.utils.parameters_to_vector(params).detach().numpy(),method='L-BFGS-B',
                     jac=True,options=dict(maxiter=1500,ftol=1e-15,gtol=1e-12))
        unpack(res.x)
        fit=dict(lbfgs_message=str(res.message),lbfgs_iterations=res.nit,
                 max_fit_error=(model.full()-target).detach().abs().max().item(),mse=res.fun)
    d=model.full().detach().numpy()
    row=evaluate(d,delta,q)
    arm=tag(delta,q)
    np.savez(folder/(arm+'.npz'),D=d,**{k:v.detach().numpy() for k,v in model.state_dict().items()})
    save(folder/(arm+'.json'),dict(initial_sha256=init,final_sha256=digest(model),logs=steps,fit=fit))
    return dict(arm=arm,**row,feasible=feasible(row,table=False),initial_sha256=init,**fit)


def run(out,witness):
    import csv
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    folder=out/('witness' if witness else 'iqe')
    folder.mkdir(exist_ok=False)
    rows=[]
    if witness:
        with (out/'table.csv').open() as f: tables={r['arm']:r for r in csv.DictReader(f)}
        with (out/'iqe.csv').open() as f: iqes={r['arm']:r for r in csv.DictReader(f)}
    for delta,q in cases():
        arm=tag(delta,q)
        target=None
        if witness:
            if tables[arm]['interval_class']!='positive' or iqes[arm]['action']!='b':continue
            target=torch.from_numpy(np.load(out/'table'/(arm+'.npz'))['D'])
        row=train(delta,q,folder,target)
        if witness:
            row['witness']=bool(row['feasible'] and row['action']=='a' and row['F']>=float(tables[arm]['F_upper'])-NEAR)
        rows.append(row)
        print(folder.name,arm,'m=',round(row['m'],6),'feasible=',row['feasible'],flush=True)
        write_rows(out/(folder.name+'.csv'),rows)
    if not rows: (out/'witness.csv').write_text('arm,witness\n')
    assert len({r['initial_sha256'] for r in rows})<=1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);p.add_argument('--witness',action='store_true')
    args=p.parse_args();run(args.out,args.witness)
