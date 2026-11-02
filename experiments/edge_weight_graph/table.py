"""Full distance-matrix optimization with feasible witnesses and global outer bounds."""
import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, linprog, minimize
from graph import A,B,G,SRC,DST,ETA,NEAR,TIE,cases,evaluate,feasible,graph,save,shortest,tag,write_rows


def analytic(cost,w,soft):
    """Upper bound by splitting s's unit of query flow; matching enlarged-edge metric."""
    base=shortest(cost)
    if not soft:return base,base[:G,G].mean(),0.,np.zeros(7)
    slope=np.array([1,1,1,-1,-1,-1,0])/5
    def counts(p):return np.array([p,1+p,2+p,1-p,2-p,3-p,0])/5
    def extension(p):
        f=counts(p)
        return np.sqrt(ETA)*f/w/np.linalg.norm(f/np.sqrt(w))
    def derivative(p):return slope@(cost+extension(p))
    if derivative(0)>=0:p=0.
    elif derivative(1)<=0:p=1.
    else:p=brentq(derivative,0,1,xtol=1e-15)
    f=counts(p);e=extension(p)
    bound=f@cost+np.sqrt(ETA)*np.linalg.norm(f/np.sqrt(w))
    d=shortest(cost+e)
    assert abs(d[:G,G].mean()-bound)<1e-11
    return d,bound,p,e*np.sqrt(w/ETA)


class MatrixProblem:
    def __init__(self,delta,q):
        self.cost,self.w=graph(delta,q)
        self.soft=q is not None
        self.truth=shortest(self.cost)
        self.cert,self.opt,self.p,self.direction=analytic(self.cost,self.w,self.soft)
        self.n=43 if self.soft else 36
        basis=np.eye(self.n)
        unit=basis[:36].reshape(6,6,self.n)
        self.F=unit[:G,G].mean(axis=0)
        self.m=unit[B,G]-unit[A,G]
        self.scale=np.sqrt(ETA/self.w)
        # Bounds are redundant on the true feasible set: a simple shortest path
        # has at most 5 edges, each at most 3+sqrt(eta/min(w)) <= 13.
        self.caps=np.r_[np.full(36,65.),np.ones(self.n-36)]
        self.caps[np.arange(6)*7]=0
        rows=[];rhs=[]
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    if len({i,j,k})==3:
                        rows.append(unit[i,k]-unit[i,j]-unit[j,k]);rhs.append(0.)
        for e,(i,j) in enumerate(zip(SRC,DST)):
            row=unit[i,j].copy()
            if self.soft:row[36+e]=-self.scale[e]
            rows.append(row);rhs.append(self.cost[e])
        self.rows=np.array(rows);self.rhs=np.array(rhs)

    def pack(self,d):
        if not self.soft:return d.ravel()
        return np.r_[d.ravel(),np.maximum(d[SRC,DST]-self.cost,0)/self.scale]

    def repair(self,x,with_floor):
        """Close distances downward, then restore ball and floor by convex mixing."""
        d=np.maximum(0,x[:36].reshape(6,6)).copy()
        np.fill_diagonal(d,0)
        if not self.soft:d[SRC,DST]=np.minimum(d[SRC,DST],self.cost)
        for k in range(6):d=np.minimum(d,d[:,k,None]+d[None,k,:])
        if self.soft:
            norm=np.linalg.norm(self.pack(d)[36:])
            if norm>1:d=self.truth+(d-self.truth)*((1-4e-15)/norm)
        if with_floor:
            floor=self.opt-NEAR+2e-13
            achieved=d[:G,G].mean()
            if achieved<floor:
                mix=min(1.,(floor-achieved)/(self.cert[:G,G].mean()-achieved))
                d=d+(self.cert-d)*mix
        return d

    def solve(self,obj,with_floor=False):
        mat,rhs=self.rows,self.rhs
        if with_floor:
            mat=np.vstack((mat,-self.F))
            # Relax by 2e-13 to account for rounding of the independent optimum.
            rhs=np.r_[rhs,-self.opt+NEAR+2e-13]
        if not self.soft:
            return self.lp(obj,mat,rhs,with_floor)
        x0=self.pack(self.cert if with_floor else self.truth*.9)
        def ball_jac(x):return np.r_[np.zeros(36),-2*x[36:]]
        constraints=[dict(type='ineq',fun=lambda x: rhs-mat@x,jac=lambda x:-mat),
                     dict(type='ineq',fun=lambda x: 1-x[36:]@x[36:],jac=ball_jac)]
        sol=minimize(lambda x: -obj@x,x0,jac=lambda x:-obj,method='SLSQP',
                     bounds=list(zip(np.zeros(self.n),self.caps)),constraints=constraints,
                     options=dict(maxiter=1200,ftol=1e-13))
        d=self.repair(sol.x,with_floor)
        # The analytic primal is also a feasible candidate.
        candidates=[d,self.cert]
        d=max(candidates,key=lambda z: obj@self.pack(z))
        best=float(obj@self.pack(d))
        def plane(y):
            norm=np.linalg.norm(y)
            return np.r_[np.zeros(36),y/max(norm,1e-30)]
        cuts=[plane(self.direction),plane(sol.x[36:])]
        # If an LP is ill-conditioned on the thin objective slab, retaining its
        # previous valid upper bound is safe; only new certified bounds tighten it.
        upper=np.inf
        for it in range(1200):
            cm=np.vstack((mat,cuts));cb=np.r_[rhs,np.full(len(cuts),1+1e-14)]
            res=linprog(-obj,A_ub=cm,b_ub=cb,bounds=list(zip(np.zeros(self.n),self.caps)),
                        method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
            if not res.success:
                raise RuntimeError(f'Outer LP failed at cut {it}: {res.message}')
            upper=min(upper,self.dual_bound(res,obj,cm,cb))
            if upper-best <= (1e-7 if not with_floor else 2e-5):break
            proposal=self.repair(res.x,with_floor)
            val=obj@self.pack(proposal)
            if val>best:d,best=proposal,val
            cuts.append(plane(res.x[36:]))
        if upper+2e-11<best:raise ArithmeticError((best,upper))
        return d,upper,dict(iterations=it+1,inner_status=str(sol.message),outer_minus_inner=upper-best)

    def lp(self,obj,mat,rhs,with_floor):
        sol=linprog(-obj,A_ub=mat,b_ub=rhs,bounds=list(zip(np.zeros(self.n),self.caps)),
                    method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
        assert sol.success,sol.message
        d=self.repair(sol.x,with_floor)
        upper=self.dual_bound(sol,obj,mat,rhs)
        return d,upper,dict(iterations=sol.nit,inner_status=sol.message,
                            outer_minus_inner=upper-obj@self.pack(d))

    def dual_bound(self,sol,obj,mat,rhs):
        # For any y>=0, z.x <= b.y + ub.[z-A'y]_+ on Ax<=b, 0<=x<=ub.
        # The final pad covers floating dot/reduction roundoff (even without
        # extended precision); it does not rely on HiGHS's success flag alone.
        y=np.maximum(0,-sol.ineqlin.marginals).astype(np.longdouble)
        a=mat.astype(np.longdouble);b=rhs.astype(np.longdouble)
        correction=self.caps@np.maximum(obj-a.T@y,0)
        scale=1+abs(b)@y+self.caps@(abs(obj)+abs(a).T@y)
        pad=(2*len(y)+16)*np.finfo(np.longdouble).eps*scale+1e-12
        return float(b@y+correction+pad)


def run(out):
    folder=out/'table';folder.mkdir(exist_ok=False)
    rows=[];ends=[]
    for delta,q in cases():
        p=MatrixProblem(delta,q);arm=tag(delta,q)
        d,ub,info=p.solve(p.F)
        # Two independent upper bounds: path certificate and matrix LP dual.
        f_upper=min(ub,p.opt+2e-13)
        row=evaluate(d,delta,q)
        assert feasible(row,table=True) and -1e-10<=f_upper-row['F']<=1e-7
        if q is None:np.testing.assert_allclose(d[:,G],p.truth[:,G],atol=1e-8,rtol=0)
        lo,neg_lo,lo_info=p.solve(-p.m,True)
        hi,pos_hi,hi_info=p.solve(p.m,True)
        pair=[evaluate(x,delta,q) for x in (lo,hi)]
        for label,r in zip(['min','max'],pair):
            assert feasible(r,table=True) and r['F']>=p.opt-NEAR-2e-12
            ends.append(dict(arm=arm,endpoint=label,**r))
        mlo,mhi=[r['m'] for r in pair]
        assert -neg_lo<=mlo+1e-10 and mhi<=pos_hi+1e-10
        assert max(mlo+neg_lo,pos_hi-mhi)<=2e-5,(arm,mlo+neg_lo,pos_hi-mhi)
        kind=('positive' if -neg_lo>TIE else 'negative' if pos_hi < -TIE else
              'mixed' if mlo < -TIE and mhi>TIE else 'tie_or_unresolved')
        rows.append(dict(arm=arm,**row,feasible=True,F_upper=f_upper,F_path_upper=p.opt+2e-13,
                         F_matrix_upper=ub,primal_gap=f_upper-row['F'],p_on_a=p.p if q is not None else None,
                         m_min_outer=-neg_lo,m_min_witness=mlo,m_max_witness=mhi,m_max_outer=pos_hi,
                         interval_class=kind,Dijkstra_J_error=float(np.abs(d[:,G]-p.truth[:,G]).max())))
        np.savez(folder/(arm+'.npz'),D=d,min_matrix=lo,max_matrix=hi,truth=p.truth,
                 analytic=p.cert,cost=p.cost,weight=p.w)
        save(folder/(arm+'.json'),dict(max_F=info,min_m=lo_info,max_m=hi_info))
        print(arm,round(mlo,6),round(mhi,6),kind,flush=True)
        write_rows(out/'table.csv',rows);write_rows(out/'interval_witnesses.csv',ends)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('out',type=Path)
    run(ap.parse_args().out)
