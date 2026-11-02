"""Read-only resolution sensitivity at the fifty common official starts."""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from metrics import path_metrics, ranking, write_rows
from pointmaze_analysis import ARMS


def run(root,out):
    sys.path[:0]=[str(root/'experiments/ogbench_critic'),str(root)]
    from common import load_data,load_base,distances
    from utils.flax_utils import restore_agent
    from experiments.ogbench_diagnostics import core
    from experiments.ogbench_diagnostics.fast_pointmaze import FastPointmaze
    base,config=load_base()
    env,_,_=load_data(config)
    engine=FastPointmaze(env,nthread=4)
    walls,r=core.geometry(env)
    grids=[core.build_graph(walls,r,spacing) for spacing in [.2,.1]]
    trees=[cKDTree(nodes) for nodes,_ in grids]
    raw=root/'results/ogbench_critic_20261029'
    bank=core.read_bank(raw/'official_counterfactual/official_task3_bank.npz')
    rows=[];attribution=[];ii,jj=np.triu_indices(121,1)
    probes=[]
    for name,case in bank.items():
        core.load_case(env,case)
        xy,_=engine.candidates(env)
        goal=case['goal']
        refs=[];snaps=[]
        for tree,(nodes,graph) in zip(trees,grids):
            targets=np.flatnonzero(np.linalg.norm(nodes-goal,axis=1)<=1)
            v=dijkstra(graph,indices=targets,min_only=True,unweighted=True)
            gap,idx=tree.query(xy)
            refs.append(np.where(np.linalg.norm(xy-goal,axis=1)<=1,0,v[idx]))
            snaps.append(gap.max())
        # Keep the coarse field, with ties on shortest routes broken by node ID.
        nodes,graph=grids[0]
        targets=np.flatnonzero(np.linalg.norm(nodes-goal,axis=1)<=1)
        field=dijkstra(graph,indices=targets,min_only=True,unweighted=True)
        probes.append((name,xy,goal,refs,snaps,field))
    engine.close();env.close()
    for model,cp,*_ in ARMS:
        agent=base if cp is None else restore_agent(base,str(raw/cp),101000)
        def score(x,y):
            n=len(x)
            xx,yy=(np.pad(np.broadcast_to(z,x.shape).astype(np.float32),((0,(-n)%256),(0,0)),mode='edge') for z in (x,y))
            return np.concatenate([np.asarray(distances(agent,xx[i:i+256],yy[i:i+256])) for i in range(0,n,256)])[:n].astype(float)
        for name,xy,goal,(v,w),snaps,field in probes:
            # Match the main diagnostic's fixed batch shape.
            j=score(xy,goal)
            rv,rw=v[ii]-v[jj],w[ii]-w[jj]
            stable=rv*rw>0
            reverse=stable & (np.sign(rv)*(j[ii]-j[jj]) < -1e-4)
            a,b=(ranking(np.ones(121),j,x) for x in (v,w))
            rows.append(dict(model=model,case=name,coarse_pairs=a['strict_pairs'],fine_pairs=b['strict_pairs'],
                             coarse_reversed=a['reversed_pairs'],fine_reversed=b['reversed_pairs'],
                             same_order_pairs=int(stable.sum()),same_order_reversed=int(reverse.sum()),
                             opposite_order_pairs=int((rv*rw<0).sum()),
                             coarse_wrong=a['wrong'],fine_wrong=b['wrong'],max_snap_coarse=snaps[0],max_snap_fine=snaps[1]))
            if a['wrong']:
                terms=[]
                nodes,graph=grids[0]
                for idx in (np.argmin(j),np.argmin(v)):
                    node=int(trees[0].query(xy[idx])[1]);route=[node]
                    assert field[node]==v[idx] and np.linalg.norm(xy[idx]-goal)>1
                    while field[node]>0:
                        ns=graph.indices[graph.indptr[node]:graph.indptr[node+1]]
                        node=int(ns[field[ns]==field[node]-1].min())
                        route.append(node)
                    xyz=nodes[route]
                    js=score(xyz,goal)
                    assert len(route)-1==v[idx] and len(route)>1
                    path=path_metrics(np.ones(len(route)-1),score(xyz[:-1],xyz[1:]),js)
                    terms.append(np.array([j[idx]-js[0],js[-1],path['sum_h'],path['sum_tau']]))
                snap,terminal,dh,dt=terms[0]-terms[1]
                assert abs(snap+terminal+dh-dt-a['chosen_error_difference'])<1e-7
                attribution.append(dict(model=model,case=name,delta_snap=snap,delta_terminal=terminal,
                                        delta_sum_h=dh,delta_sum_tau=dt,delta_error=a['chosen_error_difference'],
                                        gap=a['reference_margin'],model_gap=a['chosen_model_gap']))
    write_rows(out/'resolution.csv',rows)
    write_rows(out/'og_grid_attribution.csv',attribution)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('out',type=Path)
    args=p.parse_args();run(args.raw.resolve(),args.out)
