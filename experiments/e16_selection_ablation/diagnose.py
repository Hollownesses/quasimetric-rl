"""Frozen-input geometry diagnostics. True-candidate rollouts use only the new value."""
import argparse
import csv
import hashlib
from pathlib import Path
import sys

import numpy as np
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from support import G,RAW,REPO,UNIFIED,read,write,check
from common import load_base,load_data,distances,tree_digest
from utils.flax_utils import restore_agent
from experiments.ogbench_diagnostics import core
from experiments.ogbench_diagnostics.fast_pointmaze import FastPointmaze
from exact_score_cache import ExactScoreCache
from run_official_counterfactual import trace_values
sys.path.insert(0,str(REPO/'experiments/unified_mechanism'))
from metrics import stats,path_metrics,ranking,write_rows


def history(name):
    with (UNIFIED/'og'/name).open() as f:return [r for r in csv.DictReader(f) if r['model']=='E16']


def batch_score(agent,x,y):
    x=np.asarray(x,np.float32);y=np.broadcast_to(y,x.shape).astype(np.float32)
    xx,yy=(np.pad(z,((0,(-len(x))%256),(0,0)),mode='edge') for z in (x,y))
    return np.concatenate([np.asarray(distances(agent,xx[i:i+256],yy[i:i+256])) for i in range(0,len(x),256)])[:len(x)].astype(float)


def analyze(out,models,env,val,bank,engine):
    walls,radius=core.geometry(env);nodes,graph=core.build_graph(walls,radius,.2);tree=cKDTree(nodes)
    with np.load(G/'baseline_field/field.npz') as data:
        np.testing.assert_array_equal(data['nodes'],nodes);gridroute=data['path']
    realroute=np.load(G/'baseline_field/reference_replay.npz')['observations']
    local=[];paths=[];decisions=[]
    physical=np.flatnonzero(val.dataset['valids']);assert len(physical)==100000
    xy=val.dataset['observations']
    def path(label,agent,scope,case,xy,goal):
        edges=batch_score(agent,xy[:-1],xy[1:]);j=batch_score(agent,xy,goal)
        info=path_metrics(np.ones(len(edges)),edges,j)
        paths.append(dict(model=label,scope=scope,case=case,**info))
    for label,agent in models.items():
        h=batch_score(agent,xy[physical],xy[physical+1])-1
        local.append(dict(model=label,rms_positive=np.sqrt(np.maximum(h,0).dot(np.maximum(h,0))/len(h)),**stats(h)))
        path(label,agent,'geometry_shortest','canonical',gridroute,core.GOAL)
        path(label,agent,'reference_env','canonical',realroute,core.GOAL)
    old={r['case']:r for r in history('decisions.csv') if r['scope']=='start'}
    for name,case in bank.items():
        core.load_case(env,case);succ,before=engine.candidates(env);goal=case['goal']
        targets=np.flatnonzero(np.linalg.norm(nodes-goal,axis=1)<=1)
        field=dijkstra(graph,indices=targets,min_only=True,unweighted=True)
        snap,idx=tree.query(succ)
        ref=np.where(np.linalg.norm(succ-goal,axis=1)<=1,0,field[idx])
        route=[int(tree.query(case['observation'])[1])]
        while field[route[-1]]>0:
            k=route[-1];nbr=graph.indices[graph.indptr[k]:graph.indptr[k+1]]
            route.append(int(nbr[field[nbr]==field[k]-1].min()))
        fingerprint=hashlib.sha256(succ.tobytes()+ref.tobytes()).hexdigest()
        for label,agent in models.items():
            pred=batch_score(agent,succ,goal);a=int(np.argmin(pred))
            core.restore(env,before);obs,*_=env.step(core.ACTIONS[a]);np.testing.assert_array_equal(obs,succ[a])
            info=ranking(np.ones(121),pred,ref)
            if label=='high':
                for key in ('strict_pairs','reversed_pairs','model_tied_strict','wrong'):assert info[key]==int(old[name][key])
                for key in ('reference_margin','chosen_error_difference','chosen_model_gap'):
                    np.testing.assert_allclose(info[key],float(old[name][key]),rtol=0,atol=1e-7)
            decisions.append(dict(model=label,case=name,action_index=a,snap_max=snap.max(),candidates_reference_sha256=fingerprint,**info))
            path(label,agent,'official_grid',name,nodes[route],goal)
    prior=next(r for r in history('metrics.csv') if r['scope']=='val_records' and r['metric']=='h')
    for k in ['mean','p95','p99','maximum','positive_fraction','positive_mean']:
        np.testing.assert_allclose(local[0][k],float(prior[k]),rtol=0,atol=1e-7)
    for r in history('paths.csv'):
        if r['scope'] not in ['reference_env','geometry_shortest']:continue
        cur=next(p for p in paths if p['model']=='high' and p['scope']==r['scope'])
        for k in ('cost','end','sum_h','sum_tau','tau_over_cost'):
            np.testing.assert_allclose(cur[k],float(r[k]),rtol=0,atol=1e-7)
    write_rows(out/'decisions.csv',decisions);write_rows(out/'paths.csv',paths);write_rows(out/'local.csv',local)
    print('COMMON_INPUTS_VERIFIED',len(old),flush=True)


def run(out):
    checks=read(out/'preflight.json')['paths_sha256']
    for p in [G/'official_counterfactual/official_task3_bank.npz',G/'hard_goal_1000/model/params_101000.pkl']:
        check(p,checks[str(p)])
    check(out/'critic/params_101000.pkl',read(out/'critic.json')['checkpoint']['sha256'])
    base,cfg=load_base()
    models={k:restore_agent(base,str(p),101000) for k,p in [('high',G/'hard_goal_1000/model'),('random',out/'critic')]}
    digests={k:tree_digest(m) for k,m in models.items()}
    env,_,val=load_data(cfg);engine=FastPointmaze(env,nthread=4)
    bank=core.read_bank(G/'official_counterfactual/official_task3_bank.npz');assert len(bank)==50
    analyze(out,models,env,val,bank,engine)
    cache=ExactScoreCache(models['random'],audit=True)
    previous=read(G/'official_counterfactual_completion/analysis.json')['initials']['improved']
    directory=out/'greedy';directory.mkdir(exist_ok=False)
    pairs=[]
    for name,case in bank.items():
        info,trace=engine.rollout(case,cache)
        trace['final_integration']=core.save_integration(env)
        trace['state_values']=trace_values(models['random'],trace['observations'],case['goal'])
        np.testing.assert_array_equal(trace['observations'][0],case['observation'])
        np.testing.assert_array_equal(trace['goal'],case['goal'])
        assert bool(info['success'])==bool(np.linalg.norm(trace['observations'][-1]-case['goal'])<=1)
        np.savez_compressed(directory/(name+'.npz'),**trace)
        pairs.append(dict(case=name,high_success=int(previous[name]['success']),random_success=int(info['success']),
                          high_steps=previous[name]['steps'],random_steps=info['steps']))
        write_rows(out/'greedy_pairs.csv',pairs)
        if len(pairs)%5==0:print('GREEDY',len(pairs),flush=True)
    assert len(pairs)==50 and sum(r['high_success'] for r in pairs)==42
    assert all(tree_digest(m)==digests[k] for k,m in models.items())
    write(out/'diagnostic_audit.json',dict(matched_e16_history=True,cases=50,actions=121,valid_edges=100000,
          spacing=.2,radius=radius_of(env),body_and_goal_radius_distinct=True,cache=cache.statistics(),critic_sha256=digests))
    engine.close();env.close()


def radius_of(env):return float(core.geometry(env)[1])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);run(p.parse_args().out.resolve())
