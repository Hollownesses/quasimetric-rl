"""Read-only geometry ranking and true-successor closed-loop evaluation."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import dijkstra

def run(raw,out):
    repo=Path(__file__).resolve().parents[2]
    sys.path[:0]=[str(raw/'experiments/ogbench_critic'),str(raw),str(repo/'experiments/native_margin'),str(repo/'experiments/unified_mechanism')]
    from common import load_base,load_data,distances,tree_digest
    from utils.flax_utils import restore_agent
    from exact_score_cache import ExactScoreCache
    from field import graph_metrics,GOAL,ANCHOR
    from experiments.ogbench_diagnostics import core
    from experiments.ogbench_diagnostics.fast_pointmaze import FastPointmaze
    from shared import rows,write,sha
    from metrics import ranking,stats
    agent,cfg=load_base();env,_,val=load_data(cfg);engine=FastPointmaze(env,nthread=4)
    g=raw/'results/ogbench_critic_20261029';bank=core.read_bank(g/'official_counterfactual/official_task3_bank.npz')
    hist=json.loads((g/'official_counterfactual_completion/analysis.json').read_text())['initials']['original']
    source=repo/'results/native_margin_20261102_verified/og';pairs=np.load(source/'pairs.npz')
    probes=list(csv.DictReader((source/'probes.csv').open()))
    points=pairs['points'].reshape(-1,2);goals=pairs['goals']
    walls,r=core.geometry(env);nodes,graph=core.build_graph(walls,r,.2);tree=cKDTree(nodes)
    frozen={}
    for name,case in bank.items():
        core.load_case(env,case);xy,before=engine.candidates(env)
        targets=np.flatnonzero(np.linalg.norm(nodes-case['goal'],axis=1)<=1)
        v=dijkstra(graph,indices=targets,min_only=True,unweighted=True)
        snap,idx=tree.query(xy);ref=np.where(np.linalg.norm(xy-case['goal'],axis=1)<=1,0,v[idx])
        frozen[name]=(xy,ref)
    margins=[];local=[];decisions=[];episodes=[];audits={};fields=[]
    for arm in ['parent','native','no_push','no_local']:
        a=agent if arm=='parent' else restore_agent(agent,str(out/arm),100512);digest=tree_digest(a)
        def score(x,y):
            n=len(x);x=np.asarray(x,np.float32);y=np.broadcast_to(y,x.shape).astype(np.float32)
            xx,yy=(np.pad(z,((0,(-n)%256),(0,0)),mode='edge') for z in (x,y))
            return np.concatenate([np.asarray(distances(a,xx[i:i+256],yy[i:i+256])) for i in range(0,n,256)])[:n].astype(float)
        vv=np.asarray(distances(a,points,goals)).reshape(-1,2).astype(float)
        for i,p in enumerate(probes):
            m=vv[i,1]-vv[i,0]
            if arm=='parent':np.testing.assert_allclose(m,float(p['margin']),atol=1e-5,rtol=0)
            margins.append(dict(arm=arm,case=p['case'],margin=m,parent_margin=float(p['margin']),ref_gap=p['ref_gap']))
        ii=np.flatnonzero(val.dataset['valids']);obs=val.dataset['observations']
        h=score(obs[ii],obs[ii+1])-1;local.append(dict(arm=arm,scope='validation',**stats(h)))
        field,fielddata=graph_metrics(nodes,graph,score(nodes,GOAL),goal=GOAL,anchor=ANCHOR)
        fields.append(dict(arm=arm,**field))
        for name,case in bank.items():
            xy,ref=frozen[name];pred=score(xy,case['goal'])
            decisions.append(dict(arm=arm,case=name,**ranking(np.ones(121),pred,ref)))
        if arm=='parent':
            episodes.extend(dict(arm=arm,case=k,success=int(v['success']),steps=v['steps']) for k,v in hist.items());continue
        folder=out/arm/'greedy';folder.mkdir();cache=ExactScoreCache(a,audit=True)
        for name,case in bank.items():
            info,trace=engine.rollout(case,cache)
            trace['final_integration']=core.save_integration(env)
            np.testing.assert_array_equal(trace['observations'][0],case['observation'])
            np.testing.assert_array_equal(trace['goal'],case['goal'])
            assert bool(info['success'])==bool(np.linalg.norm(trace['observations'][-1]-case['goal'])<=1)
            assert len(trace['actions'])<=1000
            np.savez_compressed(folder/(name+'.npz'),**trace)
            episodes.append(dict(arm=arm,case=name,success=int(info['success']),steps=info['steps']))
            rows(out/'episodes.csv',episodes)
            if len(episodes)%5==0:print('OG eval',arm,name,flush=True)
        assert tree_digest(a)==digest;audits[arm]=cache.statistics()
    engine.close();env.close()
    rows(out/'margins.csv',margins);rows(out/'local.csv',local);rows(out/'decisions.csv',decisions)
    rows(out/'fields.csv',fields)
    write(out/'evaluation.json',dict(cases=50,actions=121,cache=audits,
         parent_counterfactual_sha256=sha(g/'official_counterfactual_completion/analysis.json'),parent_successes=sum(v['success'] for v in hist.values())))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('out',type=Path);a=p.parse_args();run(a.raw.resolve(),a.out)
