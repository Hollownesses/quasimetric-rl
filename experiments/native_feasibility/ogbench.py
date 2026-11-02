"""Full empirical official mean constraint, retaining the original IQE class."""
import argparse
import copy
import json
from pathlib import Path
import sys
import numpy as np

def run(raw,out):
    out.mkdir(parents=True,exist_ok=False);repo=Path(__file__).resolve().parents[2]
    sys.path[:0]=[str(raw/'experiments/ogbench_critic'),str(raw),str(repo)]
    from common import load_base,load_data,distances,tree_equal,tree_digest
    from exact_score_cache import ExactScoreCache
    from experiments.ogbench_diagnostics import core
    from experiments.ogbench_diagnostics.fast_pointmaze import FastPointmaze
    from field import graph_metrics,GOAL,ANCHOR
    from experiments.native_feasibility.core import mean_scale,risk
    from experiments.native_margin.shared import rows,write,sha
    import jax
    agent,cfg=load_base();original_digest=tree_digest(agent);env,train,_=load_data(cfg)
    ii=np.flatnonzero(train.dataset['valids']);obs=train.dataset['observations'];assert len(ii)==1000000
    def score(a,x,y):
        x=np.asarray(x,np.float32);y=np.broadcast_to(y,x.shape).astype(np.float32);n=len(x)
        xx,yy=(np.pad(z,((0,(-n)%1024),(0,0)),mode='edge') for z in (x,y))
        return np.concatenate([np.asarray(distances(a,xx[k:k+1024],yy[k:k+1024])) for k in range(0,n,1024)])[:n].astype(float)
    d0=score(agent,obs[ii],obs[ii+1]);print('OG full native risk',risk(d0),flush=True)
    lo,hi,alpha=mean_scale(d0,cfg['eps'])
    params=copy.deepcopy(agent.network.params);phi=params['modules_value']['phi']
    last=max((k for k in phi if k.startswith('Dense_')),key=lambda x:int(x.split('_')[-1]))
    phi[last]={k:v*alpha for k,v in phi[last].items()}
    projected=agent.replace(network=agent.network.replace(params=params))
    for key in params:
        if key!='modules_value':assert tree_equal(params[key],agent.network.params[key])
    assert tree_equal(projected.network.opt_state,agent.network.opt_state) and tree_equal(projected.rng,agent.rng)
    assert jax.tree_util.tree_structure(agent)==jax.tree_util.tree_structure(projected)
    assert all(x.shape==y.shape for x,y in zip(jax.tree_util.tree_leaves(agent),jax.tree_util.tree_leaves(projected)))
    d1=d0 if alpha==1 else score(projected,obs[ii],obs[ii+1])
    new_risk=risk(d1);assert new_risk<=cfg['eps']+1e-6
    np.testing.assert_allclose(d1,alpha*d0,rtol=2e-4,atol=2e-5)
    np.savez_compressed(out/'local.npz',distance=d0,projected=d1)
    # Only now load geometric references and evaluation inputs.
    engine=FastPointmaze(env,nthread=4);g=raw/'results/ogbench_critic_20261029'
    bank=core.read_bank(g/'official_counterfactual/official_task3_bank.npz')
    history=json.loads((g/'official_counterfactual_completion/analysis.json').read_text())['initials']['original']
    walls,r=core.geometry(env);nodes,graph=core.build_graph(walls,r,.2)
    oldfield,_=graph_metrics(nodes,graph,score(agent,nodes,GOAL),goal=GOAL,anchor=ANCHOR)
    newfield,_=graph_metrics(nodes,graph,score(projected,nodes,GOAL),goal=GOAL,anchor=ANCHOR,tol=alpha*1e-6)
    first=[]
    for name,case in bank.items():
        core.load_case(env,case);xy,_=engine.candidates(env)
        a=np.asarray(distances(agent,xy,np.broadcast_to(case['goal'],xy.shape))).astype(float)
        b=np.asarray(distances(projected,xy,np.broadcast_to(case['goal'],xy.shape))).astype(float)
        first.append(dict(case=name,parent_action=int(np.argmin(a)),projected_action=int(np.argmin(b)),
                          algebraic_action=int(np.argmin(alpha*a)),max_error=float(abs(b-alpha*a).max())))
    episodes=[];cache=None
    if alpha==1:
        assert tree_equal(agent,projected)
        episodes=[dict(case=k,success=int(v['success']),parent_success=int(v['success']),steps=v['steps'],reused=1) for k,v in history.items()]
    else:
        cache=ExactScoreCache(projected,audit=True);folder=out/'greedy';folder.mkdir()
        for name,case in bank.items():
            info,tr=engine.rollout(case,cache)
            tr['final_integration']=core.save_integration(env)
            np.savez_compressed(folder/(name+'.npz'),**tr)
            episodes.append(dict(case=name,success=int(info['success']),parent_success=int(history[name]['success']),steps=info['steps'],reused=0))
            if len(episodes)%5==0:print('feasible OG',len(episodes),flush=True)
            rows(out/'episodes.csv',episodes)
    rows(out/'episodes.csv',episodes);rows(out/'first.csv',first)
    engine.close();env.close();assert tree_digest(agent)==original_digest
    write(out/'analysis.json',dict(scene='OG',training_records=len(ii),eps=cfg['eps'],old_risk=risk(d0),
        alpha_lower=lo,alpha_upper=hi,alpha=alpha,new_risk=new_risk,old_max_h=float(d0.max()-1),new_max_h=float(d1.max()-1),
        homogeneity_max_error=float(abs(d1-alpha*d0).max()),last_phi_layer=last,shape_and_other_modules_unchanged=True,
        optimizer_rng_unchanged=True,parent_agent_unchanged=True,oldfield=oldfield,newfield=newfield,
        successes=sum(r['success'] for r in episodes),episodes=50,reused_parent_episodes=bool(alpha==1),
        first_argmin_changed=sum(r['parent_action']!=r['projected_action'] for r in first),
        cache=cache.statistics() if cache else None,sources={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('out',type=Path);a=p.parse_args();run(a.raw.resolve(),a.out)
