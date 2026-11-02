"""Pinned official joint loss/Adam pressure on true MuJoCo successors."""
import argparse
from functools import partial
import hashlib
from pathlib import Path
import sys
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import dijkstra
from shared import rows, select_pair, sha, source_weights, verify_inputs, write

def run(raw,out):
    out.mkdir(parents=True,exist_ok=False); ids=verify_inputs(raw)
    sys.path[:0]=[str(raw/'experiments/ogbench_critic'),str(raw)]
    from common import load_base,load_data,tree_digest
    from critic import subset
    from mechanism import part_gradient
    from full_gradient import gradient,assert_gradient_equal
    from experiments.ogbench_diagnostics import core
    from experiments.ogbench_diagnostics.fast_pointmaze import FastPointmaze
    import jax
    import jax.numpy as jnp
    import optax
    agent,config=load_base();digest=tree_digest(agent);env,train,_=load_data(config)
    engine=FastPointmaze(env,nthread=4);walls,r=core.geometry(env)
    grids=[core.build_graph(walls,r,h) for h in [.2,.1]];trees=[cKDTree(x) for x,_ in grids]
    bank=core.read_bank(raw/'results/ogbench_critic_20261029/official_counterfactual/official_task3_bank.npz')
    physical=np.flatnonzero(train.dataset['valids']);obs=train.dataset['observations'][physical]
    weights,inverse,count=source_weights(np.floor(obs/.2+.5).astype(np.int64))
    key_to_count={tuple(k):int(c) for k,c in zip(np.unique(np.floor(obs/.2+.5).astype(np.int64),axis=0),count)}
    trtree=cKDTree(obs);probes=[];points=[];goals=[];excluded=[]
    for name,case in bank.items():
        core.load_case(env,case);xy,_=engine.candidates(env);refs=[]
        for tree,(nodes,graph) in zip(trees,grids):
            targets=np.flatnonzero(np.linalg.norm(nodes-case['goal'],axis=1)<=1)
            v=dijkstra(graph,indices=targets,min_only=True,unweighted=True)
            refs.append(np.where(np.linalg.norm(xy-case['goal'],axis=1)<=1,0,v[tree.query(xy)[1]]))
        ab=select_pair(refs[0],refs[1])
        if ab is None:excluded.append(name);continue
        a,b=ab;points.append(xy[[a,b]]);goals.append(case['goal'])
        rec=dict(case=name,good_action=a,bad_action=b,ref_gap=float(refs[0][b]-refs[0][a]),fine_gap=float(refs[1][b]-refs[1][a]))
        for label,idx in [('good',a),('bad',b)]:
            rec[label+'_nearest']=float(trtree.query(xy[idx])[0]);rec[label+'_source_records']=len(trtree.query_ball_point(xy[idx],.2))
        probes.append(rec)
    engine.close();env.close()
    points=np.asarray(points,np.float32);goals=np.repeat(np.asarray(goals,np.float32),2,axis=0)
    np.savez_compressed(out/'pairs.npz',points=points,goals=goals)
    params=subset(agent.network.params)
    @jax.jit
    def margins(p):
        j=agent.network.select('value')(points.reshape(-1,2),goals,params={**agent.network.params,**p}).reshape(-1,2)
        return j[:,1]-j[:,0]
    # JVP avoids a 50 x parameter-count Jacobian allocation.
    @jax.jit
    def projection(g):return -config['lr']*jax.jvp(margins,(params,),(g,))[1]
    @jax.jit
    def linear(delta):return jax.jvp(margins,(params,),(delta,))[1]
    @jax.jit
    def equal_gradient(batch,w):
        def loss(p):
            h=agent.network.select('value')(batch['observations'],batch['next_observations'],params={**agent.network.params,**p})-1
            return agent.network.select('lam')()*jnp.mean(w*jax.nn.relu(h)**2)
        return jax.grad(loss)(params)
    base=np.asarray(margins(params)).astype(float)
    zero_grad=jax.tree_util.tree_map(jnp.zeros_like,agent.network.params)
    dz,_=agent.network.tx.update(zero_grad,agent.network.opt_state,agent.network.params)
    zp=optax.apply_updates(agent.network.params,dz)
    zero=np.asarray(margins(subset(zp))).astype(float)-base
    np.random.seed(4400);stream=hashlib.sha256();output=[];sampleobs=[];nactive=0;max_linear=0.;actor_norm=0.
    for i in range(32):
        batch=train.sample(1024);sampleobs.append(batch['observations'])
        for key in ['observations','next_observations','value_goals']:stream.update(batch[key].tobytes())
        w=np.asarray([len(physical)/(len(count)*key_to_count[tuple(k)]) for k in np.floor(batch['observations']/.2+.5).astype(np.int64)],np.float32)
        gg={k:part_gradient(agent,params,batch,part) for k,part in [('push','push'),('local','pair')]}
        gg['local_equal']=equal_gradient(batch,w)
        gg['dynamics']=gradient(agent,params,batch,'dynamics')
        actor=gradient(agent,params,batch,'actor');total=gradient(agent,params,batch,'total')
        value=gradient(agent,params,batch,'value')
        norm=max(float(np.max(abs(np.asarray(x)))) for x in jax.tree_util.tree_leaves(actor['modules_value']))
        actor_norm=max(actor_norm,norm);assert norm==0
        summed=jax.tree_util.tree_map(lambda a,b,c:a+b+c,gg['push'],gg['local'],gg['dynamics'])
        assert_gradient_equal({'modules_value':summed['modules_value']},{'modules_value':total['modules_value']})
        fullsum=jax.tree_util.tree_map(lambda a,b,c:a+b+c,value,gg['dynamics'],actor)
        assert_gradient_equal(fullsum,total)
        gg['total']=total
        pp={k:np.asarray(projection(g)).astype(float) for k,g in gg.items()}
        new,info=agent.update(batch)
        delta=jax.tree_util.tree_map(lambda a,b:a-b,subset(new.network.params),params)
        exact=np.asarray(margins(subset(new.network.params))).astype(float)-base
        lin=np.asarray(linear(delta)).astype(float)
        max_linear=max(max_linear,float(abs(exact-lin).max()))
        nactive+=int(np.sum(np.asarray(agent.network.select('value')(batch['observations'],batch['next_observations']))>1))
        for k,p in enumerate(probes):
            output.append(dict(case=p['case'],batch=i,**{key:float(val[k]) for key,val in pp.items()},adam=float(exact[k]),adam_linear=float(lin[k])))
        print('OG batch',i,flush=True)
    hist=raw/'results/ogbench_critic_20261029/mechanism_gradient/analysis.json'
    import json
    assert stream.hexdigest()==json.loads(hist.read_text())['paired_batch_stream_sha256']
    sampled=cKDTree(np.concatenate(sampleobs))
    for i,p in enumerate(probes):
        p.update(margin=float(base[i]),zero_adam=float(zero[i]))
        for label,xy in zip(['good','bad'],points[i]):p[label+'_sample_visits']=len(sampled.query_ball_point(xy,.2))
    assert tree_digest(agent)==digest
    for path,d in ids.items():assert sha(raw/path)==d
    rows(out/'probes.csv',probes);rows(out/'pressure.csv',output)
    write(out/'audit.json',dict(inputs=ids,lr=config['lr'],probes=len(probes),excluded=excluded,
           batch_rows=32768,stream_sha256=stream.hexdigest(),historical_stream_exact=True,
           local_active_fraction=nactive/32768,actor_value_gradient_max=actor_norm,
           adam_linear_max_error=max_linear,agent_full_state_unchanged=True,
           source_weights_mean=float(weights.mean()),source_bins=len(count),
           source_files={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();run(a.raw.resolve(),a.out)
