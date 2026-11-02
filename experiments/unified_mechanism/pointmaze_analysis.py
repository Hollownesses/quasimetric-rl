"""Diagnose saved PointMaze models with the pinned upstream runtime, no updates."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import dijkstra

from metrics import path_metrics, ranking, read, stats, verified, write_rows

ARMS = [
    ('100k', None, 'extraction_1000/original',102000,'397e919f2c453cee02ec840c615dce3532f31532fd639fbea6757c970fedbf0f',
     '0f0c1fc98bb1d4cbba09590f6004a132a012ae221a101feb703ea95e465b87a9'),
    ('E6','tail_1000/mean','mean_extraction_1000/mean',103000,'d92cf664f10345f96a77045f988a689f0985ac2120781ec596ef6b932fdaacec',
     '8219d3c77c62cc3fff7f9960adb9b62a56144d036b3149973d6f6dbb2adfe516'),
    ('E16','hard_goal_1000/model','extraction_1000/improved',103000,'71a8d61900ac5db435e6c71c3786996d779abe81749884c3cc834a557fbb6777',
     'dbec9e4ad128ca519611f40ad4d3a1f02118f92e8a6513afd074c8d1a623978e')]


def run(root, out):
    out.mkdir(parents=True, exist_ok=False)
    sys.path[:0]=[str(root/'experiments/ogbench_critic'),str(root)]
    from common import BASE, load_base, load_data, distances, tree_equal
    from utils.flax_utils import restore_agent
    from experiments.ogbench_diagnostics import core
    from experiments.ogbench_diagnostics.fast_pointmaze import FastPointmaze
    import jax
    import jax.numpy as jnp
    @jax.jit
    def latent(agent,s,g,a):
        value=agent.network.select('value')
        _,z,zg=value(s,g,info=True)
        pred=z+agent.network.select('dynamics')(jnp.concatenate([z,a],axis=-1))
        return value(pred,zg,is_phi=True)
    g=root/'results/ogbench_critic_20261029'
    off=root/'results/ogbench_official_qrl_20261026_seed0'
    ids={}
    def check(p,expected):
        ids[str(p.relative_to(root))]=verified(p,expected)
    check(g/'evaluation_bank.npz','5bb143317adf5c9198d1fd12ee3263a57236973719bbb036ff4dffe55525a95b')
    p=g/'official_counterfactual/official_task3_bank.npz'
    check(p,read(g/'official_counterfactual/protocol.json')['bank_sha256'])
    bank=core.read_bank(p)
    assert len(bank)==50
    for source, label in [('','train'),('-val','validation')]:
        check(root/'experiments/ogbench_official_qrl/data'/f'pointmaze-medium-navigate-v0{source}.npz',
              read(off/'data_audit.json')[label]['sha256'])
    base,config=load_base()
    env,_,val=load_data(config)
    assert env.unwrapped._success_timing=='post' and env.unwrapped._goal_tol==1
    engine=FastPointmaze(env,nthread=4)
    walls,radius=core.geometry(env)
    nodes,graph=core.build_graph(walls,radius,.2)
    tree=cKDTree(nodes)
    field=np.load(g/'baseline_field/field.npz')
    np.testing.assert_array_equal(field['nodes'],nodes)
    gridroute=field['path']
    replay=np.load(g/'baseline_field/reference_replay.npz')['observations']
    data=val.dataset
    physical=np.flatnonzero(data['valids'])
    allobs=data['observations']
    starts=np.r_[0,np.flatnonzero(~data['valids'].astype(bool))[:-1]+1]
    assert len(physical)==100000 and len(starts)==100 and all(np.all(data['valids'][s:s+64]) for s in starts)
    references={}
    for name,case in bank.items():
        targets=np.flatnonzero(np.linalg.norm(nodes-case['goal'],axis=1)<=1)
        references[name]=dijkstra(graph,indices=targets,min_only=True,unweighted=True)
    rows,paths,decisions,counts=[],[],[],[]
    common_inputs={}
    for model,cp,ex,ep,cdigest,edigest in ARMS:
        src=BASE if cp is None else g/cp
        step=100000 if cp is None else 101000
        check(src/f'params_{step}.pkl',cdigest)
        orig=base if cp is None else restore_agent(base,str(src),step)
        extracted=g/ex/'actor'
        check(extracted/f'params_{ep}.pkl',edigest)
        agent=restore_agent(base,str(extracted),ep)
        assert tree_equal(orig.network.params['modules_value'],agent.network.params['modules_value'])
        def score(x,y):
            # Identical batch shape for all values on paths; bound compilation and memory.
            x=np.asarray(x,np.float32)
            y=np.broadcast_to(y,x.shape).astype(np.float32)
            n=len(x)
            xx,yy=(np.pad(z,((0,(-n)%256),(0,0)),mode='edge') for z in (x,y))
            return np.concatenate([np.asarray(distances(agent,xx[i:i+256],yy[i:i+256])) for i in range(0,n,256)])[:n].astype(float)
        def summary(scope,metric,x):
            rows.append(dict(scene='OG',model=model,scope=scope,metric=metric,**stats(x)))
        def addpath(scope,case,xy,goal):
            d=score(xy[:-1],xy[1:])
            j=score(xy,goal)
            record=dict(model=model,scope=scope,case=case,**path_metrics(np.ones(len(d)),d,j))
            paths.append(record)
        dval=score(allobs[physical],allobs[physical+1])
        summary('val_records','h',dval-1)
        for s in starts:
            xy=allobs[s:s+65]
            addpath('val_point64',s,xy,xy[-1])
        addpath('geometry_shortest','canonical',gridroute,core.GOAL)
        addpath('reference_env','canonical',replay,core.GOAL)
        evroot=g/(ex+'_evaluation')
        official=read(evroot/'official/summary.json')
        # Confirm paired starts/goals on disk, use the recorded official success flag.
        for name,rec in official['episodes'].items():
            datafile=evroot/'official'/(name+'.npz')
            with np.load(datafile) as tr:
                xy,goal=tr['observations'],tr['goal']
                np.testing.assert_array_equal(xy[0],rec['initial_xy'])
                np.testing.assert_array_equal(goal,rec['goal'])
                if name not in common_inputs: common_inputs[name]=(xy[0],goal)
                else:
                    for a,b in zip(common_inputs[name],(xy[0],goal)): np.testing.assert_array_equal(a,b)
                assert bool(rec['success'])==bool(np.linalg.norm(xy[-1]-goal)<=1)
            counts.append(dict(model=model,case=name,task=rec['task'],success=int(rec['success']),steps=rec['steps']))
        for name,case in bank.items():
            tr=np.load(evroot/'official'/(name+'.npz'))
            xy,actions,goal=tr['observations'],tr['actions'],tr['goal']
            np.testing.assert_array_equal(goal,case['goal'])
            success=official['episodes'][name]['success']
            outcome='success' if success else 'timeout'
            addpath('actor_'+outcome,name,xy,goal)
            paths[-1]['last100_span']=np.ptp(xy[-100:],axis=0).max()
            vref=references[name]
            def reference(points):
                snap,k=tree.query(points)
                geom=np.where(np.linalg.norm(points-goal,axis=1)<=1,0,vref[k])
                return geom,snap
            for t in [0,20]+([] if success else [len(actions)]):
                if t==20 and t>=len(actions):continue
                if t==len(actions):
                    core.load_case(env,{**case,'integration':tr['final_integration'],'observation':xy[-1]})
                    scope='timeout_end'
                else:
                    core.load_case(env,case)
                    for k in range(t):
                        obs,*_=env.step(actions[k])
                        np.testing.assert_array_equal(obs,xy[k+1])
                    scope='start' if t==0 else 'step20'
                successors,_=engine.candidates(env)
                pred=score(successors,goal)
                lp=np.asarray(latent(agent,np.tile(xy[t],(121,1)),np.tile(goal,(121,1)),core.ACTIONS))
                c=np.ones(121)
                refv,snap=reference(successors)
                j=score(xy[t:t+1],goal)[0]
                terminal=np.linalg.norm(successors-goal,axis=1)<=1
                policy=np.argmin(pred); dyn=np.argmin(lp)
                # Check a chosen candidate with actual env.step then restore the full state.
                before=core.capture(env)
                obs,*_=env.step(core.ACTIONS[policy])
                np.testing.assert_array_equal(obs,successors[policy])
                core.restore(env,before)
                info=dict(model=model,case=name,scope=scope,outcome=outcome,t=t,
                          b=1+pred.min()-j,b_norm=1+pred.min()-j,
                          b_set=1+np.where(terminal,0,pred).min()-j,
                          snap_max=snap.max(),snap_mean=snap.mean(),terminal_candidates=int(terminal.sum()),
                          dyn_best_true_gap=pred[dyn]-pred.min(),latent_score_mae=np.abs(lp-pred).mean(),
                          latent_ref_regret=refv[dyn]-refv.min(),
                          ref_progress=reference(xy[t:t+1])[0][0]-refv[policy],
                          **ranking(c,pred,refv))
                if t<len(actions):
                    obs,*_=env.step(actions[t])
                    np.testing.assert_array_equal(obs,xy[t+1])
                    actual=score(obs[None],goal)[0]
                    info.update(actor_true_gap=actual-pred.min(),
                                actor_ref_excess=reference(obs[None])[0][0]-refv.min())
                decisions.append(info)
            tr.close()
        print(model,'done',flush=True)
    assert len(common_inputs)==250
    engine.close();env.close()
    write_rows(out/'metrics.csv',rows);write_rows(out/'paths.csv',paths)
    write_rows(out/'decisions.csv',decisions);write_rows(out/'official.csv',counts)
    (out/'inputs.json').write_text(json.dumps(ids,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path);parser.add_argument('out',type=Path)
    args=parser.parse_args();run(args.root.resolve(),args.out)
