"""A single random8 arm; reuse the E16 value update and both E17 extraction updates."""
import argparse
import hashlib
import json
import shutil
import sys
import traceback
from pathlib import Path
import numpy as np
from support import HERE,RAW,CODE,G,UNIFIED,read,write,check,sha
from common import load_base,load_data,BASE,tree_equal,tree_digest,distances,UPSTREAM
from hard_goal import HardGoal,paired_errors
from roadmap_metric import RoadmapMetric
import iqe_distill
import extraction
import jax
from utils.flax_utils import save_agent,restore_agent

THREE=('observations','next_observations','value_goals')
FIVE=(*THREE,'actions','actor_goals')
SEED=20261102

def feed(h,arrays):
    for a in arrays:h.update(a.tobytes())

def stream_batch(stream,batch,names):feed(stream,[batch[n] for n in names])

def parse_streams(h3,h5):
    return dict(original_three_fields_sha256=h3.hexdigest(),full_actor_fields_sha256=h5.hexdigest())

def choose(agent,batch,start,teacher,sampler,rng):
    uniform,sources,pool,ids=sampler.proposals(batch,start)
    flat=pool.reshape(-1,2)
    labels=teacher(sources,flat).astype(np.float32).reshape(pool.shape[:2])
    err=np.asarray(paired_errors(agent,sources,flat,labels.ravel())).reshape(labels.shape)
    assert err.shape==(128,8) and np.isfinite(err).all()
    cols=rng.integers(0,8,len(err),dtype=np.int64)
    rows=np.arange(len(cols))
    selected=dict(uniform)
    selected['value_goals']=uniform['value_goals'].copy()
    selected['value_goals'][1::2]=pool[rows,cols]
    targets=np.empty(256,np.float32)
    targets[::2]=teacher(uniform['observations'][::2],uniform['value_goals'][::2]).astype(np.float32)
    targets[1::2]=labels[rows,cols]
    np.testing.assert_array_equal(selected['observations'][::2],batch['observations'][::2])
    np.testing.assert_array_equal(selected['value_goals'][::2],batch['value_goals'][::2])
    return selected,targets,uniform,pool,labels,err,cols,ids

def save_full(agent,base,directory,epoch):
    assert int(agent.network.opt_state[0].count)==epoch and int(agent.network.step)==epoch+1
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(agent))
    directory.mkdir(parents=True,exist_ok=False)
    save_agent(agent,str(directory),epoch)
    assert tree_equal(agent,restore_agent(base,str(directory),epoch))
    p=directory/f'params_{epoch}.pkl'
    return dict(path=str(p.resolve()),sha256=sha(p),count=epoch)

def preflight(out,base,cfg,train,val):
    hp=read(G/'hard_goal_1000/protocol.json');ep=read(G/'extraction_1000/protocol.json')
    ident=read(UNIFIED/'og/inputs.json')
    assert json.loads(json.dumps(cfg))==ep['config']==hp['config']
    checks={str(RAW/n):check(RAW/n,v) for n,v in ident.items()}
    for name,key in [('graph.npz','teacher_graph_sha256'),('distances.npy','teacher_distance_sha256')]:
        p=G/'roadmap_metric'/name;checks[str(p)]=check(p,hp[key])
    for n in ['iqe_distill.py','hard_goal.py','goal_support.py','query_augmentation.py','roadmap_metric.py','common.py']:
        checks[str(CODE/n)]=check(CODE/n,hp['source_sha256'][n])
    for n in ['extraction.py','evaluate_extraction.py','actor_recording.py']:
        checks[str(CODE/n)]=check(CODE/n,sha(G/'extraction_1000/source_snapshot'/n))
    e17=read(G/'extraction_1000/improved_evaluation/analysis.json')
    for f,key in [('utils/evaluation.py','official_evaluation_sha256'),('agents/qrl.py','official_qrl_sha256')]:
        p=UPSTREAM/'impls'/f;checks[str(p)]=check(p,e17[key])
    e18=read(G/'mean_extraction_1000/report_audit.json')
    assert e18['passed'] and e18['matching_e17_training_streams']
    assert iqe_distill.frozen_proof(base)==hp['initial_frozen']
    np.random.seed(9001);validation=val.sample(2048)
    assert tree_digest(validation)==ep['validation_sha256']==hp['validation_sha256']
    expect=read(G/'extraction_1000/training_complete.json')['stage_streams']
    np.random.seed(2001)
    for stage in ('dynamics','actor'):
        h3,h5=hashlib.sha256(),hashlib.sha256()
        for _ in range(1000):
            batch=train.sample(256);stream_batch(h3,batch,THREE);stream_batch(h5,batch,FIVE)
        assert parse_streams(h3,h5)==expect[stage]
    write(out/'preflight.json',dict(paths_sha256=checks,expected_stage_streams=expect,config=cfg,
          validation_sha256=tree_digest(validation),initial_agent_sha256=tree_digest(base),
          seed=SEED,devices=[str(x) for x in jax.devices()],runtime=sys.executable))
    return validation,expect

def critic(out,base,train):
    t=RoadmapMetric.load(G/'roadmap_metric/graph.npz',G/'roadmap_metric/distances.npy')
    sampler=HardGoal(t.nodes,candidates=8)
    old=read(G/'hard_goal_1000/training_complete.json')
    frozen=iqe_distill.frozen_proof(base)
    for smoke in (True,False):
        rng=np.random.Generator(np.random.PCG64(SEED));np.random.seed(2001)
        agent=base;opt=iqe_distill.init_optimizer(base)
        streams={k:hashlib.sha256() for k in ('paired','pool','uniform_queries','selection','queries','targets')}
        choices=[];logs=[];ids_count=np.zeros(len(t.nodes),int)
        for i in range(2 if smoke else 1000):
            batch=train.sample(256)
            before=np.random.get_state() if smoke else None
            selected,target,uniform,pool,labels,err,cols,ids=choose(agent,batch,i*128,t,sampler,rng)
            if smoke:
                maxsel,maxtarget,_,historical=sampler.choose(agent,batch,i*128,t)
                np.testing.assert_array_equal(pool,historical['goals']);np.testing.assert_array_equal(labels,historical['labels'])
                np.testing.assert_array_equal(np.argmax(err,axis=1),historical['indices'])
                np.testing.assert_array_equal(maxsel['observations'],selected['observations'])
                np.testing.assert_array_equal(maxtarget[::2],target[::2])
                after=np.random.get_state()
                for a,b in zip(before,after):np.testing.assert_array_equal(a,b)
            stream_batch(streams['paired'],batch,THREE)
            stream_batch(streams['uniform_queries'],uniform,('observations','value_goals'))
            feed(streams['pool'],[selected['observations'][1::2],pool,labels])
            feed(streams['selection'],[cols]);feed(streams['targets'],[target])
            stream_batch(streams['queries'],selected,('observations','value_goals'))
            rows=np.arange(128);choices.append(cols)
            np.add.at(ids_count,ids[rows,cols],1)
            agent,opt,info=iqe_distill.update(agent,opt,selected,target)
            loss=float(info['loss']); assert np.isfinite(loss)
            logs.append([i+1,loss,float(err.mean()),float(err.max(axis=1).mean()),float(err[rows,cols].mean()),
                         float((err.argmax(axis=1)==cols).mean())])
            if (i+1)%100==0:print('critic',i+1,loss,flush=True)
        assert iqe_distill.frozen_proof(agent)==frozen and tree_equal(base.rng,agent.rng)
        if smoke:
            save_full(agent,base,out/'smoke/critic',100002)
            smoke_digest={k:h.hexdigest() for k,h in streams.items()}
            write(out/'smoke/critic.json',dict(passed=True,prefix=smoke_digest))
        else:
            for key in ('paired','uniform_queries','pool'):assert streams[key].hexdigest()==old[key+'_stream_sha256'],key
            assert streams['selection'].hexdigest()!=old['selection_stream_sha256']
            ck=save_full(agent,base,out/'critic',101000)
            np.savez_compressed(out/'selection.npz',indices=np.array(choices),prototype_counts=ids_count)
            np.savetxt(out/'critic_metrics.csv',logs,delimiter=',',header='step,loss,pool_mean_error,pool_max_error,chosen_error,chosen_is_max_fraction',comments='')
            write(out/'critic.json',dict(checkpoint=ck,streams={k:h.hexdigest() for k,h in streams.items()},
                  column_counts=np.bincount(np.concatenate(choices),minlength=8).tolist(),unique_prototypes=int((ids_count>0).sum()),
                  frozen=True,selector_rng=rng.bit_generator.state))
        if not smoke:return agent

def extract(out,base,initial,train,validation,expected):
    fixed=extraction.proof(initial,['modules_value','modules_lam'])
    values=np.asarray(distances(initial,validation['observations'],validation['value_goals']))
    for smoke in (True,False):
        np.random.seed(2001);agent=initial
        for n,stage in enumerate(('dynamics','actor')):
            opt=extraction.init_optimizer(agent,stage);h3,h5=hashlib.sha256(),hashlib.sha256()
            key=extraction.module_key(stage);frozen=extraction.proof(agent,[k for k in agent.network.params if k!=key])
            before=agent;logs=[]
            for i in range(2 if smoke else 1000):
                batch=train.sample(256);stream_batch(h3,batch,THREE);stream_batch(h5,batch,FIVE)
                agent,opt,info=extraction.update(agent,opt,batch,stage)
                vals={k:float(v) for k,v in info.items()};assert all(np.isfinite(list(vals.values())))
                if (i+1)%100==0 or i==0:
                    logs.append(dict(step=i+1,train=vals,validation={k:float(v) for k,v in extraction.statistics(agent,validation).items()}))
                    if not smoke:print(stage,i+1,flush=True)
            assert extraction.proof(agent,list(frozen))==frozen and extraction.proof(agent,list(fixed))==fixed
            assert not tree_equal(before.network.params[key],agent.network.params[key])
            np.testing.assert_array_equal(distances(agent,validation['observations'],validation['value_goals']),values)
            if not smoke:
                streams=parse_streams(h3,h5);assert streams==expected[stage]
                ck=save_full(agent,base,out/stage,102000+n*1000)
                write(out/(stage+'.json'),dict(checkpoint=ck,streams=streams,logs=logs,fixed=fixed))
        if smoke:
            save_full(agent,base,out/'smoke/extraction',101004)
            write(out/'smoke/extraction.json',dict(passed=True,reset_to=101000))

def run(out):
    base,cfg=load_base();env,train,val=load_data(cfg);env.close()
    validation,expected=preflight(out,base,cfg,train,val)
    initial=critic(out,base,train)
    extract(out,base,initial,train,validation,expected)
    for p,h in read(out/'preflight.json')['paths_sha256'].items():check(p,h)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);out=p.parse_args().out.resolve()
    out.mkdir(parents=True,exist_ok=False);shutil.copy2(HERE/'PROTOCOL.md',out/'PROTOCOL.md')
    write(out/'freeze.json',dict(protocol_sha256=sha(out/'PROTOCOL.md'),train_code={n:sha(HERE/n) for n in ['train.py','support.py']},
          update_budget=[1000,1000,1000],selection_seed=SEED))
    try:run(out)
    except BaseException:
        write(out/'training_failure.json',dict(traceback=traceback.format_exc()));raise
