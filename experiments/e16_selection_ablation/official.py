"""Observe seeds/keys at the boundary of the unchanged E17 official evaluator."""
import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
from support import G,read,write,check
from common import load_base,load_data
from utils.flax_utils import restore_agent
import evaluate_extraction as e17
import jax


def np_hash():
    name,state,pos,gauss,cached=np.random.get_state()
    h=hashlib.sha256(name.encode());h.update(state.tobytes())
    h.update(np.array([pos,gauss],dtype=np.int64).tobytes());h.update(np.float64(cached).tobytes())
    return h.hexdigest()


class ResetObserver(e17.EpisodeRecorder):
    def reset(self,*args,**kwargs):
        prior=np_hash()
        result=super().reset(*args,**kwargs)
        self.starts[-1].update(before=prior,after=np_hash())
        return result


class KeyObserver:
    def __init__(self,agent,env,records):
        self.agent,self.env,self.records=agent,env,records

    def sample_actions(self,*,seed,**kw):
        ep=len(self.env.starts)-1
        key=np.asarray(seed,np.uint32)
        if ep not in self.records:self.records[ep]=dict(first=key.tolist(),stream=hashlib.sha256(),count=0)
        r=self.records[ep];r['last']=key.tolist();r['count']+=1;r['stream'].update(key.tobytes())
        return self.agent.sample_actions(seed=seed,**kw)


@jax.jit
def key_block(initial):
    return jax.lax.scan(lambda key,_:tuple(jax.random.split(key)),initial,None,length=512)


def make_keys(seed,steps):
    carry=jax.random.PRNGKey(seed);blocks=[]
    for _ in range((steps+511)//512):
        carry,keys=key_block(carry);blocks.append(np.asarray(keys))
    return np.concatenate(blocks)[:steps]


def audit(out,taps):
    base,cfg=load_base()
    env,_,_=load_data(cfg)
    bdir=G/'extraction_1000/improved'
    hist=restore_agent(base,str(bdir/'actor'),103000)
    assert e17.tree_digest(hist.network.params['modules_value'])==read(G/'extraction_1000/improved_evaluation/analysis.json')['critic_digest']
    old=G/'extraction_1000/improved_evaluation/official'
    new=out/'official_evaluation/official'
    prev,now=(read(p/'summary.json')['episodes'] for p in [old,new])
    assert list(prev)==list(now) and len(now)==250
    np.random.seed(0);env.reset(seed=0);np.random.seed(0)
    rows=[];key_rows=[]
    for task in range(1,6):
        tap=taps[task-1];seed=int(np.random.randint(0,2**32))
        assert seed==tap['task_seed'] and tap['task']==task
        names=[f'task{task}_episode{i:03d}' for i in range(50)]
        counts=[[s[name]['steps'] for name in names] for s in (prev,now)]
        keys=make_keys(seed,max(map(sum,counts)))
        offsets=[0,0]
        for i,name in enumerate(names):
            prior=np_hash();x,info=env.reset(options=dict(task_id=task,render_goal=False));after=np_hash()
            expected=tap['episodes'][i]
            assert prior==expected['before'] and after==expected['after']
            record=dict(case=name,task=task,task_seed=seed,reset_before=prior,reset_after=after)
            for n,(arm,trdir,summary) in enumerate(zip(['high','random'],[old,new],[prev,now])):
                steps=counts[n][i]
                with np.load(trdir/(name+'.npz')) as tr:
                    np.testing.assert_array_equal(tr['observations'][0],x);np.testing.assert_array_equal(tr['goal'],info['goal'])
                    assert len(tr['actions'])==steps and len(tr['observations'])==steps+1
                    assert tr['done'][-1] and not tr['done'][:-1].any()
                    assert np.isfinite(tr['observations']).all() and np.isfinite(tr['actions']).all()
                    assert bool(summary[name]['success'])==bool(np.linalg.norm(tr['observations'][-1]-tr['goal'])<=1)
                    stream=keys[offsets[n]:offsets[n]+steps]
                    hexhash=hashlib.sha256(stream.tobytes()).hexdigest()
                    if n==1:
                        assert expected['count']==steps and expected['stream']==hexhash
                        assert expected['first']==stream[0].tolist() and expected['last']==stream[-1].tolist()
                    else:
                        # Historical keys were not logged. Audit the reconstruction against
                        # both actual boundary actions, retaining temperature zero.
                        for index in (0,steps-1):
                            a=hist.sample_actions(observations=tr['observations'][index],goals=tr['goal'],
                                                  temperature=0,seed=stream[index])
                            np.testing.assert_array_equal(a,tr['actions'][index])
                    record.update({arm+'_offset':offsets[n],arm+'_steps':steps,arm+'_keys_sha256':hexhash,
                                   arm+'_first':stream[0].tolist(),arm+'_last':stream[-1].tolist()})
                offsets[n]+=steps
            key_rows.append(record)
            rows.append(dict(case=name,task=task,high_success=int(prev[name]['success']),
                             random_success=int(now[name]['success']),high_steps=counts[0][i],random_steps=counts[1][i]))
    env.close()
    with (out/'official_pairs.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(out/'rng_audit.json',dict(actual_random_vs_reconstructed=True,historical_key_source='pinned evaluator reconstruction; verified both recorded boundary actions per episode; no historical key log',
                                    matched_inputs=250,boundary_actions_checked_high=500,rows=key_rows))


def run(out):
    actor=read(out/'actor.json')
    check(out/'actor/params_103000.pkl',actor['checkpoint']['sha256'])
    original=e17.evaluate
    taps=[]
    def observed(agent,env,**kwargs):
        # Read a copy of the legacy generator state, leaving the actual evaluator draw intact.
        copy=np.random.RandomState();copy.set_state(np.random.get_state())
        task_seed=int(copy.randint(0,2**32));records={};offset=len(env.starts)
        result=original(KeyObserver(agent,env,records),env,**kwargs)
        episodes=[]
        for ep in range(offset,len(env.starts)):
            row=records[ep].copy();row['stream']=row['stream'].hexdigest()
            row.update({k:env.starts[ep][k] for k in ['before','after']});episodes.append(row)
        taps.append(dict(task=kwargs['task_id'],task_seed=task_seed,episodes=episodes))
        write(out/'observed_rng.json',taps)
        return result
    e17.EpisodeRecorder=ResetObserver;e17.evaluate=observed
    e17.run(G,out/'actor',103000,actor['fixed']['modules_value']['params'],out/'official_evaluation',1800)
    audit(out,taps)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);p.add_argument('--audit-only',action='store_true');a=p.parse_args()
    if a.audit_only:audit(a.out,read(a.out/'observed_rng.json'))
    else:run(a.out.resolve())
