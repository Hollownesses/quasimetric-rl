"""Matched critic-only one/multistep continuation; all new state is explicit."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time

from common import load_base,load_data,tree_digest,BASE_SHA
from train import freeze_proof
import multistep as m
import jax
import numpy as np


def run(output,steps=10000,seed=4501,log_interval=500,save_interval=5000):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    start=time.time();base,config=load_base();env,train,val=load_data(config);env.close()
    initial=m.initialize(base);states={a:initial for a in ['one','multi']}
    np.random.seed(9001);validation=val.sample(2048)
    rng=np.random.RandomState(seed);stream=hashlib.sha256()
    manifest=dict(steps=steps,seed=seed,batch_size=m.BATCH,uniform_batch_goal_samples=m.OFFDIAG,
        gamma=m.GAMMA,waypoint_discount=m.WAYPOINT_DISCOUNT,next_probability=m.NEXT_PROB,
        diagonal_weight=m.DIAG_WEIGHT,linex_threshold=m.CLIP,distance_unit=m.UNIT,
        learning_rate=config['lr'],initialization='scaled original V; cloned Q with zero action input weights',
        optimizer='fresh Adam count0 for new critic; authoritative state in critic_N.pkl',
        base_sha256=BASE_SHA,frozen=freeze_proof(base),initial_state_digest=tree_digest(initial),
        heldout_sha256=tree_digest(validation),log_interval=log_interval,save_interval=save_interval,
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
        protocol_sha256=hashlib.sha256((Path(__file__).parent/'MULTISTEP_PROTOCOL.md').read_bytes()).hexdigest(),
        jax=jax.__version__,devices=[str(d) for d in jax.devices()],start_unix=start,training_pid=os.getpid())
    (output/'protocol.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for arm,state in states.items():m.save(base,state,output/arm,arm,rng)
    with (output/'metrics.csv').open('w') as f:
        writer=None
        for i in range(1,steps+1):
            batch=m.sample(train,rng)
            for key in ['source_indices','goal_indices','waypoint_indices','goal_choices']:
                stream.update(batch[key].tobytes())
            info={}
            for arm in states:
                states[arm],info[arm]=m.update(base,states[arm],batch,arm)
            if i==1 or i%log_interval==0 or i==steps:
                for arm in states:
                    row=dict(step=i,arm=arm,elapsed_seconds=time.time()-start)
                    row.update({f'train/{k}':float(v) for k,v in info[arm].items()})
                    row.update({f'heldout/{k}':float(v) for k,v in m.diagnostics(base,states[arm],validation).items()})
                    assert all(np.isfinite(v) for v in row.values() if isinstance(v,(int,float)))
                    if writer is None:
                        writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader()
                    writer.writerow(row);f.flush();print(json.dumps(row),flush=True)
            if i%save_interval==0 or i==steps:
                for arm,state in states.items():
                    assert int(state.step)==i and int(state.opt_state[0].count)==i
                    assert freeze_proof(m.as_agent(base,state))==manifest['frozen']
                    path=m.save(base,state,output/arm,arm,rng)
                    print('Saved',path,flush=True)
    done=dict(steps=steps,source_transitions_per_arm=steps*m.BATCH,
        shared_batch_stream_sha256=stream.hexdigest(),elapsed_seconds=time.time()-start,
        final={a:dict(step=int(s.step),adam_count=int(s.opt_state[0].count),state_digest=tree_digest(s),
                     value_digest=tree_digest(s.params['value']),q_digest=tree_digest(s.params['q'])) for a,s in states.items()})
    (output/'training_complete.json').write_text(json.dumps(done,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=10000)
    p.add_argument('--seed',type=int,default=4501);p.add_argument('--log-interval',type=int,default=500);p.add_argument('--save-interval',type=int,default=5000)
    a=p.parse_args();run(a.output,a.steps,a.seed,a.log_interval,a.save_interval)
