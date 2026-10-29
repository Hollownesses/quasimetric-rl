"""Matched A/B critic continuation; one sampled batch is supplied to both arms."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

from common import ROOT, BASE, BASE_SHA, UPSTREAM, load_base, load_data, tree_digest
from critic import diagnostics, init_optimizer, update
import jax
import numpy as np
from utils.flax_utils import save_agent


def freeze_proof(agent):
    opt=agent.network.opt_state[0]
    return {key:dict(params=tree_digest(agent.network.params[key]),mu=tree_digest(opt.mu[key]),
                     nu=tree_digest(opt.nu[key])) for key in ['modules_actor','modules_dynamics']}


def run(output,steps,seed,log_interval,save_interval):
    output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    start=time.time()
    agent,config=load_base()
    env,train,val=load_data(config)
    env.close()
    proof=freeze_proof(agent)
    # Fixed held-out batch without perturbing training's official NumPy sampler.
    np.random.seed(9001)
    validation=val.sample(2048)
    np.random.seed(seed)
    params=dict(pair=agent,goal=agent)
    opts={k:init_optimizer(v) for k,v in params.items()}
    manifest=dict(steps=steps,seed=seed,log_interval=log_interval,save_interval=save_interval,
        base_checkpoint=str(BASE/'params_100000.pkl'),base_sha256=BASE_SHA,config=config,
        official_commit=(ROOT/'experiments/ogbench_official_qrl/OFFICIAL_COMMIT').read_text().strip(),
        batch_size=1024,endpoint='fixed; no early stopping or best selection',
        arms={'pair':'official value_loss only','goal':'replace pair penalty with d(s,g)-d(s_next,g)-1'},
        actor_dynamics='frozen parameters and first/second Adam moments; no dynamics loss',
        critic_optimizer='pretrained moments and count preserved; lr unchanged',
        validation_batch_sha256=tree_digest(validation),initial_frozen=proof,
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
        jax=jax.__version__,devices=[str(d) for d in jax.devices()],start_unix=start)
    frozen=output/'protocol.json'
    frozen.write_text(json.dumps(manifest,indent=2)+'\n')
    (output/'protocol.sha256').write_text(hashlib.sha256(frozen.read_bytes()).hexdigest()+'\n')
    for arm in params: (output/arm).mkdir()
    stream=hashlib.sha256()
    with (output/'metrics.csv').open('w') as f:
        writer=None
        for i in range(1,steps+1):
            batch=train.sample(1024)
            for name in ['observations','next_observations','value_goals']:
                stream.update(batch[name].tobytes())
            infos={}
            for arm in params:
                params[arm],opts[arm],infos[arm]=update(params[arm],opts[arm],batch,arm)
            if i%log_interval==0 or i==1 or i==steps:
                for arm in params:
                    row=dict(step=i,arm=arm,elapsed_seconds=time.time()-start)
                    row.update({f'train/{k}':float(v) for k,v in infos[arm].items()})
                    row.update({f'heldout/{k}':float(v) for k,v in diagnostics(params[arm],validation).items()})
                    # The official A info contains extra diagnostic keys: use common schema.
                    keys=['total_loss','value_loss','lam_loss','d_neg_loss','d_pos_loss','lam']
                    row={k:v for k,v in row.items() if not k.startswith('train/') or k[6:] in keys}
                    assert all(np.isfinite(v) for v in row.values() if isinstance(v,(float,int)))
                    if writer is None:
                        writer=csv.DictWriter(f,fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    f.flush()
                    print(json.dumps(row),flush=True)
            if i%save_interval==0 or i==steps:
                for arm in params:
                    assert freeze_proof(params[arm])==proof
                    assert int(params[arm].network.opt_state[0].count)==100000+i
                    save_agent(params[arm],str(output/arm),100000+i)
    final={arm:dict(frozen=freeze_proof(params[arm]),
        critic_digest=tree_digest(params[arm].network.params['modules_value']),
        adam_count=int(params[arm].network.opt_state[0].count)) for arm in params}
    (output/'training_complete.json').write_text(json.dumps(dict(final=final,
        paired_batch_stream_sha256=stream.hexdigest(),elapsed_seconds=time.time()-start,
        steps=steps,end_unix=time.time()),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=10000)
    p.add_argument('--seed',type=int,default=2001)
    p.add_argument('--log-interval',type=int,default=500)
    p.add_argument('--save-interval',type=int,default=5000)
    a=p.parse_args()
    run(a.output,a.steps,a.seed,a.log_interval,a.save_interval)
