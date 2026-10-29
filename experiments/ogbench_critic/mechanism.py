"""Original-checkpoint-only gradient decomposition of the fixed task3 barrier."""
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import time

from common import ROOT,BASE_SHA,load_base,load_data,tree_digest
from critic import subset,init_optimizer,update
from train import freeze_proof
import jax
import jax.numpy as jnp
import numpy as np
import optax


@jax.jit
def probe(agent,params,points,goals):
    values=agent.network.select('value')(points,goals,params={**agent.network.params,**params})
    gap=values[1]-values[0]
    shape=gap/(values.mean()+1e-6)
    return jnp.stack([gap,shape])


@partial(jax.jit,static_argnames=('part',))
def part_gradient(agent,params,batch,part):
    def loss(p):
        full={**agent.network.params,**p}
        d=partial(agent.network.select('value'),params=full)
        if part=='push':
            sg=d(batch['observations'],batch['value_goals'])
            return (100*jax.nn.softplus(5-sg/100)).mean()
        lam=jax.lax.stop_gradient(agent.network.select('lam')())
        if part=='pair':
            h=d(batch['observations'],batch['next_observations'])-1
        elif part=='goal':
            h=d(batch['observations'],batch['value_goals'])-d(batch['next_observations'],batch['value_goals'])-1
        else:raise ValueError(part)
        return lam*jnp.square(jax.nn.relu(h)).mean()
    return jax.grad(loss)(params)


def dot(a,b):
    return sum(jnp.vdot(x,y) for x,y in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)))


@jax.jit
def zero_momentum_motion(agent,opt):
    params=subset(agent.network.params)
    zero=jax.tree_util.tree_map(jnp.zeros_like,params)
    delta,_=optax.adam(agent.config['lr']).update(zero,opt,params)
    new=optax.apply_updates(params,delta)
    return agent.replace(network=agent.network.replace(params={**agent.network.params,**new}))


def run(root,output,batches=32,seed=4400):
    start=time.time();root=Path(root).resolve();output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    base,config=load_base();env,train,_=load_data(config);env.close()
    field=np.load(root/'baseline_field/field.npz')
    points=field['path'][[0,15]].astype(np.float32)
    goal=np.array([4.2055,12.0898],np.float32)
    goals=np.broadcast_to(goal,points.shape)
    params=subset(base.network.params);opt=init_optimizer(base)
    original=np.asarray(probe(base,params,points,goals))
    jac=jax.jacrev(lambda p:probe(base,p,points,goals))(params)
    grad_probes=[jax.tree_util.tree_map(lambda x:x[i],jac) for i in range(2)]
    protocol=dict(base_sha256=BASE_SHA,batches=batches,seed=seed,batch_size=1024,
        points=points.tolist(),goal=goal.tolist(),initial_probe=original.tolist(),
        criterion='Fixed original checkpoint; independent one-step counterfactuals, no training continuation.',
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
        spec_sha256=hashlib.sha256((Path(__file__).parent/'MECHANISM_PROTOCOL.md').read_bytes()).hexdigest(),
        original_frozen=freeze_proof(base),start_unix=start)
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    zero=zero_momentum_motion(base,opt)
    zero_delta=(np.asarray(probe(zero,subset(zero.network.params),points,goals))-original).tolist()
    np.random.seed(seed)
    stream=hashlib.sha256();rows=[]
    for i in range(batches):
        batch=train.sample(1024)
        for key in ['observations','next_observations','value_goals']:stream.update(batch[key].tobytes())
        gradients={part:part_gradient(base,params,batch,part) for part in ['push','pair','goal']}
        row=dict(batch=i)
        for part,g in gradients.items():
            for name,gp in zip(['gap','shape'],grad_probes):
                row[f'sgd/{part}/{name}']=float(-base.config['lr']*dot(gp,g))
        for arm in ['pair','goal']:
            updated,_,_=update(base,opt,batch,arm)
            assert freeze_proof(updated)==protocol['original_frozen']
            delta=jax.tree_util.tree_map(lambda a,b:a-b,subset(updated.network.params),params)
            actual=np.asarray(probe(updated,subset(updated.network.params),points,goals))-original
            for index,(name,gp) in enumerate(zip(['gap','shape'],grad_probes)):
                row[f'adam/{arm}/{name}']=float(actual[index])
                row[f'adam_linear/{arm}/{name}']=float(dot(gp,delta))
        assert all(np.isfinite(v) for v in row.values())
        rows.append(row)
        (output/'progress.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(json.dumps(row),flush=True)
    stats={}
    for key in rows[0]:
        if key=='batch':continue
        a=np.array([r[key] for r in rows])
        stats[key]=dict(mean=float(a.mean()),std=float(a.std(ddof=1)),positive=int((a>0).sum()),
                       negative=int((a<0).sum()),zero=int((a==0).sum()),n=len(a))
    for arm in ['pair','goal']:
        for name in ['gap','shape']:
            error=np.array([r[f'adam/{arm}/{name}']-r[f'adam_linear/{arm}/{name}'] for r in rows])
            stats[f'linearization_error/{arm}/{name}']=dict(mean=float(error.mean()),
                mean_absolute=float(np.abs(error).mean()),max_absolute=float(np.abs(error).max()))
    result=dict(protocol=protocol,statistics=stats,zero_new_gradient_adam_probe_delta=zero_delta,
        paired_batch_stream_sha256=stream.hexdigest(),elapsed_seconds=time.time()-start,rows=rows)
    (output/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print('COMPLETE',output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--output',required=True)
    a=p.parse_args();run(a.root,a.output)
