"""Include dynamics-to-value gradients missing from the original M1 audit."""
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import time

from common import BASE_SHA,load_base,load_data,tree_digest
from critic import subset,init_optimizer,update
from mechanism import probe,dot,zero_momentum_motion
import jax
import numpy as np


@partial(jax.jit,static_argnames=('part',))
def gradient(agent,params,batch,part):
    def loss(p):
        full={**agent.network.params,**p}
        if part=='value':return agent.value_loss(batch,full)[0]
        if part=='dynamics':return agent.dynamics_loss(batch,full)[0]
        if part=='actor':return agent.actor_loss(batch,full,rng=jax.random.PRNGKey(0))[0]
        if part=='total':return agent.total_loss(batch,full)[0]
        raise ValueError(part)
    return jax.grad(loss)(params)


def difference(a,b):
    return max(float(np.max(np.abs(np.asarray(x)-np.asarray(y)))) for x,y in
               zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)))


def assert_gradient_equal(a,b):
    # Independently fused FP32 gradients can differ by a few ulps; compare
    # elementwise relative error as well as a small absolute cancellation floor.
    for x,y in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)):
        np.testing.assert_allclose(np.asarray(x),np.asarray(y),rtol=2e-5,atol=5e-5)


def stats(values):
    a=np.asarray(values)
    return dict(mean=float(a.mean()),std=float(a.std(ddof=1)),positive=int((a>0).sum()),
                negative=int((a<0).sum()),zero=int((a==0).sum()),n=len(a))


def run(root,output):
    start=time.time();root=Path(root).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    base,config=load_base();base_digest=tree_digest(base);env,train,_=load_data(config);env.close()
    previous=json.loads((root/'mechanism_gradient/analysis.json').read_text())
    field=np.load(root/'baseline_field/field.npz');points=field['path'][[0,15]].astype(np.float32)
    goals=np.broadcast_to(np.array([4.2055,12.0898],np.float32),points.shape)
    params=subset(base.network.params);opt=init_optimizer(base)
    initial=np.asarray(probe(base,params,points,goals))
    assert np.array_equal(points,np.asarray(previous['protocol']['points'],np.float32))
    np.testing.assert_allclose(initial,previous['protocol']['initial_probe'],rtol=0,atol=1e-7)
    jac=jax.jacrev(lambda p:probe(base,p,points,goals))(params)
    probes=[jax.tree_util.tree_map(lambda x:x[i],jac) for i in range(2)]
    protocol=dict(base_sha256=BASE_SHA,batches=32,seed=4400,batch_size=1024,points=points.tolist(),
        goal=goals[0].tolist(),initial_probe=initial.tolist(),start_unix=start,
        spec_sha256=hashlib.sha256((Path(__file__).parent/'FULL_GRADIENT_PROTOCOL.md').read_bytes()).hexdigest(),
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
        previous_analysis_sha256=hashlib.sha256((root/'mechanism_gradient/analysis.json').read_bytes()).hexdigest(),
        criterion='One-step independent counterfactuals of original checkpoint; no sequential continuation or saved models.')
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    zero=zero_momentum_motion(base,opt)
    zero_delta=np.asarray(probe(zero,subset(zero.network.params),points,goals))-initial
    stream=hashlib.sha256();np.random.seed(4400);rows=[]
    for i in range(32):
        batch=train.sample(1024)
        for key in ['observations','next_observations','value_goals']:stream.update(batch[key].tobytes())
        grads={part:gradient(base,params,batch,part) for part in ['value','dynamics','actor','total']}
        summed=jax.tree_util.tree_map(lambda a,b,c:a+b+c,grads['value'],grads['dynamics'],grads['actor'])
        error=difference(summed,grads['total']);assert_gradient_equal(summed,grads['total'])
        actor_norm=float(np.sqrt(dot(grads['actor']['modules_value'],grads['actor']['modules_value'])))
        assert actor_norm==0
        row=dict(batch=i,gradient_sum_max_error=error,actor_to_value_gradient_norm=actor_norm)
        for part,g in grads.items():
            row[f'gradient_norm/{part}']=float(np.sqrt(dot(g['modules_value'],g['modules_value'])))
            for name,gp in zip(['gap','shape'],probes):row[f'sgd/{part}/{name}']=float(-config['lr']*dot(gp,g))
        norm_product=row['gradient_norm/value']*row['gradient_norm/dynamics']
        row['gradient_cosine/value_dynamics']=float(dot(grads['value']['modules_value'],grads['dynamics']['modules_value']))/max(norm_product,1e-30)
        continued,_,_=update(base,opt,batch,'pair')
        joint,_=base.update(batch)
        assert int(joint.network.opt_state[0].count)==100001
        assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(joint))
        for name,new_agent in [('value_only',continued),('joint',joint)]:
            actual=np.asarray(probe(new_agent,subset(new_agent.network.params),points,goals))-initial
            delta=jax.tree_util.tree_map(lambda a,b:a-b,subset(new_agent.network.params),params)
            for k,(label,gp) in enumerate(zip(['gap','shape'],probes)):
                row[f'adam/{name}/{label}']=float(actual[k]);row[f'adam_linear/{name}/{label}']=float(dot(gp,delta))
        # Repeat M1's exact value-only counterfactual rather than just its average.
        for label in ['gap','shape']:
            np.testing.assert_allclose(row[f'adam/value_only/{label}'],previous['rows'][i][f'adam/pair/{label}'],rtol=0,atol=1e-7)
            row[f'adam/joint_minus_value/{label}']=row[f'adam/joint/{label}']-row[f'adam/value_only/{label}']
        assert all(np.isfinite(v) for v in row.values());rows.append(row)
        (output/'progress.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
    assert stream.hexdigest()==previous['paired_batch_stream_sha256']
    assert tree_digest(base)==base_digest,'Original functional state must remain unchanged.'
    summary={key:stats([r[key] for r in rows]) for key in rows[0] if key!='batch'}
    result=dict(protocol=protocol,rows=rows,statistics=summary,paired_batch_stream_sha256=stream.hexdigest(),
        previous_stream_equal=True,previous_value_only_counterfactuals_equal=True,original_state_unchanged=True,
        zero_new_gradient_adam_probe_delta=zero_delta.tolist(),elapsed_seconds=time.time()-start)
    (output/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print('COMPLETE',output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--output',required=True)
    a=p.parse_args();run(a.root,a.output)
