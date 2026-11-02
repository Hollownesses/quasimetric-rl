"""Original joint training, with only one value primal term removed per arm."""
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

def run(raw,out):
    out.mkdir(parents=True,exist_ok=False)
    sys.path[:0]=[str(raw/'experiments/ogbench_critic'),str(raw)]
    from common import load_base,load_data,tree_digest,tree_equal,sha256
    from utils.flax_utils import save_agent,restore_agent
    import jax
    import jax.numpy as jnp
    agent,cfg=load_base();before=tree_digest(agent);env,train,_=load_data(cfg);env.close()
    arms=('native','no_push','no_local')
    @partial(jax.jit,static_argnames=('arm',))
    def ablate(a,batch,arm):
        nrng,rng=jax.random.split(a.rng)
        def loss(p):
            value,info=a.value_loss(batch,p)
            if arm=='no_push':value=value-info['d_neg_loss']
            elif arm=='no_local':value=value-jax.lax.stop_gradient(info['lam'])*info['d_pos_loss']
            else:raise ValueError(arm)
            dynamics,di=a.dynamics_loss(batch,p)
            _,actor_rng=jax.random.split(rng)
            actor,ai=a.actor_loss(batch,p,actor_rng)
            return value+dynamics+actor,{}
        return a.replace(network=a.network.apply_loss_fn(loss_fn=loss)[0],rng=nrng)
    def update(a,batch,arm):
        # Keep native on its exact upstream compilation path.
        return a.update(batch)[0] if arm=='native' else ablate(a,batch,arm)
    np.random.seed(4400);batch=train.sample(1024)
    assert tree_equal(update(agent,batch,'native'),agent.update(batch)[0])
    first={arm:update(agent,batch,arm) for arm in arms}
    isolation=0.
    for arm in ['no_push','no_local']:
        for key in ['modules_lam','modules_dynamics','modules_actor']:
            for x,y in zip(jax.tree_util.tree_leaves(first['native'].network.params[key]),jax.tree_util.tree_leaves(first[arm].network.params[key])):
                isolation=max(isolation,float(np.max(abs(np.asarray(x)-np.asarray(y)))))
                np.testing.assert_allclose(x,y,rtol=1e-6,atol=1e-6)
    smoke={}
    for arm in arms:
        np.random.seed(4400);a=agent
        for i in range(2):a=update(a,train.sample(1024),arm)
        dest=out/(arm+'_smoke');dest.mkdir();save_agent(a,str(dest),100002);b=restore_agent(agent,str(dest),100002)
        assert tree_equal(a,b);smoke[arm]=True
    models={k:agent for k in arms};np.random.seed(4400);stream=hashlib.sha256();full=hashlib.sha256();first32=None
    for i in range(512):
        batch=train.sample(1024)
        for key in ['observations','next_observations','value_goals']:stream.update(batch[key].tobytes())
        for key in sorted(batch):
            full.update(key.encode());full.update(np.asarray(batch[key]).tobytes())
        if i==31:first32=stream.hexdigest()
        for arm in arms:models[arm]=update(models[arm],batch,arm)
        if (i+1)%32==0:
            for a in models.values():assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(a))
            print('OG trained',i+1,flush=True)
    repo=Path(__file__).resolve().parents[2]
    expected=json.loads((repo/'results/native_margin_20261102_verified/og/audit.json').read_text())['stream_sha256'];assert first32==expected
    ck={}
    for arm,a in models.items():
        dest=out/arm;dest.mkdir();save_agent(a,str(dest),100512);b=restore_agent(agent,str(dest),100512)
        assert tree_equal(a,b);assert int(a.network.opt_state[0].count)==100512
        ck[arm]=dict(sha256=sha256(dest/'params_100512.pkl'),agent_sha256=tree_digest(a),lam=float(a.network.select('lam')()))
    assert tree_digest(agent)==before
    result=dict(steps=512,lr=cfg['lr'],first32_sha256=first32,stream_sha256=stream.hexdigest(),
           full_stream_sha256=full.hexdigest(),smoke=smoke,native_matches_original=True,checkpoints=ck,
           first_step_nonvalue_parameter_max_difference=isolation,
           original_unchanged=True,sources={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')})
    (out/'training.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('out',type=Path);a=p.parse_args();run(a.raw.resolve(),a.out)
