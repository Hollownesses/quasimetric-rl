"""MQE-style IQE critic with an exactly matched one-step control."""
import copy
from functools import partial
from pathlib import Path
import pickle
from typing import Any

from common import BASE_SHA,load_base
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from utils.networks import MLP

GAMMA=.995
WAYPOINT_DISCOUNT=.95
NEXT_PROB=.2
DIAG_WEIGHT=.5
CLIP=5.
# Uniform batch-goal draws include the diagonal, matching the full-pair mean.
OFFDIAG=16
BATCH=256
UNIT=-np.log(GAMMA)


@flax.struct.dataclass
class BackupState:
    params:Any
    opt_state:Any
    step:Any


def encoder(base):
    return MLP((*base.config['value_hidden_dims'],base.config['latent_dim']),
               activate_final=False,layer_norm=base.config['layer_norm'])


def initialize(base):
    value=copy.deepcopy(base.network.params['modules_value'])
    phi=value['phi']
    last=max((k for k in phi if k.startswith('Dense_')),key=lambda k:int(k.split('_')[-1]))
    phi[last]=jax.tree_util.tree_map(lambda x:x*UNIT,phi[last])
    q=copy.deepcopy(phi)
    q['Dense_0']['kernel']=jnp.concatenate([q['Dense_0']['kernel'],
        jnp.zeros((2,q['Dense_0']['kernel'].shape[1]),dtype=q['Dense_0']['kernel'].dtype)],axis=0)
    params=dict(value=value,q=q)
    return BackupState(params,optax.adam(base.config['lr']).init(params),jnp.array(0))


def as_agent(base,state):
    # Inference view only. The authoritative optimizer is BackupState.opt_state.
    return base.replace(network=base.network.replace(params={**base.network.params,
        'modules_value':state.params['value']}))


def embeddings(base,params,observations,goals):
    full={**base.network.params,'modules_value':params['value']}
    return base.network.select('value')(observations,goals,info=True,params=full)


def iqe(base,params,x,y):
    return base.network.select('value')(x,y,is_phi=True,
        params={**base.network.params,'modules_value':params['value']})


def loss(base,params,batch,mode):
    if mode not in ['one','multi']:raise ValueError(mode)
    _,psi_s,psi_g=embeddings(base,params,batch['observations'],batch['value_goals'])
    phi=encoder(base).apply({'params':params['q']},jnp.concatenate([batch['observations'],batch['actions']],axis=-1))
    w=batch['next_observations'] if mode=='one' else batch['waypoint']
    k=jnp.ones(len(w)) if mode=='one' else batch['offsets']
    _,psi_w,_=embeddings(base,params,w,batch['value_goals'])
    chosen=psi_g[batch['goal_choices']]
    dist=iqe(base,params,phi[:,None,:],chosen)
    target=jax.lax.stop_gradient(iqe(base,params,psi_w[:,None,:],chosen))
    delta=dist-target
    divergence=jnp.where(delta>CLIP,delta,GAMMA**k[:,None]*jnp.exp(jnp.minimum(delta,CLIP))-dist)
    backup=DIAG_WEIGHT*divergence[:,0].mean()+(1-DIAG_WEIGHT)*divergence[:,1:].mean()
    invariance=jnp.square(jnp.exp(-iqe(base,params,psi_s,phi))-1).mean()
    return backup+invariance,dict(loss=backup+invariance,backup=backup,invariance=invariance,
        mean_distance=dist.mean(),mean_k=k.mean(),zero_k=(k==0).mean(),clipped_fraction=(delta>CLIP).mean())


@partial(jax.jit,static_argnames=('mode',))
def update(base,state,batch,mode):
    grads,info=jax.grad(lambda p:loss(base,p,batch,mode),has_aux=True)(state.params)
    delta,opt=optax.adam(base.config['lr']).update(grads,state.opt_state,state.params)
    return state.replace(params=optax.apply_updates(state.params,delta),opt_state=opt,step=state.step+1),info


def sample(dataset,rng,batch_size=BATCH):
    ds=dataset.dataset
    idx=ds.valid_idxs[rng.randint(len(ds.valid_idxs),size=batch_size)]
    batch=ds.sample(batch_size,idx)
    final=dataset.terminal_locs[np.searchsorted(dataset.terminal_locs,idx)]
    g_offset=np.minimum(rng.geometric(1-GAMMA,size=batch_size),final-idx)
    goal=idx+g_offset
    offset=np.minimum(rng.geometric(1-WAYPOINT_DISCOUNT,size=batch_size),g_offset)
    take_next=rng.rand(batch_size)<NEXT_PROB
    waypoint=np.where(take_next,idx+1,idx+offset)
    offset=np.where(take_next,1,offset)
    choices=np.column_stack([np.arange(batch_size),rng.randint(batch_size,size=(batch_size,OFFDIAG))])
    batch.update(value_goals=ds['observations'][goal],waypoint=ds['observations'][waypoint],
        offsets=offset.astype(np.float32),goal_choices=choices.astype(np.int32),
        source_indices=idx,goal_indices=goal,waypoint_indices=waypoint,final_indices=final)
    return batch


@jax.jit
def diagnostics(base,state,batch):
    params=state.params
    d,_,_=embeddings(base,params,batch['observations'],batch['value_goals'])
    nd,_,_=embeddings(base,params,batch['next_observations'],batch['value_goals'])
    pair,_,_=embeddings(base,params,batch['observations'],batch['next_observations'])
    return dict(pair_squared_positive_step_units=jnp.square(jnp.maximum(pair/UNIT-1,0)).mean(),
        goal_squared_positive_step_units=jnp.square(jnp.maximum((d-nd)/UNIT-1,0)).mean(),
        mean_goal_log_units=d.mean(),mean_goal_step_units=(d/UNIT).mean(),
        goal_uphill_fraction=(nd>d).mean())


def save(base,state,folder,mode,rng):
    folder=Path(folder);folder.mkdir(exist_ok=True)
    path=folder/f'critic_{int(state.step)}.pkl'
    bundle=dict(base_sha256=BASE_SHA,mode=mode,extra_updates=int(state.step),
        unit=UNIT,state=flax.serialization.to_state_dict(state),rng=rng.get_state())
    with path.open('xb') as f:pickle.dump(bundle,f)
    return path


def load_state(base,folder,step):
    path=Path(folder)/f'critic_{step}.pkl'
    with path.open('rb') as f:bundle=pickle.load(f)
    assert bundle['base_sha256']==BASE_SHA and bundle['extra_updates']==step
    assert bundle['unit']==UNIT and bundle['mode'] in ['one','multi']
    state=flax.serialization.from_state_dict(initialize(base),bundle['state'])
    assert int(state.step)==step and int(state.opt_state[0].count)==step
    return state,bundle


def load_for_evaluation(base,folder,step):
    state,bundle=load_state(base,folder,step)
    return as_agent(base,state),dict(checkpoint_kind='multistep',mode=bundle['mode'],
        training_extra_updates=step,distance_unit='log-discount',unit=UNIT,
        inference_optimizer_note='Original agent optimizer is an unused inference view; use critic bundle to resume.')
