"""Actual trajectory upper bounds with unchanged official batch/RNG semantics."""
import jax
import jax.numpy as jnp
import numpy as np
import optax

from critic import subset

HORIZONS=np.array([2,4,8,16,32,64],np.int64)


def sample_with_indices(dataset,batch_size):
    state=np.random.get_state()
    try:indices=dataset.dataset.get_random_idxs(batch_size)
    finally:np.random.set_state(state)
    batch=dataset.sample(batch_size)
    np.testing.assert_array_equal(batch['observations'],dataset.get_observations(indices))
    return batch,indices


def requested_horizons(step,size):
    return HORIZONS[(step*size+np.arange(size))%len(HORIZONS)]


class PathBounds:
    def __init__(self,dataset):
        assert dataset.config['frame_stack'] is None and 'next_observations' not in dataset.dataset
        assert dataset.dataset['observations'].ndim==2 and dataset.dataset['observations'].shape[1]==2
        self.dataset=dataset
        self.invalid_prefix=np.r_[0,np.cumsum(dataset.dataset['valids']<=0)]

    def endpoints(self,indices,horizons):
        indices=np.asarray(indices);horizons=np.asarray(horizons)
        assert indices.shape==horizons.shape and (horizons>=0).all()
        assert indices.dtype.kind in 'iu' and horizons.dtype.kind in 'iu'
        assert ((indices>=0)&(indices<self.dataset.size)).all()
        assert (self.dataset.dataset['valids'][indices]>0).all()
        final=self.dataset.terminal_locs[np.searchsorted(self.dataset.terminal_locs,indices)]
        ends=np.minimum(indices+horizons,final)
        assert (ends>=indices).all()
        assert np.all(self.invalid_prefix[ends]-self.invalid_prefix[indices]==0),'Path crosses invalid source/episode boundary'
        return self.dataset.get_observations(ends),(ends-indices).astype(np.float32),ends


def loss(agent,batch,params,path_goals,path_costs):
    value=agent.network.select('value')
    d_neg=value(batch['observations'],batch['value_goals'],params=params)
    d_pair=value(batch['observations'],batch['next_observations'],params=params)
    d_path=value(batch['observations'],path_goals,params=params)
    pair=jnp.square(jax.nn.relu(d_pair-1));path=jnp.square(jax.nn.relu(d_path-path_costs))
    penalty=jnp.maximum(pair,path).mean()
    push=(100*jax.nn.softplus(5-d_neg/100)).mean();lam=agent.network.select('lam')(params=params)
    primal=push+penalty*jax.lax.stop_gradient(lam)
    dual=lam*(agent.config['eps']-jax.lax.stop_gradient(penalty))
    return primal+dual,dict(total_loss=primal+dual,value_loss=primal,lam_loss=dual,d_neg_loss=push,
        d_pos_loss=penalty,pair_loss=pair.mean(),path_loss=path.mean(),path_active_fraction=(path>pair).mean(),
        path_cost_mean=path_costs.mean(),path_zero_fraction=(path_costs==0).mean(),path_distance_mean=d_path.mean(),lam=lam)


@jax.jit
def update(agent,opt_state,batch,path_goals,path_costs):
    params=subset(agent.network.params)
    def loss_fn(trainable):return loss(agent,batch,{**agent.network.params,**trainable},path_goals,path_costs)
    grads,info=jax.grad(loss_fn,has_aux=True)(params)
    updates,opt=optax.adam(agent.config['lr']).update(grads,opt_state,params)
    trained=optax.apply_updates(params,updates)
    adam=agent.network.opt_state[0]._replace(count=opt[0].count,
        mu={**agent.network.opt_state[0].mu,**opt[0].mu},nu={**agent.network.opt_state[0].nu,**opt[0].nu})
    network=agent.network.replace(params={**agent.network.params,**trained},opt_state=(adam,agent.network.opt_state[1]),step=agent.network.step+1)
    return agent.replace(network=network),opt,info


@jax.jit
def statistics(agent,batch,path_goals,path_costs):
    return loss(agent,batch,agent.network.params,path_goals,path_costs)[1]
