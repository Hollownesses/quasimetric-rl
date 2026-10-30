"""Official module losses, scheduled updates with all other Adam leaves frozen."""
from functools import partial
import jax
import optax
from common import tree_digest


def module_key(stage):
    assert stage in ['dynamics','actor'];return 'modules_'+stage


def proof(agent,keys):
    adam=agent.network.opt_state[0]
    return {k:dict(params=tree_digest(agent.network.params[k]),mu=tree_digest(adam.mu[k]),nu=tree_digest(adam.nu[k])) for k in keys}


def init_optimizer(agent,stage):
    key=module_key(stage);adam=agent.network.opt_state[0]
    return (adam._replace(mu={key:adam.mu[key]},nu={key:adam.nu[key]}),agent.network.opt_state[1])


def official_loss(agent,batch,params,stage,rng):
    if stage=='dynamics':return agent.dynamics_loss(batch,params)
    assert stage=='actor';_,actor_rng=jax.random.split(rng)
    return agent.actor_loss(batch,params,rng=actor_rng)


@partial(jax.jit,static_argnames=('stage',))
def update(agent,opt_state,batch,stage):
    key=module_key(stage);trainable={key:agent.network.params[key]};new_rng,rng=jax.random.split(agent.rng)
    def fn(params):return official_loss(agent,batch,{**agent.network.params,**params},stage,rng)
    grads,info=jax.grad(fn,has_aux=True)(trainable)
    updates,opt=optax.adam(agent.config['lr']).update(grads,opt_state,trainable);learned=optax.apply_updates(trainable,updates)
    adam=agent.network.opt_state[0]._replace(count=opt[0].count,mu={**agent.network.opt_state[0].mu,**opt[0].mu},nu={**agent.network.opt_state[0].nu,**opt[0].nu})
    network=agent.network.replace(params={**agent.network.params,**learned},opt_state=(adam,agent.network.opt_state[1]),step=agent.network.step+1)
    return agent.replace(network=network,rng=new_rng),opt,info


@jax.jit
def statistics(agent,batch):
    dyn,di=agent.dynamics_loss(batch,agent.network.params);actor,ai=agent.actor_loss(batch,agent.network.params,rng=agent.rng)
    return dict(dynamics_loss=dyn,**{'actor/'+k:v for k,v in ai.items()})
