"""Supervised frozen-teacher distance fitting, updating original IQE only."""
import jax
import jax.numpy as jnp
import optax

from common import tree_digest


def frozen_proof(agent):
    adam=agent.network.opt_state[0]
    return {k:dict(params=tree_digest(agent.network.params[k]),mu=tree_digest(adam.mu[k]),nu=tree_digest(adam.nu[k]))
            for k in ['modules_actor','modules_dynamics','modules_lam']}


def init_optimizer(agent):
    adam=agent.network.opt_state[0]
    return (adam._replace(mu={'modules_value':adam.mu['modules_value']},nu={'modules_value':adam.nu['modules_value']}),agent.network.opt_state[1])


def loss(agent,batch,params,targets):
    value=agent.network.select('value')
    forward=value(batch['observations'],batch['value_goals'],params=params)
    reverse=value(batch['value_goals'],batch['observations'],params=params)
    f_error=forward-targets;r_error=reverse-targets
    mse=.5*(jnp.square(f_error)+jnp.square(r_error)).mean()
    return mse,dict(loss=mse,forward_rmse=jnp.sqrt(jnp.square(f_error).mean()),
        reverse_rmse=jnp.sqrt(jnp.square(r_error).mean()),target_mean=targets.mean(),
        forward_mean=forward.mean(),reverse_mean=reverse.mean(),absolute_direction_gap=jnp.abs(forward-reverse).mean())


@jax.jit
def update(agent,opt_state,batch,targets):
    trainable={'modules_value':agent.network.params['modules_value']}
    def loss_fn(p):return loss(agent,batch,{**agent.network.params,**p},targets)
    grads,info=jax.grad(loss_fn,has_aux=True)(trainable)
    updates,opt=optax.adam(agent.config['lr']).update(grads,opt_state,trainable)
    learned=optax.apply_updates(trainable,updates)
    adam=agent.network.opt_state[0]._replace(count=opt[0].count,
        mu={**agent.network.opt_state[0].mu,**opt[0].mu},nu={**agent.network.opt_state[0].nu,**opt[0].nu})
    network=agent.network.replace(params={**agent.network.params,**learned},opt_state=(adam,agent.network.opt_state[1]),step=agent.network.step+1)
    return agent.replace(network=network),opt,info


@jax.jit
def statistics(agent,batch,targets):return loss(agent,batch,agent.network.params,targets)[1]
