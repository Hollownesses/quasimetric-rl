"""Signed teacher finite-difference matching on fixed original IQE queries."""
import jax
import jax.numpy as jnp
import optax
from iqe_distill import init_optimizer,frozen_proof


def loss(agent,batch,params,targets):
    value=agent.network.select('value');source=batch['observations'];goal=batch['value_goals'];successor=batch['next_observations']
    f=value(source,goal,params=params);r=value(goal,source,params=params)
    fn=value(successor,goal,params=params);rn=value(goal,successor,params=params)
    fe=f-targets[:,0];re=r-targets[:,0];delta=targets[:,0]-targets[:,1]
    fd=(f-fn-delta)/.2;rd=(r-rn-delta)/.2
    absolute=.5*(jnp.square(fe)+jnp.square(re)).mean();shape=.5*(jnp.square(fd)+jnp.square(rd)).mean()
    return absolute+shape,dict(loss=absolute+shape,absolute_loss=absolute,shape_loss=shape,
        forward_rmse=jnp.sqrt(jnp.square(fe).mean()),reverse_rmse=jnp.sqrt(jnp.square(re).mean()),
        forward_shape_rmse=jnp.sqrt(jnp.square(fd).mean()),reverse_shape_rmse=jnp.sqrt(jnp.square(rd).mean()),
        target_mean=targets[:,0].mean(),target_delta_mean=delta.mean(),forward_mean=f.mean(),reverse_mean=r.mean(),
        absolute_direction_gap=jnp.abs(f-r).mean())


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
