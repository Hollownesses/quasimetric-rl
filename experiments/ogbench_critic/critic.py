"""Only value and dual parameters are optimized; actor/dynamics stay immutable."""
from functools import partial
import flax
import jax
import jax.numpy as jnp
import optax

CRITIC_KEYS = ('modules_value', 'modules_lam')


def subset(params):
    return {key:params[key] for key in CRITIC_KEYS}


def init_optimizer(agent):
    old = agent.network.opt_state
    # Preserve exactly the pretrained Adam moments and bias-correction count.
    adam = old[0]._replace(mu=subset(old[0].mu), nu=subset(old[0].nu))
    return (adam, old[1])


def residuals(agent, batch, params=None):
    d = partial(agent.network.select('value'), params=params)
    sg = d(batch['observations'],batch['value_goals'])
    ng = d(batch['next_observations'],batch['value_goals'])
    pair = d(batch['observations'],batch['next_observations'])
    return sg, pair-1, sg-ng-1


def critic_loss(agent, batch, params, mode):
    if mode == 'pair':
        loss, info = agent.value_loss(batch, params)
        # This branch is the official value loss verbatim.
        return loss, info
    # Direct gradient through both endpoint distances; no target network.
    sg, pair_h, goal_h = residuals(agent,batch,params)
    negative = (100*jax.nn.softplus(5-sg/100)).mean()
    positive = jnp.square(jax.nn.relu(goal_h)).mean()
    lam = agent.network.select('lam')(params=params)
    value = negative + positive*jax.lax.stop_gradient(lam)
    dual = lam*(agent.config['eps']-jax.lax.stop_gradient(positive))
    return value+dual, dict(total_loss=value+dual, value_loss=value, lam_loss=dual,
        d_neg_loss=negative, d_pos_loss=positive, lam=lam)


@partial(jax.jit, static_argnames=('mode',))
def update(agent, opt_state, batch, mode):
    trainable=subset(agent.network.params)

    def loss_fn(critic_params):
        params={**agent.network.params,**critic_params}
        return critic_loss(agent,batch,params,mode)

    grads,info=jax.grad(loss_fn,has_aux=True)(trainable)
    tx=optax.adam(agent.config['lr'])
    updates,new_opt=tx.update(grads,opt_state,trainable)
    new_params=optax.apply_updates(trainable,updates)
    # Frozen parameter leaves never pass through Adam, including its momentum.
    full_params={**agent.network.params,**new_params}
    full_adam=agent.network.opt_state[0]._replace(count=new_opt[0].count,
        mu={**agent.network.opt_state[0].mu,**new_opt[0].mu},
        nu={**agent.network.opt_state[0].nu,**new_opt[0].nu})
    full_opt=(full_adam,agent.network.opt_state[1])
    network=agent.network.replace(params=full_params,opt_state=full_opt,step=agent.network.step+1)
    return agent.replace(network=network),new_opt,info


@jax.jit
def diagnostics(agent,batch):
    sg,pair_h,goal_h=residuals(agent,batch)
    return dict(pair_squared_positive=jnp.square(jax.nn.relu(pair_h)).mean(),
        goal_squared_positive=jnp.square(jax.nn.relu(goal_h)).mean(),
        pair_positive_fraction=(pair_h>0).mean(), goal_positive_fraction=(goal_h>0).mean(),
        goal_uphill_fraction=(goal_h < -1).mean(), d_goal_mean=sg.mean(),
        pair_residual_mean=pair_h.mean(), goal_residual_mean=goal_h.mean(),
        lam=agent.network.select('lam')())
