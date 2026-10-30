"""CVaR-style local-constraint aggregation; all other QRL value terms unchanged."""
from functools import partial
import math

import jax
import jax.numpy as jnp
import optax

from critic import subset


def upper_tail_mean(values, fraction=.1):
    assert 0 < fraction <= 1 and values.ndim == 1
    if fraction == 1: return values.mean()
    count = int(math.ceil(fraction * values.shape[0]))
    return jax.lax.top_k(values, count)[0].mean()


def loss(agent, batch, params, mode, fraction=.1):
    if mode == 'mean': return agent.value_loss(batch, params)
    assert mode == 'tail'
    negative = agent.network.select('value')(batch['observations'], batch['value_goals'], params=params)
    positive = agent.network.select('value')(batch['observations'], batch['next_observations'], params=params)
    lam = agent.network.select('lam')(params=params)
    push = (100 * jax.nn.softplus(5 - negative / 100)).mean()
    penalty = upper_tail_mean(jnp.square(jax.nn.relu(positive - 1)), fraction)
    value = push + jax.lax.stop_gradient(lam) * penalty
    dual = lam * (agent.config['eps'] - jax.lax.stop_gradient(penalty))
    total = value + dual
    return total, dict(total_loss=total, value_loss=value, lam_loss=dual,
                       d_neg_loss=push, d_pos_loss=penalty, lam=lam)


@partial(jax.jit, static_argnames=('mode', 'fraction'))
def update(agent, opt_state, batch, mode, fraction=.1):
    trainable = subset(agent.network.params)
    def loss_fn(critic_params):
        return loss(agent, batch, {**agent.network.params, **critic_params}, mode, fraction)
    grads, info = jax.grad(loss_fn, has_aux=True)(trainable)
    updates, opt = optax.adam(agent.config['lr']).update(grads, opt_state, trainable)
    new_params = optax.apply_updates(trainable, updates)
    adam = agent.network.opt_state[0]._replace(count=opt[0].count,
        mu={**agent.network.opt_state[0].mu, **opt[0].mu}, nu={**agent.network.opt_state[0].nu, **opt[0].nu})
    network = agent.network.replace(params={**agent.network.params, **new_params},
        opt_state=(adam, agent.network.opt_state[1]), step=agent.network.step + 1)
    return agent.replace(network=network), opt, info


@jax.jit
def statistics(agent, batch):
    value = agent.network.select('value')
    pair = value(batch['observations'], batch['next_observations'])
    distance = value(batch['observations'], batch['value_goals'])
    after = value(batch['next_observations'], batch['value_goals'])
    positive = jnp.square(jax.nn.relu(pair - 1))
    return dict(pair_mean_squared_positive=positive.mean(), pair_tail_squared_positive=upper_tail_mean(positive),
        pair_q99=jnp.quantile(pair, .99), pair_max=pair.max(), pair_mean=pair.mean(),
        goal_squared_positive=jnp.square(jax.nn.relu(distance - after - 1)).mean(),
        goal_mean=distance.mean(), lam=agent.network.select('lam')())
