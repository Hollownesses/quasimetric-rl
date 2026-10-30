"""Population-normalized complete-observation cell importance weighting."""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax

from critic import subset


@dataclass
class SourceCells:
    lower: np.ndarray
    counts: np.ndarray
    total: int
    occupied: int
    spacing: float = .2

    @classmethod
    def build(cls, observations):
        assert observations.ndim == 2 and observations.shape[1] == 2 and observations.dtype == np.float32
        assert np.isfinite(observations).all()
        grid, counts = np.unique(np.rint(observations / .2).astype(np.int32), axis=0, return_counts=True)
        lower = grid.min(axis=0); table = np.zeros(grid.max(axis=0) - lower + 1, np.int64)
        table[tuple((grid - lower).T)] = counts
        return cls(lower, table, len(observations), len(grid))

    def weights(self, observations):
        assert observations.ndim == 2 and observations.shape[1] == 2 and observations.dtype == np.float32
        grid = np.rint(observations / self.spacing).astype(np.int32) - self.lower
        assert ((grid >= 0) & (grid < self.counts.shape)).all()
        count = self.counts[tuple(grid.T)]; assert (count > 0).all()
        return (self.total / (self.occupied * count)).astype(np.float32)

    def save(self, path):
        np.savez_compressed(path, lower=self.lower, counts=self.counts, total=self.total,
                            occupied=self.occupied, spacing=self.spacing)

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            return cls(data['lower'].copy(), data['counts'].copy(), int(data['total']), int(data['occupied']), float(data['spacing']))


def loss(agent, batch, params, weights):
    assert weights.shape == batch['observations'].shape[:1]
    value = agent.network.select('value')
    negative = value(batch['observations'], batch['value_goals'], params=params)
    positive = value(batch['observations'], batch['next_observations'], params=params)
    lam = agent.network.select('lam')(params=params)
    push = (weights * (100 * jax.nn.softplus(5 - negative / 100))).mean()
    penalty = (weights * jnp.square(jax.nn.relu(positive - 1))).mean()
    primal = push + penalty * jax.lax.stop_gradient(lam)
    dual = lam * (agent.config['eps'] - jax.lax.stop_gradient(penalty))
    return primal + dual, dict(total_loss=primal + dual, value_loss=primal, lam_loss=dual,
        d_neg_loss=push, d_pos_loss=penalty, lam=lam,
        weight_mean=weights.mean(), weight_max=weights.max(),
        weight_effective_samples=weights.sum() ** 2 / jnp.square(weights).sum())


@jax.jit
def update(agent, opt_state, batch, weights):
    params = subset(agent.network.params)
    def loss_fn(trainable): return loss(agent, batch, {**agent.network.params, **trainable}, weights)
    grads, info = jax.grad(loss_fn, has_aux=True)(params)
    updates, opt = optax.adam(agent.config['lr']).update(grads, opt_state, params)
    trained = optax.apply_updates(params, updates)
    adam = agent.network.opt_state[0]._replace(count=opt[0].count,
        mu={**agent.network.opt_state[0].mu, **opt[0].mu}, nu={**agent.network.opt_state[0].nu, **opt[0].nu})
    network = agent.network.replace(params={**agent.network.params, **trained},
        opt_state=(adam, agent.network.opt_state[1]), step=agent.network.step + 1)
    return agent.replace(network=network), opt, info
