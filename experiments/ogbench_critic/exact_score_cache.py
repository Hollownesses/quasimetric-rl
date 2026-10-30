"""Memoize a frozen QRL score at identical effective JAX float32 inputs."""
from collections import OrderedDict

from common import distances
import jax
import numpy as np


class ExactScoreCache:
    def __init__(self, agent, capacity=2048, audit=True):
        assert not jax.config.x64_enabled
        assert agent.config['encoder'] is None
        assert capacity > 0
        self.agent = agent; self.capacity = capacity; self.audit = audit
        self.entries = OrderedDict(); self.calls = self.hits = self.evaluations = self.audits = 0

    def __call__(self, observations, goals):
        self.calls += 1
        observations, goals = np.asarray(observations), np.asarray(goals)
        assert observations.shape == goals.shape and observations.ndim == 2 and observations.shape[1] == 2
        assert observations.dtype in (np.dtype('float32'), np.dtype('float64'))
        assert goals.dtype in (np.dtype('float32'), np.dtype('float64'))
        assert np.isfinite(observations).all() and np.isfinite(goals).all()
        key = (observations.shape, observations.astype(np.float32).tobytes(), goals.astype(np.float32).tobytes())
        if key in self.entries:
            self.hits += 1; values, checked = self.entries.pop(key)
            if self.audit and not checked:
                reference = np.asarray(distances(self.agent, observations, goals))
                self.evaluations += 1; self.audits += 1
                np.testing.assert_array_equal(values, reference)
                checked = True
            self.entries[key] = (values, checked)
            return values.copy()
        values = np.asarray(distances(self.agent, observations, goals)).copy()
        assert np.isfinite(values).all()
        self.evaluations += 1; self.entries[key] = (values, False)
        if len(self.entries) > self.capacity: self.entries.popitem(last=False)
        return values.copy()

    def statistics(self):
        return dict(calls=self.calls, hits=self.hits, evaluations=self.evaluations,
                    first_hit_audits=self.audits, capacity=self.capacity, entries=len(self.entries))
