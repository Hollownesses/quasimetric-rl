"""Validate diagnostic Q semantics and hard time-budget rejection."""
import time
import unittest
from unittest.mock import patch

from temporal_diagnostic import budget,q_values,grid_q
from diagnostics import MQEAgent,core
import numpy as np


class TemporalTests(unittest.TestCase):
    def test_q_scores_are_native_conservative_q_not_state_v(self):
        from agents.mqe import get_config
        cfg=get_config().to_dict();cfg.update(actor_hidden_dims=(16,16),value_hidden_dims=(16,16),latent_dim=16,batch_size=8)
        agent=MQEAgent.create(8,np.zeros((1,2),np.float32),np.zeros((1,2),np.float32),cfg)
        ob=np.array([1.,2.]);goal=np.array([3.,4.]);obs=np.broadcast_to(ob,core.ACTIONS.shape);goals=np.broadcast_to(goal,obs.shape)
        raw=np.asarray(agent.get_distance(obs,goals,core.ACTIONS))
        np.testing.assert_allclose(grid_q(agent,ob,goal),raw.max(axis=0),rtol=1e-6,atol=1e-6)

    def test_expired_budget_rejects_more_work(self):
        with patch('temporal_diagnostic.time.monotonic',return_value=11):
            with self.assertRaises(TimeoutError):budget(10)
            budget(12)


if __name__=='__main__':unittest.main()
