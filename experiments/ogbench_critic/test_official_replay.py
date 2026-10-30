"""Actual pinned evaluator RNG parity, plus a stochastic-step counterexample."""
import contextlib
import io
import unittest
import numpy as np
from common import UPSTREAM
from official_replay import official_resets
from utils.evaluation import evaluate


class FakeEnv:
    def __init__(self,random_step=False):self.starts=[];self.random_step=random_step
    def reset(self,seed=None,options=None):
        if seed is not None:self.rng=np.random.default_rng(seed)
        self.ob=np.random.uniform(size=2)+self.rng.uniform(size=2);goal=np.random.uniform(size=2)
        self.starts.append((self.ob.copy(),goal.copy()));return self.ob,dict(goal=goal)
    def step(self,action):
        if self.random_step:np.random.uniform()
        return self.ob,0,True,False,dict(success=1)


class Actor:
    def sample_actions(self,observations,goals,seed,temperature):return np.zeros(2)


class ReplayTests(unittest.TestCase):
    def real(self,env):
        np.random.seed(0);env.reset(seed=0);np.random.seed(0)
        with contextlib.redirect_stderr(io.StringIO()):
            for task in range(1,6):evaluate(Actor(),env,task_id=task,config=dict(discrete=False),num_eval_episodes=50,num_video_episodes=0)
        return env.starts[1:]

    def test_actual_official_evaluator_all_reset_parity(self):
        actual=self.real(FakeEnv());env=FakeEnv();expected=[(s.copy(),i['goal'].copy()) for _,_,s,i in official_resets(env)]
        self.assertEqual(len(actual),250)
        for (a,b),(c,d) in zip(actual,expected):np.testing.assert_array_equal(a,c);np.testing.assert_array_equal(b,d)

    def test_skipping_stochastic_steps_is_not_generally_valid(self):
        actual=self.real(FakeEnv(random_step=True));env=FakeEnv();replay=[(s.copy(),i['goal'].copy()) for _,_,s,i in official_resets(env)]
        self.assertTrue(any(not np.array_equal(a,c) or not np.array_equal(b,d) for (a,b),(c,d) in zip(actual,replay)))


if __name__=='__main__':unittest.main()
