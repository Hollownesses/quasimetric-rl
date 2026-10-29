import unittest
import numpy as np
from common import load_base,load_data
from evaluate import capture,restore,candidates,ACTIONS


class RestoreTests(unittest.TestCase):
    def test_full_state_restore_at_wall_contact(self):
        _,config=load_base()
        env,_,_=load_data(config)
        env.reset(seed=3100,options=dict(task_id=3))
        env.unwrapped.set_xy(np.array([10.1156,14.6827]))
        for _ in range(10): env.step(np.array([-.0293,-.1729]))
        state=capture(env)
        act=np.array([-.0293,-.1729])
        ref=env.step(act)
        restore(env,state)
        for other in [np.array([1.,1.]),np.array([-1.,-1.]),np.zeros(2)]:
            env.step(other)
            restore(env,state)
            actual=env.step(act)
            self.assertTrue(np.array_equal(ref[0],actual[0]))
            self.assertEqual(ref[1:4],actual[1:4])
            restore(env,state)
        obs,_=candidates(env,ACTIONS)
        for i in [0,60,120]:
            restore(env,state)
            self.assertTrue(np.array_equal(obs[i],env.step(ACTIONS[i])[0]))
        env.close()


if __name__=='__main__': unittest.main()
