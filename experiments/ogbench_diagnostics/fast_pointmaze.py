"""Exact native MuJoCo candidate batching for the pinned state PointMaze only."""
import types

import mujoco
from mujoco.rollout import Rollout
import numpy as np
from ogbench.locomaze.point import PointEnv

from . import core


class FastPointmaze:
    def __init__(self, env, nthread=4):
        point = env.unwrapped; model = point.model
        assert isinstance(point, PointEnv) and point._teleport_info is None
        assert 'pointmaze-medium' in env.spec.id
        assert model.nq == model.nv == 2 and model.na == model.nplugin == model.nmocap == 0
        assert point.frame_skip == 5 and point.get_ob().shape == (2,)
        np.testing.assert_array_equal(point.get_ob(), point.data.qpos)
        self.full = mujoco.mjtState.mjSTATE_FULLPHYSICS
        self.user = mujoco.mjtState.mjSTATE_USER
        self.nfull = mujoco.mj_stateSize(model, self.full)
        self.nuser = mujoco.mj_stateSize(model, self.user)
        assert self.nfull == 5
        assert self.nfull + self.nuser + model.nv == mujoco.mj_stateSize(model, core.STATE_SPEC)
        assert nthread in (0, 4)
        self.env = env; self.model = model
        self.data = [mujoco.MjData(model) for _ in range(nthread)] if nthread else mujoco.MjData(model)
        self.engine = Rollout(nthread=nthread)
        self.states = np.empty((121, 5, self.nfull))
        self.sensors = np.empty((121, 5, model.nsensordata))
        self.initial = np.empty((121, self.nfull))
        self.control = np.empty((1, 1, self.nuser))
        self.full_state = np.empty(self.nfull)
        # Preserve the exact original episode loop in independent globals.
        self.bound_rollout = types.FunctionType(core.rollout.__code__,
            {**core.rollout.__globals__, 'candidates': self.candidates},
            name='pointmaze_native_candidates', argdefs=core.rollout.__defaults__, closure=core.rollout.__closure__)

    def candidates(self, env, actions=core.ACTIONS):
        assert env is self.env and np.array_equal(actions, core.ACTIONS)
        before = core.capture(env)
        mujoco.mj_getState(self.model, env.unwrapped.data, self.full_state, self.full)
        mujoco.mj_getState(self.model, env.unwrapped.data, self.control[0, 0], self.user)
        self.initial[:] = self.full_state
        self.initial[:, 1:3] += .2 * actions
        self.initial[:, 3:5] = 0
        self.engine.rollout(self.model, self.data, self.initial, self.control,
            control_spec=self.user, nstep=5, initial_warmstart=env.unwrapped.data.qacc_warmstart[None],
            state=self.states, sensordata=self.sensors)
        return self.states[:, -1, 1:3].copy(), before

    def rollout(self, case, score):
        return self.bound_rollout(self.env, case, score=score)

    def close(self):
        self.engine.close()
