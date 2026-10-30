"""Replay official reset RNG consumption; caller must verify actual state fidelity."""
import numpy as np


def official_resets(env):
    np.random.seed(0);env.reset(seed=0);np.random.seed(0)
    for task in range(1,6):
        # Official evaluate constructs supply_rng once per task using this draw.
        np.random.randint(0,2**32)
        for episode in range(50):
            observation,info=env.reset(options=dict(task_id=task,render_goal=False))
            yield task,episode,observation,info
