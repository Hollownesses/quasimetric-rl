# gym/__init__.py
"""
Gym shim: redirect `import gym` to `gymnasium` for compatibility.

- 支持:
  - import gym
  - import gym.spaces
  - import gym.wrappers
  - gym.make, gym.Env, gym.spaces, gym.wrappers
"""

import sys
import gymnasium as _gym

# 1. 把 gymnasium 的常用接口 re-export 出去
from gymnasium import *  # noqa: F401,F403
__version__ = _gym.__version__

spaces = _gym.spaces
wrappers = _gym.wrappers
Env = _gym.Env

# 2. 关键：把子模块注册进 sys.modules，让 `import gym.spaces` / `import gym.wrappers` 生效
sys.modules[__name__ + ".spaces"] = _gym.spaces
sys.modules[__name__ + ".wrappers"] = _gym.wrappers