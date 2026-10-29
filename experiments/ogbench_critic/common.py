"""Pinned official checkpoint/data loading for critic-only experiments."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments/ogbench_official_qrl'))
from runtime import prepare, UPSTREAM
prepare()

import jax
import numpy as np
from agents.qrl import QRLAgent
from utils.flax_utils import restore_agent
from utils.env_utils import make_env_and_datasets
from utils.datasets import Dataset, GCDataset

BASE = ROOT/'results/ogbench_official_qrl_20261026_seed0/main/OGBench/navigate_100k/sd000_20261026_115628'
BASE_SHA = '397e919f2c453cee02ec840c615dce3532f31532fd639fbea6757c970fedbf0f'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_base():
    assert sha256(BASE/'params_100000.pkl') == BASE_SHA
    config = json.loads((BASE/'flags.json').read_text())['agent']
    for key in ['actor_hidden_dims', 'value_hidden_dims']:
        config[key] = tuple(config[key])
    initial = QRLAgent.create(0, np.zeros((1,2),np.float32), np.zeros((1,2),np.float32), config)
    agent = restore_agent(initial, str(BASE), 100000)
    assert int(agent.network.opt_state[0].count) == 100000
    return agent, config


def load_data(config):
    env, train, val = make_env_and_datasets('pointmaze-medium-navigate-v0')
    return env, GCDataset(Dataset.create(**train), config), GCDataset(Dataset.create(**val), config)


def tree_equal(a,b):
    return jax.tree_util.tree_structure(a)==jax.tree_util.tree_structure(b) and all(
        np.array_equal(x,y) for x,y in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)))


def tree_digest(tree):
    h = hashlib.sha256()
    for x in jax.tree_util.tree_leaves(tree):
        a=np.asarray(x)
        h.update(str((a.shape,str(a.dtype))).encode())
        h.update(a.tobytes())
    return h.hexdigest()


@jax.jit
def distances(agent, observations, goals):
    return agent.network.select('value')(observations, goals)
