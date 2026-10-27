"""Verify actual saved optimizer step, update, full state round-trip and episode."""
import json
import hashlib
from pathlib import Path
import random
import sys

from runtime import prepare

prepare()
import jax
import numpy as np
from agents.qrl import QRLAgent
from utils.datasets import Dataset, GCDataset
from utils.env_utils import make_env_and_datasets
from utils.evaluation import evaluate
from utils.flax_utils import restore_agent, save_agent

run, epoch = Path(sys.argv[1]).resolve(), int(sys.argv[2])
directories = [p for p in run.glob('OGBench/*/*') if p.is_dir() and (p/'flags.json').is_file()]
assert len(directories) == 1, directories
directory = directories[0]
flags = json.loads((directory/'flags.json').read_text())
config = flags['agent']
for key in ('actor_hidden_dims', 'value_hidden_dims'):
    config[key] = tuple(config[key])
env, train, _ = make_env_and_datasets(flags['env_name'])
random.seed(0)
np.random.seed(0)
dataset = GCDataset(Dataset.create(**train), config)
batch = dataset.sample(1)
initial = QRLAgent.create(0, batch['observations'], batch['actions'], config)
restored = restore_agent(initial, str(directory), epoch)
assert int(restored.network.step) == int(initial.network.step) + epoch
assert int(restored.network.opt_state[0].count) == epoch
changed = any(not np.array_equal(a,b) for a,b in zip(
    jax.tree_util.tree_leaves(initial.network.params), jax.tree_util.tree_leaves(restored.network.params)))
assert changed
finite = all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(restored))
assert finite
roundtrip = run/'checkpoint_verification'
roundtrip.mkdir(exist_ok=True)
save_agent(restored, str(roundtrip), epoch)
again = restore_agent(initial, str(roundtrip), epoch)
equal = all(np.array_equal(a,b) for a,b in zip(jax.tree_util.tree_leaves(restored),jax.tree_util.tree_leaves(again)))
assert equal
stats, trajs, _ = evaluate(again, env, task_id=1, config=config, num_eval_episodes=1, num_video_episodes=0)
assert len(trajs)==1 and trajs[0]['done'][-1]
result = dict(passed=True, optimizer_step=int(restored.network.opt_state[0].count),
              official_trainstate_step=int(restored.network.step), parameters_changed=changed,
              all_state_leaves_finite=finite, jax_version=jax.__version__,
              jax_backend=jax.default_backend(), devices=[str(x) for x in jax.devices()],
              full_state_roundtrip_equal=equal, complete_episode_steps=len(trajs[0]['action']),
              success=float(stats['success']), checkpoint=str(directory/f'params_{epoch}.pkl'),
              checkpoint_sha256=hashlib.sha256((directory/f'params_{epoch}.pkl').read_bytes()).hexdigest(),
              verification_scope='One additional task1 episode; excluded from formal endpoint metrics.')
(run/'checkpoint_verification.json').write_text(json.dumps(result,indent=2)+'\n')
env.close()
print(json.dumps(result,indent=2))
