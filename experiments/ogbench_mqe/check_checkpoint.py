"""Load native MQE's complete checkpoint and verify sampler/inference roundtrip."""
import csv
import hashlib
import json
from pathlib import Path
import sys

from runtime import prepare,COMMIT,BENCHMARK_COMMIT,BENCHMARK,UPSTREAM
prepare()
import jax
import numpy as np
import ogbench
from agents.mqe import MQEAgent
from utils.datasets import GCDataset
from utils.env_utils import make_env_and_datasets
from utils.flax_utils import restore_agent,save_agent


def equal_tree(a,b):
    la,ta=jax.tree_util.tree_flatten(a);lb,tb=jax.tree_util.tree_flatten(b)
    return ta==tb and all(np.array_equal(np.asarray(x),np.asarray(y)) for x,y in zip(la,lb))


def run(root,epoch):
    root=Path(root).resolve();runtime=json.loads((root/'runtime.json').read_text())
    assert runtime['exit_code']==0 and (root/'TRAINING_COMPLETE').exists()
    frozen=json.loads((root/'frozen_config.json').read_text());assert frozen['endpoint']==epoch
    dirs=list(root.glob('OGBench/*/*/flags.json'));assert len(dirs)==1,dirs
    checkpoint=dirs[0].parent;flags=json.loads(dirs[0].read_text());config=flags['agent']
    assert config==frozen['flags']['agent'] and config['alpha']==.03 and config['diag_backup']==.5
    assert flags['eval_tasks'] is None and flags['video_episodes']==0
    for key in ['actor_hidden_dims','value_hidden_dims']:config[key]=tuple(config[key])
    initial=MQEAgent.create(flags['seed'],np.zeros((1,2),np.float32),np.zeros((1,2),np.float32),config)
    agent=restore_agent(initial,str(checkpoint),epoch)
    assert int(agent.network.opt_state[0].count)==epoch
    assert int(agent.network.step)==epoch+1
    assert not equal_tree(initial.network.params,agent.network.params)
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(agent))
    output=root/'checkpoint_verification';output.mkdir(exist_ok=False)
    save_agent(agent,str(output),epoch);again=restore_agent(initial,str(output),epoch)
    assert equal_tree(again,agent)
    env,train,val=make_env_and_datasets('pointmaze-medium-navigate-v0')
    dataset=GCDataset(train,agent.config)
    np.random.seed(7781)
    last_valid=np.flatnonzero((train['terminals']>0)&(train['valids']>0))
    idxs=np.concatenate([last_valid[:128],train.get_random_idxs(128)])
    batch=dataset.sample(len(idxs),idxs=idxs)
    final=dataset.terminal_locs[np.searchsorted(dataset.terminal_locs,idxs)]
    offsets=batch['value_goals_offsets'];waypoints=batch['intermediate_value_goals_offsets']
    assert (waypoints>=0).all() and (waypoints<=offsets).all() and (idxs+offsets<=final).all()
    assert np.array_equal(batch['value_goals'],train['observations'][idxs+offsets])
    assert np.array_equal(batch['intermediate_value_goals'],train['observations'][idxs+waypoints])
    assert np.array_equal(batch['next_observations'],train['observations'][idxs+1])
    assert np.any(waypoints==0),'Official compact last-valid boundary case must be exercised.'
    obs=batch['observations'][:25];goals=batch['value_goals'][:25]
    kwargs=dict(seed=jax.random.PRNGKey(7782),temperature=0)
    actions=np.asarray(agent.sample_actions(obs,goals,**kwargs))
    assert np.array_equal(actions,np.asarray(again.sample_actions(obs,goals,**kwargs)))
    assert actions.shape==(25,2) and np.isfinite(actions).all() and np.abs(actions).max()<=1
    psi=agent.network.select('psi')(obs);psi_g=agent.network.select('psi')(goals)
    value_distance=np.asarray(agent.distance(psi,psi_g))
    q_distance=np.asarray(agent.get_distance(obs,goals,actions))
    assert value_distance.shape==q_distance.shape==(2,25)
    assert np.isfinite(value_distance).all() and np.isfinite(q_distance).all()
    with (checkpoint/'eval.csv').open(newline='') as f:evals=list(csv.DictReader(f))
    assert int(float(evals[-1]['step']))==epoch
    success_keys=[k for k in evals[-1] if k.startswith('evaluation/') and k.endswith('_success')]
    assert len(success_keys)==6
    assert Path(ogbench.__file__).resolve().is_relative_to(BENCHMARK.resolve())
    assert Path(sys.modules['agents.mqe'].__file__).resolve().is_relative_to(UPSTREAM.resolve())
    result=dict(passed=True,mqe_commit=COMMIT,benchmark_commit=BENCHMARK_COMMIT,
        checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256((checkpoint/f'params_{epoch}.pkl').read_bytes()).hexdigest(),
        adam_count=int(agent.network.opt_state[0].count),full_state_roundtrip=True,actor_actions_equal=True,
        native_sampler_boundaries_verified=True,boundary_rows=int((waypoints==0).sum()),
        train_stored_states=train.size,train_valid_transitions=len(train.valid_idxs),validation_stored_states=val.size,
        benchmark_import=str(ogbench.__file__),agent_import=str(sys.modules['agents.mqe'].__file__),
        evaluation_endpoint=evals[-1],seconds=runtime['elapsed_seconds'],devices=[str(d) for d in jax.devices()])
    env.close();(root/'checkpoint_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':run(sys.argv[1],int(sys.argv[2]))
