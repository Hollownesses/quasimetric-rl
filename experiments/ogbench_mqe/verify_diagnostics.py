"""Read-only integrity checks for fixed-bank native MQE diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

UNIT=-np.log(.995)
MODES=('ensemble0','ensemble1','max')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(output,bank,baseline_field,smoke=False):
    output=Path(output).resolve();summary=json.loads((output/'summary.json').read_text());protocol=summary['protocol']
    assert summary['complete'] and protocol['bank_sha256']==sha(bank)
    assert protocol['epoch']==(2 if smoke else 100000)
    assert protocol['primary_value_aggregation']=='max' and protocol['unit']==UNIT
    with np.load(bank) as data:names=sorted({k.split('__')[0] for k in data.files})
    assert len(names)==25
    if smoke:names=['task3_seed3100']
    assert protocol['cases']==names
    checkpoint=Path(protocol['checkpoint'])/f'params_{protocol["epoch"]}.pkl'
    assert sha(checkpoint)==protocol['checkpoint_sha256']
    for mode in [*MODES,'actor']:
        rollout=summary['rollouts'][mode];directory=output/f'{mode}_rollout'
        assert rollout==json.loads((directory/'summary.json').read_text())
        assert list(rollout['cases'])==names
        for key in ['checkpoint_sha256','network_digest','psi_digest','phi_digest','bank_sha256','action_grid_sha256']:
            assert rollout[key]==protocol[key],(mode,key)
        assert rollout['epoch']==protocol['epoch']
        assert rollout['aggregation']==(None if mode=='actor' else mode)
        assert rollout['controller']==('native-actor' if mode=='actor' else 'real-greedy')
        for name,info in rollout['cases'].items():
            assert isinstance(info['success'],bool) and 1<=info['steps']<=1000
            with np.load(directory/f'{name}.npz') as trajectory:
                n=info['steps'];assert trajectory['observations'].shape==(n+1,2) and trajectory['actions'].shape==(n,2)
                assert trajectory['distances'].shape==((0,) if mode=='actor' else (n,))
                assert all(np.isfinite(trajectory[k]).all() for k in trajectory.files)
                assert np.abs(trajectory['actions']).max()<=1
        for task,stats in rollout['tasks'].items():
            cases=[v for name,v in rollout['cases'].items() if name.startswith(task+'_')]
            assert stats['episodes']==len(cases)==(1 if smoke else 5)
            assert stats['successes']==sum(c['success'] for c in cases)
    with np.load(Path(baseline_field)/'field.npz') as baseline:
        baseline_nodes=baseline['nodes'].copy();baseline_walls=baseline['walls'].copy()
    components=None;degenerate={}
    for mode in MODES:
        directory=output/f'{mode}_field';field=summary['fields'][mode]
        assert field==json.loads((directory/'summary.json').read_text())
        assert field['checkpoint_sha256']==protocol['checkpoint_sha256'] and field['epoch']==protocol['epoch']
        assert field['aggregation']==mode and field['spacing']==.2 and field['radius']==.7
        assert field['reference_replay_success'] and field['reference_replay_max_tracking_error']<1e-7
        assert field['descent_tolerance']==1e-6*UNIT and field['distance_iqr']>=0
        degenerate[mode]=field['distance_iqr']==0
        with np.load(directory/'field.npz') as data:
            assert np.array_equal(data['nodes'],baseline_nodes) and np.array_equal(data['walls'],baseline_walls)
            assert np.isfinite(data['values']).all() and (data['values']>=0).all()
            if components is None:components=data['component_values'].copy()
            else:assert np.array_equal(components,data['component_values'])
            assert components.shape==(2,len(baseline_nodes))
            expected=components.max(axis=0) if mode=='max' else components[int(mode[-1])]
            assert np.array_equal(data['values'],expected),mode
    result=dict(passed=True,epoch=protocol['epoch'],cases_per_controller=len(names),
                checkpoint_sha256=protocol['checkpoint_sha256'],same_bank_and_action_grid=True,
                same_collision_graph=True,component_and_max_fields_equal=True,all_trajectories_verified=True,
                degenerate_iqr=degenerate,scope='I/O smoke only' if smoke else 'Fixed native MQE endpoint diagnostics')
    target=output/'verification.json'
    with target.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('--bank',required=True)
    p.add_argument('--baseline-field',required=True);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();run(a.output,a.bank,a.baseline_field,a.smoke)
