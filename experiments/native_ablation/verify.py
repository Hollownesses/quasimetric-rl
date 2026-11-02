"""Check complete endpoints, paired episode identities, and retained full traces."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
REPO=Path(__file__).resolve().parents[2];sys.path[:0]=[str(REPO),str(REPO/'experiments/native_margin')]
from shared import sha,write
from experiments.native_ablation.summarize import success

def read(p):return json.loads(p.read_text())

def run(raw,out):
    protocol=read(REPO/'results/unified_mechanism_20261102/audit.json')['verified_sha256']
    checked={}
    for rel in ['results/task_source_balance_20261025/main/R0/final.pt',
                'results/ogbench_official_qrl_20261026_seed0/main/OGBench/navigate_100k/sd000_20261026_115628/params_100000.pkl',
                'results/connected_expert_utrap_20261021/main/verified_replay.pt',
                'results/connected_reference_fix_20261024/main/graph_reference_audit_only.npz',
                'experiments/ogbench_official_qrl/data/pointmaze-medium-navigate-v0.npz',
                'experiments/ogbench_official_qrl/data/pointmaze-medium-navigate-v0-val.npz']:
        checked[rel]=sha(raw/rel);assert checked[rel]==protocol[rel]
    bank=np.load(raw/'results/ogbench_critic_20261029/official_counterfactual/official_task3_bank.npz')
    states={r['id']:r for r in read(raw/'results/r_series_mppi_20261026/main/states.json')['records']}
    trace_hashes={};models={}
    for scene in ['u','og']:
        training=read(out/scene/'training.json');assert training['steps']==512 and training['native_matches_original']
        first=read(REPO/f'results/native_margin_20261102_verified/{scene}/audit.json')['stream_sha256']
        assert training['first32_sha256']==first
        source='train_u.py' if scene=='u' else 'train_og.py'
        assert training['sources'][source]==sha(Path(__file__).parent/source),'training source changed'
        with (out/scene/'episodes.csv').open() as f:episodes=list(csv.DictReader(f))
        for arm in ['native','no_push','no_local']:
            file=out/scene/arm/('final.pt' if scene=='u' else 'params_100512.pkl')
            models[str(file.relative_to(out))]=sha(file)
            assert models[str(file.relative_to(out))]==training['checkpoints'][arm]['sha256']
            ee=[r for r in episodes if r['arm']==arm];assert len(ee)==(64 if scene=='u' else 50)
            for r in ee:
                key=f"{r['state_id']}_{r['planner_seed']}" if scene=='u' else r['case']
                folder='mppi' if scene=='u' else 'greedy';p=out/scene/arm/folder/(key+'.npz')
                tr=np.load(p);n=len(tr['actions']);assert n==int(r['steps']) and n<= (300 if scene=='u' else 1000)
                if scene=='u':
                    assert len(tr['states'])==n+1 and len(tr['rewards'])==n
                    np.testing.assert_array_equal(tr['states'][0],np.asarray(states[r['state_id']]['pose'],np.float32))
                    np.testing.assert_allclose(-tr['rewards'].sum(),float(r['cost']),atol=1e-9,rtol=0)
                else:
                    assert len(tr['observations'])==n+1 and 'final_integration' in tr
                    np.testing.assert_array_equal(tr['observations'][0],bank[key+'__observation'])
                    np.testing.assert_array_equal(tr['goal'],bank[key+'__goal'])
                    assert (np.linalg.norm(tr['observations'][-1]-tr['goal'])<=1)==bool(success(r['success']))
                trace_hashes[str(p.relative_to(out))]=sha(p)
    assert len(trace_hashes)==342
    write(out/'audit.json',dict(inputs=checked,models=models,traces=trace_hashes,trace_count=len(trace_hashes),
           all_endpoints_complete=True,training_sources_unchanged=True,full_traces_match_episode_tables=True,
           sources={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}))
    print('VERIFIED',len(models),'models',len(trace_hashes),'new full trajectories')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('out',type=Path);a=p.parse_args();run(a.raw.resolve(),a.out)
