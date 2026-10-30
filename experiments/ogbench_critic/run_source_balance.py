"""One new weighted arm, reusing the already completed E6 control exactly."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

from common import ROOT, BASE, BASE_SHA, load_base, load_data, tree_digest, tree_equal, sha256, distances
from critic import init_optimizer
from source_balance import SourceCells, update
from tail_critic import statistics
from train import freeze_proof
from exact_score_cache import ExactScoreCache
sys.path.insert(0, str(ROOT))
from experiments.ogbench_diagnostics import core
from experiments.ogbench_diagnostics.fast_pointmaze import FastPointmaze
import jax
import numpy as np
from utils.flax_utils import save_agent, restore_agent


def read(path): return json.loads(Path(path).read_text())
def numbers(info): return {k:float(v) for k,v in info.items()}


def run(root, output):
    started=time.monotonic(); deadline=started+300; phase='setup'; steps_done=0
    root=Path(root).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    snapshot=output/'source_snapshot';snapshot.mkdir();env=engine=None
    files=[Path(__file__).with_name(n) for n in ['SOURCE_BALANCE_PROTOCOL.md','source_balance.py','test_source_balance.py',
        'run_source_balance.py','common.py','critic.py','train.py','tail_critic.py','exact_score_cache.py']]
    files += [ROOT/'experiments/ogbench_diagnostics'/n for n in ['fast_pointmaze.py','core.py']]
    for path in files:shutil.copy2(path,snapshot/path.name)
    def budget():
        if time.monotonic()>deadline:raise TimeoutError('300-second source-balance budget exhausted; no automatic extension')
    try:
        control=read(root/'tail_1000/analysis.json');assert control['complete'] and control['training']['steps']==1000
        cp=control['protocol'];proof=control['training']['verification']['mean']
        assert cp['batch_size']==256 and cp['sampling_seed']==2001 and proof['passed'] and proof['adam_count']==101000
        assert sha256(root/'tail_1000/mean/params_101000.pkl')==proof['checkpoint_sha256']
        fast=read(root/'cached_pointmaze_verification/verification.json')
        assert fast['passed'] and len(fast['cases'])==25 and len(fast['candidates'])==35
        for name,digest in fast['source_sha256'].items():
            directory=Path(__file__).parent if name in ['exact_score_cache.py','test_exact_score_cache.py'] else ROOT/'experiments/ogbench_diagnostics'
            assert sha256(directory/name)==digest
        base,config=load_base();assert json.loads(json.dumps(config))==cp['original_config']
        env,train,val=load_data(config);frozen=freeze_proof(base);original=tree_digest(base)
        source=train.dataset['observations'][train.dataset.valid_idxs]
        cells=SourceCells.build(source);cells.save(output/'source_cells.npz');population=cells.weights(source)
        assert np.isclose(population.mean(dtype=np.float64),1,atol=1e-7)
        stride=np.arange(0,len(source),max(1,len(source)//32))
        np.testing.assert_array_equal(cells.weights(source[stride]),SourceCells.load(output/'source_cells.npz').weights(source[stride]))
        np.random.seed(9001);validation=val.sample(2048);assert tree_digest(validation)==cp['validation_sha256']
        protocol=dict(steps=1000,batch_size=256,sampling_seed=2001,config=json.loads(json.dumps(config)),
            base_sha256=BASE_SHA,time_limit_seconds=300,initial_frozen=frozen,
            source_cells=dict(total=cells.total,occupied=cells.occupied,spacing=.2,minimum_count=int(cells.counts[cells.counts>0].min()),
                maximum_count=int(cells.counts.max()),population_weight_mean=float(population.mean(dtype=np.float64)),
                minimum_weight=float(population.min()),maximum_weight=float(population.max()),
                population_effective_samples=float(population.sum(dtype=np.float64)**2/np.square(population.astype(np.float64)).sum())),
            cells_sha256=sha256(output/'source_cells.npz'),validation_sha256=tree_digest(validation),
            control_analysis_sha256=sha256(root/'tail_1000/analysis.json'),control_checkpoint_sha256=proof['checkpoint_sha256'],
            expected_stream_sha256=control['training']['paired_stream_sha256'],
            fast_verification_sha256=sha256(root/'cached_pointmaze_verification/verification.json'),
            source_sha256={p.name:sha256(p) for p in snapshot.iterdir()},devices=[str(d) for d in jax.devices()])
        (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
        (output/'initial_statistics.json').write_text(json.dumps(numbers(statistics(base,validation)),indent=2)+'\n')
        print('PROTOCOL',protocol,flush=True)
        phase='smoke';np.random.seed(2001);agent=base;opt=init_optimizer(base)
        for _ in range(2):
            budget();batch=train.sample(256);agent,opt,info=update(agent,opt,batch,cells.weights(batch['observations']))
            jax.block_until_ready(info['total_loss']);assert np.isfinite(float(info['total_loss']))
        smoke=output/'smoke';smoke.mkdir();save_agent(agent,str(smoke),100002)
        restored=restore_agent(base,str(smoke),100002)
        assert tree_equal(agent,restored) and freeze_proof(agent)==frozen and int(agent.network.opt_state[0].count)==100002
        (smoke/'verification.json').write_text(json.dumps(dict(passed=True,full_state_roundtrip=True,adam_count=100002,frozen=True),indent=2)+'\n')
        phase='training';agent=base;opt=init_optimizer(base);np.random.seed(2001);stream=hashlib.sha256()
        with (output/'metrics.csv').open('w') as handle:
            writer=None
            for step in range(1,1001):
                budget();batch=train.sample(256)
                for name in ['observations','next_observations','value_goals']:stream.update(batch[name].tobytes())
                agent,opt,info=update(agent,opt,batch,cells.weights(batch['observations']))
                jax.block_until_ready(info['total_loss']);steps_done=step
                if step==1 or step%100==0:
                    row=dict(step=step,elapsed_seconds=time.monotonic()-started)
                    row.update({f'train/{k}':v for k,v in numbers(info).items()})
                    row.update({f'heldout/{k}':v for k,v in numbers(statistics(agent,validation)).items()})
                    assert all(np.isfinite(v) for v in row.values())
                    if writer is None:writer=csv.DictWriter(handle,fieldnames=list(row));writer.writeheader()
                    writer.writerow(row);handle.flush();print(json.dumps(row),flush=True)
        assert stream.hexdigest()==protocol['expected_stream_sha256'],'E6 training stream differs; refuse comparison'
        phase='checkpoint';budget();assert freeze_proof(agent)==frozen
        assert int(agent.network.opt_state[0].count)==101000 and int(agent.network.step)==101001
        assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(agent))
        model=output/'model';model.mkdir();save_agent(agent,str(model),101000)
        assert tree_equal(agent,restore_agent(base,str(model),101000))
        cases=core.read_bank(root/'evaluation_bank.npz');assert len(cases)==25
        obs=np.array([c['observation'] for c in cases.values()]);goals=np.array([c['goal'] for c in cases.values()])
        key=jax.random.PRNGKey(123)
        assert tree_equal(base.sample_actions(obs,goals,seed=key,temperature=0),agent.sample_actions(obs,goals,seed=key,temperature=0))
        training=dict(passed=True,steps=1000,adam_count=101000,paired_stream_sha256=stream.hexdigest(),
            full_state_roundtrip=True,frozen_parameters_and_moments=True,frozen_bank_actor_outputs=True,
            checkpoint_sha256=sha256(model/'params_101000.pkl'),train_verify_seconds=time.monotonic()-started)
        (output/'training_complete.json').write_text(json.dumps(training,indent=2)+'\n')
        phase='evaluation';engine=FastPointmaze(env,nthread=4);score_cache=ExactScoreCache(agent,audit=True)
        greedy=output/'greedy';greedy.mkdir();results={}
        for name,case in cases.items():
            budget()
            def score(observations,goals):budget();return score_cache(observations,goals)
            info,trace=engine.rollout(case,score);trace['final_integration']=core.save_integration(env)
            np.savez_compressed(greedy/f'{name}.npz',**trace);results[name]=info
            (greedy/'progress.json').write_text(json.dumps(results,indent=2)+'\n');print(name,info,flush=True)
        (greedy/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
        phase='field';budget();walls,radius=core.geometry(env);nodes,graph=core.build_graph(walls,radius,.2)
        with np.load(root/'baseline_field/field.npz') as old:np.testing.assert_array_equal(nodes,old['nodes'])
        values=np.asarray(distances(agent,nodes,np.broadcast_to(core.GOAL,nodes.shape)))
        field_stats,field=core.graph_metrics(nodes,graph,values)
        np.savez_compressed(output/'field.npz',**{**field,'walls':walls})
        budget();assert tree_digest(base)==original and sha256(BASE/'params_100000.pkl')==BASE_SHA
        result=dict(complete=True,protocol=protocol,training=training,cases=results,field=field_stats,
            cache=score_cache.statistics(),bank_sha256=sha256(root/'evaluation_bank.npz'),elapsed_seconds=time.monotonic()-started)
        (output/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print('COMPLETE',result['elapsed_seconds'],flush=True)
    except BaseException as error:
        (output/'failure.json').write_text(json.dumps(dict(error=repr(error),phase=phase,actual_updates=steps_done,
            elapsed_seconds=time.monotonic()-started,traceback=traceback.format_exc(),complete=False),indent=2)+'\n');raise
    finally:
        if engine is not None:engine.close()
        if env is not None:env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.root,args.output)
