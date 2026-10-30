"""Fixed 1000-step original-IQE path constraints with exact E6 control reuse."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

from common import ROOT,BASE,BASE_SHA,load_base,load_data,tree_digest,tree_equal,sha256,distances
from critic import init_optimizer
from path_bound import PathBounds,sample_with_indices,requested_horizons,update,statistics as path_statistics
from tail_critic import statistics
from train import freeze_proof
from exact_score_cache import ExactScoreCache
sys.path.insert(0,str(ROOT))
from experiments.ogbench_diagnostics import core
from experiments.ogbench_diagnostics.fast_pointmaze import FastPointmaze
import jax
import numpy as np
from utils.flax_utils import save_agent,restore_agent


def read(path):return json.loads(Path(path).read_text())
def numbers(info):return {k:float(v) for k,v in info.items()}


def run(root,output):
    started=time.monotonic();deadline=started+300;phase='setup';steps_done=0
    root=Path(root).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    snapshot=output/'source_snapshot';snapshot.mkdir();env=engine=None
    files=[Path(__file__).with_name(n) for n in ['PATH_BOUND_PROTOCOL.md','path_bound.py','test_path_bound.py',
        'run_path_bound.py','common.py','critic.py','train.py','tail_critic.py','exact_score_cache.py']]
    files += [ROOT/'experiments/ogbench_diagnostics'/n for n in ['fast_pointmaze.py','core.py']]
    for path in files:shutil.copy2(path,snapshot/path.name)
    def budget():
        if time.monotonic()>deadline:raise TimeoutError('300-second path-bound budget exhausted; no automatic extension')
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
        bounds=PathBounds(train);val_bounds=PathBounds(val)
        np.random.seed(9001);validation,val_ids=sample_with_indices(val,2048)
        assert tree_digest(validation)==cp['validation_sha256']
        val_goals,val_costs,val_ends=val_bounds.endpoints(val_ids,requested_horizons(0,2048))
        np.savez_compressed(output/'validation_paths.npz',indices=val_ids,end_indices=val_ends,goals=val_goals,costs=val_costs)
        control_agent=restore_agent(base,str(root/'tail_1000/mean'),101000)
        (output/'initial_statistics.json').write_text(json.dumps(numbers(statistics(base,validation)),indent=2)+'\n')
        (output/'initial_path_statistics.json').write_text(json.dumps(numbers(path_statistics(base,validation,val_goals,val_costs)),indent=2)+'\n')
        (output/'control_path_statistics.json').write_text(json.dumps(numbers(path_statistics(control_agent,validation,val_goals,val_costs)),indent=2)+'\n')
        protocol=dict(steps=1000,batch_size=256,sampling_seed=2001,horizons=[2,4,8,16,32,64],
            aggregation='mean(max(local_squared_positive,path_squared_positive))',config=json.loads(json.dumps(config)),
            base_sha256=BASE_SHA,time_limit_seconds=300,initial_frozen=frozen,validation_sha256=tree_digest(validation),
            validation_paths_sha256=sha256(output/'validation_paths.npz'),
            control_analysis_sha256=sha256(root/'tail_1000/analysis.json'),control_checkpoint_sha256=proof['checkpoint_sha256'],
            expected_stream_sha256=control['training']['paired_stream_sha256'],
            fast_verification_sha256=sha256(root/'cached_pointmaze_verification/verification.json'),
            source_sha256={p.name:sha256(p) for p in snapshot.iterdir()},devices=[str(d) for d in jax.devices()])
        (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');print('PROTOCOL',protocol,flush=True)
        phase='smoke';np.random.seed(2001);agent=base;opt=init_optimizer(base)
        for step in [1,2]:
            budget();batch,ids=sample_with_indices(train,256);goals,costs,_=bounds.endpoints(ids,requested_horizons(step,256))
            agent,opt,info=update(agent,opt,batch,goals,costs);jax.block_until_ready(info['total_loss'])
            assert all(np.isfinite(float(v)) for v in info.values())
        smoke=output/'smoke';smoke.mkdir();save_agent(agent,str(smoke),100002)
        assert tree_equal(agent,restore_agent(base,str(smoke),100002)) and freeze_proof(agent)==frozen
        assert int(agent.network.opt_state[0].count)==100002
        (smoke/'verification.json').write_text(json.dumps(dict(passed=True,full_state_roundtrip=True,adam_count=100002,frozen=True),indent=2)+'\n')
        phase='training';agent=base;opt=init_optimizer(base);np.random.seed(2001)
        stream=hashlib.sha256();path_stream=hashlib.sha256();cost_counts=np.zeros(65,np.int64)
        with (output/'metrics.csv').open('w') as handle:
            writer=None
            for step in range(1,1001):
                budget();batch,ids=sample_with_indices(train,256)
                for name in ['observations','next_observations','value_goals']:stream.update(batch[name].tobytes())
                goals,costs,ends=bounds.endpoints(ids,requested_horizons(step,256))
                path_stream.update(ids.tobytes());path_stream.update(ends.tobytes());path_stream.update(costs.tobytes())
                cost_counts += np.bincount(costs.astype(np.int64),minlength=65)
                agent,opt,info=update(agent,opt,batch,goals,costs);jax.block_until_ready(info['total_loss']);steps_done=step
                if step==1 or step%100==0:
                    row=dict(step=step,elapsed_seconds=time.monotonic()-started)
                    row.update({f'train/{k}':v for k,v in numbers(info).items()})
                    row.update({f'heldout/{k}':v for k,v in numbers(statistics(agent,validation)).items()})
                    row.update({f'path_heldout/{k}':v for k,v in numbers(path_statistics(agent,validation,val_goals,val_costs)).items()})
                    assert all(np.isfinite(v) for v in row.values())
                    if writer is None:writer=csv.DictWriter(handle,fieldnames=list(row));writer.writeheader()
                    writer.writerow(row);handle.flush();print(json.dumps(row),flush=True)
        assert stream.hexdigest()==protocol['expected_stream_sha256'],'E6 batches differ; refuse comparison'
        assert cost_counts.sum()==256000
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
            path_stream_sha256=path_stream.hexdigest(),path_cost_counts=cost_counts.tolist(),
            full_state_roundtrip=True,frozen_parameters_and_moments=True,frozen_bank_actor_outputs=True,
            checkpoint_sha256=sha256(model/'params_101000.pkl'),train_verify_seconds=time.monotonic()-started)
        (output/'training_complete.json').write_text(json.dumps(training,indent=2)+'\n')
        phase='evaluation';engine=FastPointmaze(env,nthread=4);cache=ExactScoreCache(agent,audit=True)
        greedy=output/'greedy';greedy.mkdir();results={}
        for name,case in cases.items():
            budget()
            def score(observations,goals):budget();return cache(observations,goals)
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
            cache=cache.statistics(),bank_sha256=sha256(root/'evaluation_bank.npz'),elapsed_seconds=time.monotonic()-started)
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
