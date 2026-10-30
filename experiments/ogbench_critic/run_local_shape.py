"""Fixed signed-shape learning with unchanged E12 current-query stream."""
import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from common import ROOT,BASE,BASE_SHA,load_base,load_data,tree_digest,tree_equal,sha256
from local_shape import init_optimizer,update,statistics,frozen_proof
from tail_critic import statistics as official_statistics
from roadmap_metric import RoadmapMetric
from shape_queries import ShapeQueries
import jax
import numpy as np
from utils.flax_utils import save_agent,restore_agent


def read(path):return json.loads(Path(path).read_text())
def numbers(info):return {k:float(v) for k,v in info.items()}


def targets(teacher,batch):
    goal=batch['value_goals']
    return np.column_stack([teacher(batch['observations'],goal),teacher(batch['next_observations'],goal)]).astype(np.float32)


def run(root,output):
    started=time.monotonic();deadline=started+300;phase='setup';steps_done=0;env=None
    root=Path(root).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    snapshot=output/'source_snapshot';snapshot.mkdir()
    files=[Path(__file__).with_name(n) for n in ['LOCAL_SHAPE_PROTOCOL.md','iqe_distill.py','local_shape.py','shape_queries.py','test_local_shape.py','query_augmentation.py',
        'run_local_shape.py','evaluate_neural.py','roadmap_metric.py','common.py','tail_critic.py','exact_score_cache.py']]
    files += [ROOT/'experiments/ogbench_diagnostics'/n for n in ['fast_pointmaze.py','core.py']]
    for path in files:shutil.copy2(path,snapshot/path.name)
    def budget():
        if time.monotonic()>deadline:raise TimeoutError('300-second local shape budget exhausted; no extension')
    try:
        previous=read(root/'query_augmentation_1000/analysis.json');assert previous['complete']
        assert sha256(root/'query_augmentation_1000/model/params_101000.pkl')==previous['training']['checkpoint_sha256']
        teacher_result=read(root/'roadmap_metric/analysis.json');assert teacher_result['complete'] and teacher_result['eligible_for_neural_study']
        assert len(teacher_result['cases'])==25 and all(x['success'] for x in teacher_result['cases'].values())
        tp=teacher_result['protocol'];assert sha256(root/'roadmap_metric/graph.npz')==tp['graph_sha256']
        assert sha256(root/'roadmap_metric/distances.npy')==tp['distance_sha256']
        assert sha256(Path(__file__).with_name('roadmap_metric.py'))==tp['source_sha256']['roadmap_metric.py']
        control=read(root/'tail_1000/analysis.json');cp=control['protocol'];proof=control['training']['verification']['mean']
        assert control['complete'] and cp['steps']==1000 and cp['batch_size']==256 and cp['sampling_seed']==2001
        assert sha256(root/'tail_1000/mean/params_101000.pkl')==proof['checkpoint_sha256']
        base,config=load_base();assert json.loads(json.dumps(config))==cp['original_config'] and config['quasimetric_type']=='iqe'
        env,train,val=load_data(config);env.close();env=None;frozen=frozen_proof(base);original=tree_digest(base)
        teacher=RoadmapMetric.load(root/'roadmap_metric/graph.npz',root/'roadmap_metric/distances.npy')
        np.random.seed(9001);validation=val.sample(2048);assert tree_digest(validation)==cp['validation_sha256']
        labels=targets(teacher,validation)
        np.save(output/'validation_targets.npy',labels)
        sampler=ShapeQueries();aug_validation=sampler(validation,128000)
        aug_labels=targets(teacher,aug_validation)
        np.save(output/'augmented_validation_targets.npy',aug_labels)
        control_agent=restore_agent(base,str(root/'tail_1000/mean'),101000)
        (output/'initial_statistics.json').write_text(json.dumps(numbers(statistics(base,validation,labels)),indent=2)+'\n')
        (output/'initial_augmented_statistics.json').write_text(json.dumps(numbers(statistics(base,aug_validation,aug_labels)),indent=2)+'\n')
        previous_agent=restore_agent(base,str(root/'query_augmentation_1000/model'),101000)
        (output/'previous_statistics.json').write_text(json.dumps(dict(original=numbers(statistics(previous_agent,validation,labels)),augmented=numbers(statistics(previous_agent,aug_validation,aug_labels))),indent=2)+'\n')
        (output/'control_statistics.json').write_text(json.dumps(numbers(statistics(control_agent,validation,labels)),indent=2)+'\n')
        protocol=dict(steps=1000,batch_size=256,sampling_seed=2001,config=json.loads(json.dumps(config)),time_limit_seconds=300,
            base_sha256=BASE_SHA,initial_frozen=frozen,query_mixture='half original, half full s/g Sobol [-1,1]',
            previous_analysis_sha256=sha256(root/'query_augmentation_1000/analysis.json'),previous_checkpoint_sha256=previous['training']['checkpoint_sha256'],
            augmented_validation_sha256=tree_digest(aug_validation),augmented_validation_targets_sha256=sha256(output/'augmented_validation_targets.npy'),teacher_analysis_sha256=sha256(root/'roadmap_metric/analysis.json'),
            teacher_graph_sha256=tp['graph_sha256'],teacher_distance_sha256=tp['distance_sha256'],
            validation_sha256=tree_digest(validation),validation_targets_sha256=sha256(output/'validation_targets.npy'),
            control_analysis_sha256=sha256(root/'tail_1000/analysis.json'),control_checkpoint_sha256=proof['checkpoint_sha256'],
            expected_stream_sha256=control['training']['paired_stream_sha256'],
            source_sha256={p.name:sha256(p) for p in snapshot.iterdir()},devices=[str(d) for d in jax.devices()])
        (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');print('PROTOCOL',protocol,flush=True)
        phase='smoke';np.random.seed(2001);agent=base;opt=init_optimizer(base)
        for smoke_step in range(2):
            budget();batch=sampler(train.sample(256),128*smoke_step);target=targets(teacher,batch)
            agent,opt,info=update(agent,opt,batch,target);jax.block_until_ready(info['loss']);assert all(np.isfinite(float(v)) for v in info.values())
        smoke=output/'smoke';smoke.mkdir();save_agent(agent,str(smoke),100002)
        assert tree_equal(agent,restore_agent(base,str(smoke),100002)) and frozen_proof(agent)==frozen
        assert int(agent.network.opt_state[0].count)==100002
        (smoke/'verification.json').write_text(json.dumps(dict(passed=True,full_state_roundtrip=True,adam_count=100002,frozen=True),indent=2)+'\n')
        phase='training';agent=base;opt=init_optimizer(base);np.random.seed(2001);stream=hashlib.sha256();targets_stream=hashlib.sha256();queries_stream=hashlib.sha256();next_queries_stream=hashlib.sha256();next_targets_stream=hashlib.sha256()
        with (output/'metrics.csv').open('w') as handle:
            writer=None
            for step in range(1,1001):
                budget();batch=train.sample(256)
                for name in ['observations','next_observations','value_goals']:stream.update(batch[name].tobytes())
                batch=sampler(batch,128*(step-1))
                for name in ['observations','value_goals']:queries_stream.update(batch[name].tobytes())
                next_queries_stream.update(batch['next_observations'].tobytes())
                target=targets(teacher,batch);targets_stream.update(target[:,0].tobytes());next_targets_stream.update(target[:,1].tobytes())
                agent,opt,info=update(agent,opt,batch,target);jax.block_until_ready(info['loss']);steps_done=step
                if step==1 or step%100==0:
                    row=dict(step=step,elapsed_seconds=time.monotonic()-started)
                    row.update({f'train/{k}':v for k,v in numbers(info).items()})
                    row.update({f'heldout/{k}':v for k,v in numbers(statistics(agent,validation,labels)).items()})
                    row.update({f'augmented_heldout/{k}':v for k,v in numbers(statistics(agent,aug_validation,aug_labels)).items()})
                    row.update({f'original_risk/{k}':v for k,v in numbers(official_statistics(agent,validation)).items()})
                    assert all(np.isfinite(v) for v in row.values())
                    if writer is None:writer=csv.DictWriter(handle,fieldnames=list(row));writer.writeheader()
                    writer.writerow(row);handle.flush();print(json.dumps(row),flush=True)
        assert stream.hexdigest()==protocol['expected_stream_sha256'],'Control stream differs; refuse comparison'
        assert queries_stream.hexdigest()==previous['training']['queries_stream_sha256']
        assert targets_stream.hexdigest()==previous['training']['targets_stream_sha256']
        phase='checkpoint';budget();assert frozen_proof(agent)==frozen
        assert int(agent.network.opt_state[0].count)==101000 and int(agent.network.step)==101001
        assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(agent))
        model=output/'model';model.mkdir();save_agent(agent,str(model),101000)
        assert tree_equal(agent,restore_agent(base,str(model),101000))
        # No environment/teacher decisions are made in the independent evaluator.
        bank=np.load(root/'evaluation_bank.npz');names=sorted({k.split('__')[0] for k in bank.files})
        observations=np.array([bank[f'{n}__observation'] for n in names]);goals=np.array([bank[f'{n}__goal'] for n in names]);key=jax.random.PRNGKey(123)
        assert tree_equal(base.sample_actions(observations,goals,seed=key,temperature=0),agent.sample_actions(observations,goals,seed=key,temperature=0))
        training=dict(passed=True,steps=1000,adam_count=101000,full_state_roundtrip=True,frozen_parameters_and_moments=True,
            frozen_bank_actor_outputs=True,paired_stream_sha256=stream.hexdigest(),targets_stream_sha256=targets_stream.hexdigest(),queries_stream_sha256=queries_stream.hexdigest(),next_queries_stream_sha256=next_queries_stream.hexdigest(),next_targets_stream_sha256=next_targets_stream.hexdigest(),
            checkpoint_sha256=sha256(model/'params_101000.pkl'),train_verify_seconds=time.monotonic()-started)
        (output/'training_complete.json').write_text(json.dumps(training,indent=2)+'\n')
        del teacher;gc.collect();phase='neural_evaluation';budget();remaining=deadline-time.monotonic()
        argv=[sys.executable,str(Path(__file__).with_name('evaluate_neural.py')),str(root),'--checkpoint',str(model),
            '--epoch','101000','--output',str(output/'neural_evaluation'),'--limit-seconds',str(remaining)]
        (output/'neural_evaluation_argv.json').write_text(json.dumps(argv,indent=2)+'\n')
        with (output/'neural_evaluation.log').open('w') as log:
            subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=remaining,cwd=ROOT)
        neural=read(output/'neural_evaluation/analysis.json')
        assert neural['complete'] and neural['only_neural_scores'] and not neural['roadmap_module_imported']
        assert neural['checkpoint_sha256']==training['checkpoint_sha256']
        budget();assert tree_digest(base)==original and sha256(BASE/'params_100000.pkl')==BASE_SHA
        result=dict(complete=True,protocol=protocol,training=training,neural=neural,elapsed_seconds=time.monotonic()-started)
        (output/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print('COMPLETE',result['elapsed_seconds'],flush=True)
    except BaseException as error:
        (output/'failure.json').write_text(json.dumps(dict(error=repr(error),phase=phase,actual_updates=steps_done,
            elapsed_seconds=time.monotonic()-started,traceback=traceback.format_exc(),complete=False),indent=2)+'\n');raise
    finally:
        if env is not None:env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.root,args.output)
