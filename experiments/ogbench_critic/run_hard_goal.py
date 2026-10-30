"""Fixed finite max-goal-pool fitting with unchanged original IQE update."""
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
from iqe_distill import init_optimizer,update,statistics,frozen_proof
from tail_critic import statistics as official_statistics
from roadmap_metric import RoadmapMetric
from query_augmentation import QueryAugmentation
from hard_goal import HardGoal,pool_statistics
import jax
import numpy as np
from utils.flax_utils import save_agent,restore_agent


def read(path):return json.loads(Path(path).read_text())
def numbers(info):return {k:float(v) for k,v in info.items()}


def run(root,output):
    started=time.monotonic();deadline=started+360;phase='setup';steps_done=0;env=None
    root=Path(root).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    snapshot=output/'source_snapshot';snapshot.mkdir()
    files=[Path(__file__).with_name(n) for n in ['HARD_GOAL_PROTOCOL.md','iqe_distill.py','query_augmentation.py','goal_support.py','hard_goal.py','test_hard_goal.py',
        'run_hard_goal.py','evaluate_neural.py','roadmap_metric.py','common.py','tail_critic.py','exact_score_cache.py']]
    files += [ROOT/'experiments/ogbench_diagnostics'/n for n in ['fast_pointmaze.py','core.py']]
    for path in files:shutil.copy2(path,snapshot/path.name)
    def budget():
        if time.monotonic()>deadline:raise TimeoutError('360-second hard goal budget exhausted; no extension')
    try:
        previous=read(root/'goal_support_1000/analysis.json');assert previous['complete']
        assert sha256(root/'goal_support_1000/model/params_101000.pkl')==previous['training']['checkpoint_sha256']
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
        labels=teacher(validation['observations'],validation['value_goals']).astype(np.float32)
        np.save(output/'validation_targets.npy',labels)
        sampler=HardGoal(teacher.nodes,candidates=8);old_sampler=QueryAugmentation();aug_validation=sampler(validation,128000)
        old_aug_validation=old_sampler(validation,128000)
        old_aug_labels=teacher(old_aug_validation['observations'],old_aug_validation['value_goals']).astype(np.float32)
        aug_labels=teacher(aug_validation['observations'],aug_validation['value_goals']).astype(np.float32)
        np.save(output/'augmented_validation_targets.npy',aug_labels)
        _,pool_sources,pool_goals,_=sampler.proposals(validation,128000)
        pool_labels=teacher(pool_sources,pool_goals.reshape(-1,2)).astype(np.float32).reshape(pool_goals.shape[:2])
        np.savez_compressed(output/'validation_pool.npz',sources=pool_sources,goals=pool_goals,labels=pool_labels)
        control_agent=restore_agent(base,str(root/'tail_1000/mean'),101000)
        (output/'initial_statistics.json').write_text(json.dumps(numbers(statistics(base,validation,labels)),indent=2)+'\n')
        (output/'initial_augmented_statistics.json').write_text(json.dumps(numbers(statistics(base,aug_validation,aug_labels)),indent=2)+'\n')
        previous_agent=restore_agent(base,str(root/'goal_support_1000/model'),101000)
        (output/'previous_statistics.json').write_text(json.dumps(dict(original=numbers(statistics(previous_agent,validation,labels)),augmented=numbers(statistics(previous_agent,aug_validation,aug_labels)),old_augmented=numbers(statistics(previous_agent,old_aug_validation,old_aug_labels))),indent=2)+'\n')
        e12=read(root/'query_augmentation_1000/analysis.json');assert e12['complete']
        assert sha256(root/'query_augmentation_1000/model/params_101000.pkl')==e12['training']['checkpoint_sha256']
        e12_agent=restore_agent(base,str(root/'query_augmentation_1000/model'),101000)
        pool_controls=dict(original=pool_statistics(base,pool_sources,pool_goals,pool_labels),e12=pool_statistics(e12_agent,pool_sources,pool_goals,pool_labels),e15=pool_statistics(previous_agent,pool_sources,pool_goals,pool_labels))
        (output/'pool_controls.json').write_text(json.dumps(pool_controls,indent=2)+'\n')
        (output/'control_statistics.json').write_text(json.dumps(numbers(statistics(control_agent,validation,labels)),indent=2)+'\n')
        protocol=dict(steps=1000,batch_size=256,sampling_seed=2001,config=json.loads(json.dumps(config)),time_limit_seconds=360,
            base_sha256=BASE_SHA,initial_frozen=frozen,query_mixture='half original; half same source, maximum paired residual from fixed 8-goal uniform-support pool',
            candidates=8,validation_pool_sha256=sha256(output/'validation_pool.npz'),e12_analysis_sha256=sha256(root/'query_augmentation_1000/analysis.json'),
            support_nodes=len(teacher.nodes),stride=sampler.stride,
            previous_analysis_sha256=sha256(root/'goal_support_1000/analysis.json'),previous_checkpoint_sha256=previous['training']['checkpoint_sha256'],
            augmented_validation_sha256=tree_digest(aug_validation),augmented_validation_targets_sha256=sha256(output/'augmented_validation_targets.npy'),teacher_analysis_sha256=sha256(root/'roadmap_metric/analysis.json'),
            teacher_graph_sha256=tp['graph_sha256'],teacher_distance_sha256=tp['distance_sha256'],
            validation_sha256=tree_digest(validation),validation_targets_sha256=sha256(output/'validation_targets.npy'),
            control_analysis_sha256=sha256(root/'tail_1000/analysis.json'),control_checkpoint_sha256=proof['checkpoint_sha256'],
            expected_stream_sha256=control['training']['paired_stream_sha256'],
            source_sha256={p.name:sha256(p) for p in snapshot.iterdir()},devices=[str(d) for d in jax.devices()])
        (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');print('PROTOCOL',protocol,flush=True)
        phase='smoke';np.random.seed(2001);agent=base;opt=init_optimizer(base)
        for smoke_step in range(2):
            budget();batch,target,_,_=sampler.choose(agent,train.sample(256),128*smoke_step,teacher)
            agent,opt,info=update(agent,opt,batch,target);jax.block_until_ready(info['loss']);assert all(np.isfinite(float(v)) for v in info.values())
        smoke=output/'smoke';smoke.mkdir();save_agent(agent,str(smoke),100002)
        assert tree_equal(agent,restore_agent(base,str(smoke),100002)) and frozen_proof(agent)==frozen
        assert int(agent.network.opt_state[0].count)==100002
        (smoke/'verification.json').write_text(json.dumps(dict(passed=True,full_state_roundtrip=True,adam_count=100002,frozen=True),indent=2)+'\n')
        phase='training';agent=base;opt=init_optimizer(base);np.random.seed(2001);stream=hashlib.sha256();targets_stream=hashlib.sha256();queries_stream=hashlib.sha256();old_queries_stream=hashlib.sha256();uniform_stream=hashlib.sha256();pool_stream=hashlib.sha256();selection_stream=hashlib.sha256();selected_counts=np.zeros(len(teacher.nodes),np.int64)
        with (output/'metrics.csv').open('w') as handle:
            writer=None
            for step in range(1,1001):
                budget();batch=train.sample(256)
                for name in ['observations','next_observations','value_goals']:stream.update(batch[name].tobytes())
                old_batch=old_sampler(batch,128*(step-1))
                batch,target,selection_info,pool=sampler.choose(agent,batch,128*(step-1),teacher)
                for name in ['observations','value_goals']:uniform_stream.update(pool['uniform'][name].tobytes())
                for array in [batch['observations'][1::2],pool['goals'],pool['labels']]:pool_stream.update(array.tobytes())
                selection_stream.update(pool['indices'].tobytes())
                chosen_ids=pool['prototype_indices'][np.arange(128),pool['indices']]
                selected_counts+=np.bincount(chosen_ids,minlength=len(teacher.nodes))
                np.testing.assert_array_equal(batch['observations'],old_batch['observations'])
                for name in ['observations','value_goals']:old_queries_stream.update(old_batch[name].tobytes())
                for name in ['observations','value_goals']:queries_stream.update(batch[name].tobytes())
                targets_stream.update(target.tobytes())
                agent,opt,info=update(agent,opt,batch,target);jax.block_until_ready(info['loss']);steps_done=step
                if step==1 or step%100==0:
                    row=dict(step=step,elapsed_seconds=time.monotonic()-started)
                    row.update({f'selection/{k}':v for k,v in selection_info.items()})
                    row.update({f'pool_heldout/{k}':v for k,v in pool_statistics(agent,pool_sources,pool_goals,pool_labels).items()})
                    row.update({f'train/{k}':v for k,v in numbers(info).items()})
                    row.update({f'heldout/{k}':v for k,v in numbers(statistics(agent,validation,labels)).items()})
                    row.update({f'augmented_heldout/{k}':v for k,v in numbers(statistics(agent,aug_validation,aug_labels)).items()})
                    row.update({f'old_augmented_heldout/{k}':v for k,v in numbers(statistics(agent,old_aug_validation,old_aug_labels)).items()})
                    row.update({f'original_risk/{k}':v for k,v in numbers(official_statistics(agent,validation)).items()})
                    assert all(np.isfinite(v) for v in row.values())
                    if writer is None:writer=csv.DictWriter(handle,fieldnames=list(row));writer.writeheader()
                    writer.writerow(row);handle.flush();print(json.dumps(row),flush=True)
        assert stream.hexdigest()==protocol['expected_stream_sha256'],'Control stream differs; refuse comparison'
        assert old_queries_stream.hexdigest()==previous['training']['old_queries_stream_sha256']
        assert uniform_stream.hexdigest()==previous['training']['queries_stream_sha256']
        assert selected_counts.sum()==128000
        np.save(output/'selected_support_counts.npy',selected_counts)
        support_counts=np.bincount(sampler.indices(0,128000),minlength=len(teacher.nodes))
        assert support_counts.sum()==128000 and support_counts.min()==20 and support_counts.max()==21
        np.save(output/'support_counts.npy',support_counts)
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
            frozen_bank_actor_outputs=True,paired_stream_sha256=stream.hexdigest(),targets_stream_sha256=targets_stream.hexdigest(),queries_stream_sha256=queries_stream.hexdigest(),old_queries_stream_sha256=old_queries_stream.hexdigest(),
            uniform_queries_stream_sha256=uniform_stream.hexdigest(),pool_stream_sha256=pool_stream.hexdigest(),selection_stream_sha256=selection_stream.hexdigest(),
            selected_counts_sha256=sha256(output/'selected_support_counts.npy'),selected_unique_prototypes=int((selected_counts>0).sum()),selected_max_count=int(selected_counts.max()),
            support_counts_sha256=sha256(output/'support_counts.npy'),support_counts_min=int(support_counts.min()),support_counts_max=int(support_counts.max()),
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
