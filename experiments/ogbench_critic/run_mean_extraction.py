"""E6 mean endpoint, one new arm through the unmodified E17 extraction helpers."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time
import traceback
from common import ROOT,UPSTREAM,BASE,BASE_SHA,load_base,load_data,distances,tree_digest,tree_equal,sha256
from extraction import update,init_optimizer,proof,statistics,module_key
from utils.flax_utils import save_agent,restore_agent
import jax
import numpy as np


def read(path):return json.loads(Path(path).read_text())
def numbers(info):return {k:float(v) for k,v in info.items()}


def run(root,output):
    started=time.monotonic();deadline=started+600;root=Path(root).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    phase='setup';actual=dict(dynamics=0,actor=0);env=None
    snapshot=output/'source_snapshot';snapshot.mkdir()
    for name in ['MEAN_EXTRACTION_PROTOCOL.md','run_mean_extraction.py','test_mean_extraction.py','extraction.py','test_extraction.py','actor_recording.py','evaluate_extraction.py','common.py']:
        shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    aliases={'official_qrl.py':UPSTREAM/'impls/agents/qrl.py','official_evaluation.py':UPSTREAM/'impls/utils/evaluation.py',
             'core.py':ROOT/'experiments/ogbench_diagnostics/core.py','fixed_evaluate.py':Path(__file__).with_name('evaluate.py')}
    for name,path in aliases.items():shutil.copy2(path,snapshot/name)
    def budget():
        if time.monotonic()>deadline:raise TimeoutError('600-second fixed mean extraction budget exhausted; no extension')
    try:
        prior_path=root/'tail_1000/analysis.json';prior=read(prior_path);e17_path=root/'extraction_1000/analysis.json';e17=read(e17_path)
        source=root/'tail_1000/mean/params_101000.pkl';verification=prior['training']['verification']['mean']
        assert prior['complete'] and e17['complete'] and prior['training']['steps']==1000 and verification['passed']
        assert sha256(source)==verification['checkpoint_sha256']=='d92cf664f10345f96a77045f988a689f0985ac2120781ec596ef6b932fdaacec'
        for name,expected in e17['protocol']['source_sha256'].items():
            if name in ['extraction.py','evaluate_extraction.py','actor_recording.py','common.py','official_qrl.py','official_evaluation.py','core.py','fixed_evaluate.py']:
                assert sha256(snapshot/name)==expected,name
        base,config=load_base();initial=restore_agent(base,str(source.parent),101000)
        assert json.loads(json.dumps(config))==e17['protocol']['config']==prior['protocol']['original_config']
        assert int(initial.network.opt_state[0].count)==101000 and int(initial.network.step)==101001
        assert proof(initial,['modules_actor','modules_dynamics'])==proof(base,['modules_actor','modules_dynamics']) and tree_equal(initial.rng,base.rng)
        assert tree_digest(initial.network.params['modules_value'])==verification['critic_digest']
        fixed=proof(initial,['modules_value','modules_lam']);initial_digest=tree_digest(initial)
        env,train,val=load_data(config);env.close();env=None
        np.random.seed(9001);validation=val.sample(2048);assert tree_digest(validation)==e17['protocol']['validation_sha256']==prior['protocol']['validation_sha256']
        initial_values=np.asarray(distances(initial,validation['observations'],validation['value_goals']))
        protocol=dict(stage_steps=1000,batch_size=256,sampling_seed=2001,time_limit_seconds=600,config=json.loads(json.dumps(config)),
            source_checkpoint=str(source),source_sha256=verification['checkpoint_sha256'],prior_analysis_sha256=sha256(prior_path),e17_analysis_sha256=sha256(e17_path),
            initial_count=101000,initial_frozen=fixed,validation_sha256=tree_digest(validation),expected_stage_streams=e17['training']['stage_streams'],
            official_episodes_per_task=50,fixed_bank_sha256=sha256(root/'evaluation_bank.npz'),evaluation_rng=0,
            lambda_changed_in_prior_training=True,base_lambda=proof(base,['modules_lam']),
            source_files_sha256={p.name:sha256(p) for p in snapshot.iterdir()},devices=[str(d) for d in jax.devices()])
        (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');print('PROTOCOL_FROZEN',flush=True)
        phase='smoke';np.random.seed(2001);agent=initial
        for stage in ['dynamics','actor']:
            opt=init_optimizer(agent,stage);frozen=proof(agent,[n for n in agent.network.params if n!=module_key(stage)])
            for _ in range(2):
                budget();agent,opt,info=update(agent,opt,train.sample(256),stage);jax.block_until_ready(info)
                assert all(np.isfinite(float(v)) for v in info.values())
            assert proof(agent,list(frozen))==frozen
        smoke=output/'smoke';smoke.mkdir();save_agent(agent,str(smoke),101004)
        assert int(agent.network.opt_state[0].count)==101004 and tree_equal(agent,restore_agent(base,str(smoke),101004))
        assert proof(agent,['modules_value','modules_lam'])==fixed
        (smoke/'verification.json').write_text(json.dumps(dict(passed=True,dynamics_updates=2,actor_updates=2,full_state_roundtrip=True,frozen_stages=True),indent=2)+'\n')
        print('SMOKE_PASSED_RESET_TO_E6_MEAN',flush=True)
        agent=initial;np.random.seed(2001);stage_streams={};stage_times={};checkpoints={};phase='training'
        for stage_number,stage in enumerate(['dynamics','actor']):
            stage_start=time.monotonic();before=agent;opt=init_optimizer(agent,stage)
            frozen=proof(agent,[n for n in agent.network.params if n!=module_key(stage)])
            stream=hashlib.sha256();full_stream=hashlib.sha256()
            with (output/f'{stage}_metrics.csv').open('w') as handle:
                writer=None
                for step in range(1,1001):
                    budget();batch=train.sample(256)
                    for name in ['observations','next_observations','value_goals']:stream.update(batch[name].tobytes())
                    for name in ['observations','next_observations','value_goals','actions','actor_goals']:full_stream.update(batch[name].tobytes())
                    agent,opt,info=update(agent,opt,batch,stage);jax.block_until_ready(info);actual[stage]=step
                    if step==1 or step%100==0:
                        row=dict(stage=stage,step=step,arm='mean',elapsed_seconds=time.monotonic()-started)
                        row.update({f'train/{k}':v for k,v in numbers(info).items()});row.update({f'heldout/{k}':v for k,v in numbers(statistics(agent,validation)).items()})
                        assert all(np.isfinite(v) for v in row.values() if isinstance(v,(int,float)))
                        assert proof(agent,list(frozen))==frozen
                        if writer is None:writer=csv.DictWriter(handle,fieldnames=list(row));writer.writeheader()
                        writer.writerow(row);handle.flush();print(json.dumps(row),flush=True)
            stage_streams[stage]=dict(original_three_fields_sha256=stream.hexdigest(),full_actor_fields_sha256=full_stream.hexdigest())
            assert stage_streams[stage]==protocol['expected_stage_streams'][stage],stage
            budget();epoch=101000+1000*(stage_number+1);directory=output/'mean'/stage;directory.mkdir(parents=True)
            assert int(agent.network.opt_state[0].count)==epoch and int(agent.network.step)==epoch+1
            assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(agent))
            assert proof(agent,list(frozen))==frozen and proof(agent,['modules_value','modules_lam'])==fixed
            assert not tree_equal(before.network.params[module_key(stage)],agent.network.params[module_key(stage)])
            np.testing.assert_array_equal(np.asarray(distances(agent,validation['observations'],validation['value_goals'])),initial_values)
            save_agent(agent,str(directory),epoch);assert tree_equal(agent,restore_agent(base,str(directory),epoch))
            checkpoints[stage]=dict(epoch=epoch,sha256=sha256(directory/f'params_{epoch}.pkl'),frozen=True,full_state_roundtrip=True)
            with (output/f'{stage}_sampler_rng.pkl').open('wb') as handle:pickle.dump(np.random.get_state(),handle)
            stage_times[stage]=time.monotonic()-stage_start
            (output/'training_progress.json').write_text(json.dumps(dict(actual=actual,checkpoints=checkpoints,stage_streams=stage_streams,stage_seconds=stage_times),indent=2)+'\n')
        training=dict(passed=True,actual=actual,checkpoints=checkpoints,stage_streams=stage_streams,stage_seconds=stage_times,frozen_critic_and_lam=True,full_state_roundtrip=True,train_verify_seconds=time.monotonic()-started)
        (output/'training_complete.json').write_text(json.dumps(training,indent=2)+'\n');phase='evaluation';budget();remaining=deadline-time.monotonic()
        argv=[sys.executable,str(Path(__file__).with_name('evaluate_extraction.py')),str(root),'--checkpoint',str(output/'mean/actor'),
              '--epoch','103000','--critic-digest',fixed['modules_value']['params'],'--output',str(output/'mean_evaluation'),'--limit-seconds',str(remaining)]
        (output/'evaluation_argv.json').write_text(json.dumps(argv,indent=2)+'\n')
        with (output/'evaluation.log').open('w') as log:subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=remaining,cwd=ROOT)
        evaluation=read(output/'mean_evaluation/analysis.json');assert evaluation['complete'] and evaluation['only_neural_actor'] and not evaluation['teacher_modules_imported']
        assert len(evaluation['official_episodes'])==250 and len(evaluation['fixed_cases'])==25
        for control in e17['evaluations'].values():
            assert list(evaluation['official_episodes'])==list(control['official_episodes'])
            for name,info in evaluation['official_episodes'].items():
                np.testing.assert_array_equal(info['initial_xy'],control['official_episodes'][name]['initial_xy']);np.testing.assert_array_equal(info['goal'],control['official_episodes'][name]['goal'])
        budget();assert sha256(source)==protocol['source_sha256'] and sha256(prior_path)==protocol['prior_analysis_sha256'] and sha256(e17_path)==protocol['e17_analysis_sha256']
        assert sha256(BASE/'params_100000.pkl')==BASE_SHA and tree_digest(initial)==initial_digest
        a=dict(complete=True,protocol=protocol,training=training,evaluation=evaluation,paired_250_initials_and_goals_with_e17=True,e17_rerun=False,elapsed_seconds=time.monotonic()-started)
        (output/'analysis.json').write_text(json.dumps(a,indent=2)+'\n');print('COMPLETE',a['elapsed_seconds'],flush=True)
    except BaseException as error:
        (output/'failure.json').write_text(json.dumps(dict(complete=False,error=repr(error),phase=phase,actual=actual,elapsed_seconds=time.monotonic()-started,traceback=traceback.format_exc()),indent=2)+'\n');raise
    finally:
        if env is not None:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--output',required=True);a=p.parse_args();run(a.root,a.output)
