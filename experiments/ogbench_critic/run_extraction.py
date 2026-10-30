"""Matched frozen-critic official dynamics/actor adaptation, then full evaluation."""
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
    phase='setup';actual=dict(original=dict(dynamics=0,actor=0),improved=dict(dynamics=0,actor=0));env=None
    snapshot=output/'source_snapshot';snapshot.mkdir()
    for name in ['EXTRACTION_PROTOCOL.md','extraction.py','test_extraction.py','actor_recording.py','evaluate_extraction.py','run_extraction.py','common.py']:
        shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    for source,name in [(UPSTREAM/'impls/agents/qrl.py','official_qrl.py'),(UPSTREAM/'impls/utils/evaluation.py','official_evaluation.py'),
            (ROOT/'experiments/ogbench_diagnostics/core.py','core.py'),(Path(__file__).with_name('evaluate.py'),'fixed_evaluate.py')]:shutil.copy2(source,snapshot/name)
    def budget():
        if time.monotonic()>deadline:raise TimeoutError('600-second fixed extraction budget exhausted; no extension')
    try:
        prior=read(root/'hard_goal_1000/analysis.json');assert prior['complete'] and all(v['success'] for v in prior['neural']['cases'].values())
        assert sha256(root/'hard_goal_1000/model/params_101000.pkl')==prior['training']['checkpoint_sha256']
        base,config=load_base();improved=restore_agent(base,str(root/'hard_goal_1000/model'),101000)
        assert config['actor_loss']=='ddpgbc' and config['const_std'] and config['alpha']==.0003 and config['lr']==.0003
        assert config['actor_p_trajgoal']==1 and config['actor_p_randomgoal']==config['actor_p_curgoal']==0
        initial=dict(original=base,improved=improved);counts={k:int(a.network.opt_state[0].count) for k,a in initial.items()}
        assert proof(base,['modules_actor','modules_dynamics','modules_lam'])==proof(improved,['modules_actor','modules_dynamics','modules_lam'])
        assert tree_equal(base.rng,improved.rng)
        fixed={k:proof(a,['modules_value','modules_lam']) for k,a in initial.items()}
        env,train,val=load_data(config);env.close();env=None
        np.random.seed(9001);validation=val.sample(2048);assert tree_digest(validation)==prior['protocol']['validation_sha256']
        initial_values={k:np.asarray(distances(a,validation['observations'],validation['value_goals'])) for k,a in initial.items()}
        protocol=dict(stage_steps=1000,batch_size=256,sampling_seed=2001,time_limit_seconds=600,config=json.loads(json.dumps(config)),
            base_sha256=BASE_SHA,improved_sha256=prior['training']['checkpoint_sha256'],prior_analysis_sha256=sha256(root/'hard_goal_1000/analysis.json'),
            initial_counts=counts,initial_frozen=fixed,validation_sha256=tree_digest(validation),expected_dynamics_stream_sha256=prior['training']['paired_stream_sha256'],
            official_episodes_per_task=50,fixed_bank_sha256=sha256(root/'evaluation_bank.npz'),evaluation_rng=0,
            source_sha256={p.name:sha256(p) for p in snapshot.iterdir()},devices=[str(d) for d in jax.devices()])
        (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');print('PROTOCOL',protocol,flush=True)
        phase='smoke';np.random.seed(2001);agents=dict(initial)
        for stage in ['dynamics','actor']:
            opts={k:init_optimizer(a,stage) for k,a in agents.items()};frozen={k:proof(a,[n for n in a.network.params if n!=module_key(stage)]) for k,a in agents.items()}
            for _ in range(2):
                budget();batch=train.sample(256)
                for arm in agents:
                    agents[arm],opts[arm],info=update(agents[arm],opts[arm],batch,stage);jax.block_until_ready(info)
                    assert all(np.isfinite(float(v)) for v in info.values())
            for arm,a in agents.items():assert proof(a,list(frozen[arm]))==frozen[arm]
        smoke=output/'smoke';smoke.mkdir()
        for arm,a in agents.items():
            directory=smoke/arm;directory.mkdir();epoch=counts[arm]+4;save_agent(a,str(directory),epoch)
            assert int(a.network.opt_state[0].count)==epoch and tree_equal(a,restore_agent(base,str(directory),epoch))
            assert proof(a,['modules_value','modules_lam'])==fixed[arm]
        (smoke/'verification.json').write_text(json.dumps(dict(passed=True,updates_per_arm=4,full_state_roundtrip=True,frozen_stages=True),indent=2)+'\n')
        agents=dict(initial);np.random.seed(2001);stage_streams={};stage_times={};checkpoints={};phase='training'
        for stage_number,stage in enumerate(['dynamics','actor']):
            stage_start=time.monotonic();opts={k:init_optimizer(a,stage) for k,a in agents.items()}
            frozen={k:proof(a,[n for n in a.network.params if n!=module_key(stage)]) for k,a in agents.items()}
            stream=hashlib.sha256();full_stream=hashlib.sha256()
            with (output/f'{stage}_metrics.csv').open('w') as handle:
                writer=None
                for step in range(1,1001):
                    budget();batch=train.sample(256)
                    for name in ['observations','next_observations','value_goals']:stream.update(batch[name].tobytes())
                    for name in ['observations','next_observations','value_goals','actions','actor_goals']:full_stream.update(batch[name].tobytes())
                    for arm in agents:
                        agents[arm],opts[arm],info=update(agents[arm],opts[arm],batch,stage);jax.block_until_ready(info);actual[arm][stage]=step
                        if step==1 or step%100==0:
                            row=dict(stage=stage,step=step,arm=arm,elapsed_seconds=time.monotonic()-started)
                            row.update({f'train/{k}':v for k,v in numbers(info).items()});row.update({f'heldout/{k}':v for k,v in numbers(statistics(agents[arm],validation)).items()})
                            assert all(np.isfinite(v) for v in row.values() if isinstance(v,(int,float)))
                            assert proof(agents[arm],list(frozen[arm]))==frozen[arm]
                            if writer is None:writer=csv.DictWriter(handle,fieldnames=list(row));writer.writeheader()
                            writer.writerow(row);handle.flush();print(json.dumps(row),flush=True)
            stage_streams[stage]=dict(original_three_fields_sha256=stream.hexdigest(),full_actor_fields_sha256=full_stream.hexdigest())
            if stage=='dynamics':assert stream.hexdigest()==protocol['expected_dynamics_stream_sha256']
            for arm,a in agents.items():
                budget();epoch=counts[arm]+1000*(stage_number+1);directory=output/arm/stage;directory.mkdir(parents=True)
                assert int(a.network.opt_state[0].count)==epoch and int(a.network.step)==epoch+1
                assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(a))
                assert proof(a,list(frozen[arm]))==frozen[arm] and proof(a,['modules_value','modules_lam'])==fixed[arm]
                np.testing.assert_array_equal(np.asarray(distances(a,validation['observations'],validation['value_goals'])),initial_values[arm])
                save_agent(a,str(directory),epoch);assert tree_equal(a,restore_agent(base,str(directory),epoch))
                checkpoints[arm+'/'+stage]=dict(epoch=epoch,sha256=sha256(directory/f'params_{epoch}.pkl'),frozen=True,full_state_roundtrip=True)
            with (output/f'{stage}_sampler_rng.pkl').open('wb') as handle:pickle.dump(np.random.get_state(),handle)
            stage_times[stage]=time.monotonic()-stage_start
            (output/'training_progress.json').write_text(json.dumps(dict(actual=actual,checkpoints=checkpoints,stage_streams=stage_streams,stage_seconds=stage_times),indent=2)+'\n')
        training=dict(passed=True,actual=actual,checkpoints=checkpoints,stage_streams=stage_streams,stage_seconds=stage_times,
            frozen_critic_and_lam=True,full_state_roundtrip=True,train_verify_seconds=time.monotonic()-started)
        (output/'training_complete.json').write_text(json.dumps(training,indent=2)+'\n');evaluations={};phase='evaluation'
        for arm,a in agents.items():
            budget();remaining=deadline-time.monotonic();epoch=counts[arm]+2000
            argv=[sys.executable,str(Path(__file__).with_name('evaluate_extraction.py')),str(root),'--checkpoint',str(output/arm/'actor'),
                '--epoch',str(epoch),'--critic-digest',tree_digest(initial[arm].network.params['modules_value']),'--output',str(output/f'{arm}_evaluation'),'--limit-seconds',str(remaining)]
            (output/f'{arm}_evaluation_argv.json').write_text(json.dumps(argv,indent=2)+'\n')
            with (output/f'{arm}_evaluation.log').open('w') as log:subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=remaining,cwd=ROOT)
            evaluations[arm]=read(output/f'{arm}_evaluation/analysis.json');assert evaluations[arm]['complete'] and evaluations[arm]['only_neural_actor']
        # Paired official samples must be identical even when closed-loop lengths differ.
        aa=evaluations['original']['official_episodes'];bb=evaluations['improved']['official_episodes'];assert list(aa)==list(bb) and len(aa)==250
        for name in aa:
            np.testing.assert_array_equal(aa[name]['initial_xy'],bb[name]['initial_xy']);np.testing.assert_array_equal(aa[name]['goal'],bb[name]['goal'])
        budget();assert sha256(BASE/'params_100000.pkl')==BASE_SHA
        assert sha256(root/'hard_goal_1000/model/params_101000.pkl')==protocol['improved_sha256']
        a=dict(complete=True,protocol=protocol,training=training,evaluations=evaluations,paired_250_initials_and_goals=True,elapsed_seconds=time.monotonic()-started)
        (output/'analysis.json').write_text(json.dumps(a,indent=2)+'\n');print('COMPLETE',a['elapsed_seconds'],flush=True)
    except BaseException as error:
        (output/'failure.json').write_text(json.dumps(dict(complete=False,error=repr(error),phase=phase,actual=actual,elapsed_seconds=time.monotonic()-started,traceback=traceback.format_exc()),indent=2)+'\n');raise
    finally:
        if env is not None:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--output',required=True);a=p.parse_args();run(a.root,a.output)
