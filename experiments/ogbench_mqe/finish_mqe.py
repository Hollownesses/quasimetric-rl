"""Attach to native MQE training, then evaluate only its fixed 100k endpoint."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

SCRIPTS=Path(__file__).resolve().parent;REPO=SCRIPTS.parents[1]
sys.path.insert(0,str(REPO/'experiments/ogbench_critic'))
from finish_e1 import process_signature,write_status


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def completed(training):
    runtime=training/'runtime.json'
    if not runtime.exists():return False
    status=json.loads(runtime.read_text())
    if status['exit_code']!=0:raise RuntimeError(f'Native main failed with exit {status["exit_code"]}; no training restart.')
    if not (training/'TRAINING_COMPLETE').exists():return False
    directories=list(training.glob('OGBench/*/*/flags.json'));assert len(directories)==1
    checkpoint=directories[0].parent;flags=json.loads(directories[0].read_text())
    assert flags['seed']==0 and flags['train_steps']==100000 and flags['eval_episodes']==50
    assert flags['agent']['agent_name']=='mqe' and flags['restore_path'] is None
    assert (checkpoint/'params_100000.pkl').is_file()
    with (checkpoint/'eval.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    assert int(float(rows[-1]['step']))==100000
    return checkpoint


def wait_for_endpoint(training,pid,signature,launcher,launcher_signature,logdir,interval=30):
    while True:
        try:
            ready=completed(training)
            if ready:return ready
        except json.JSONDecodeError:
            if process_signature(launcher)!=launcher_signature:raise
        observed=process_signature(pid)
        if observed is not None and observed!=signature:raise RuntimeError('Native producer PID identity changed; no restart.')
        live_launcher=process_signature(launcher)
        if live_launcher is not None and live_launcher!=launcher_signature:raise RuntimeError('Launcher PID identity changed; no restart.')
        if observed is None and live_launcher is None:
            ready=completed(training)
            if ready:return ready
            raise RuntimeError('Native producer and launcher exited without a verified endpoint; no restart.')
        write_status(logdir,phase='waiting_for_fixed_endpoint',producer_pid=pid,producer_signature=signature,
                     launcher_pid=launcher,postprocess_pid=os.getpid())
        time.sleep(interval)


def checked_step(logdir,label,script,*arguments):
    command=[sys.executable,str(SCRIPTS/script),*[str(v) for v in arguments]]
    write_status(logdir,phase=label,postprocess_pid=os.getpid(),command=command)
    with (logdir/f'{label}.log').open('x') as f:
        result=subprocess.run(command,cwd=REPO,stdout=f,stderr=subprocess.STDOUT,
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',MPLBACKEND='Agg'))
    if result.returncode:raise RuntimeError(f'{label} failed with exit {result.returncode}; logs preserved, no restart.')


def finish(root,pid,launcher):
    root=Path(root).resolve();training=root/'mqe_100k_seed0';logdir=root/'postprocess_mqe'
    logdir.mkdir(exist_ok=True)
    with (logdir/'LAUNCHED').open('x') as f:f.write(str(time.time())+'\n')
    start=time.time()
    try:
        expected=(training/'frozen_config.sha256').read_text().strip()
        assert sha(training/'frozen_config.json')==expected
        frozen=json.loads((training/'frozen_config.json').read_text())
        assert frozen['endpoint']==100000 and frozen['flags']['agent']['alpha']==.03
        saved_pids=json.loads((training/'pid.json').read_text());assert saved_pids==dict(launcher=launcher,training=pid)
        signature=process_signature(pid);launcher_signature=process_signature(launcher)
        ready=completed(training)
        if not ready:
            assert signature and launcher_signature
            assert 'ogbench_mqe/run_official.py' in signature and '--train_steps=100000' in signature
            assert str(training) in signature and 'ogbench_mqe/launch.py' in launcher_signature
        sources={p.name:sha(p) for p in SCRIPTS.glob('*.py')}
        core=REPO/'experiments/ogbench_diagnostics/core.py'
        manifest=dict(producer_pid=pid,producer_signature=signature,launcher_pid=launcher,launcher_signature=launcher_signature,
            postprocess_pid=os.getpid(),training_config_sha256=expected,bank_sha256=sha(root/'evaluation_bank.npz'),
            source_sha256=sources,core_sha256=sha(core),start_unix=start,
            purpose='Fixed endpoint diagnostics only. Research analysis, figures and commit remain.')
        (logdir/'protocol.json').write_text(json.dumps(manifest,indent=2)+'\n')
        snapshot=logdir/'source_snapshot';snapshot.mkdir(exist_ok=False)
        for name in sources:(snapshot/name).write_bytes((SCRIPTS/name).read_bytes())
        (snapshot/'core.py').write_bytes(core.read_bytes())
        for name,digest in frozen['local_source_sha256'].items():assert sha(SCRIPTS/name)==digest,name
        checkpoint=wait_for_endpoint(training,pid,signature,launcher,launcher_signature,logdir)
        assert sha(training/'frozen_config.json')==expected and sha(root/'evaluation_bank.npz')==manifest['bank_sha256']
        for name,digest in sources.items():assert sha(SCRIPTS/name)==digest,name
        assert sha(core)==manifest['core_sha256']
        checked_step(logdir,'diagnostics','diagnostics.py','--checkpoint',checkpoint,'--epoch',100000,
                     '--bank',root/'evaluation_bank.npz','--output',root/'mqe_100k_diagnostics')
        checked_step(logdir,'verify','verify_diagnostics.py',root/'mqe_100k_diagnostics',
                     '--bank',root/'evaluation_bank.npz','--baseline-field',root/'baseline_field')
        # The existing shell job owns this full checkpoint roundtrip; do not run it twice.
        verified=json.loads((training/'checkpoint_verification.json').read_text())
        assert verified['passed'] and verified['adam_count']==100000 and verified['full_state_roundtrip']
        assert verified['checkpoint_sha256']==sha(checkpoint/'params_100000.pkl')
        write_status(logdir,phase='diagnostics_ready_for_analysis',postprocess_pid=os.getpid(),elapsed_seconds=time.time()-start)
        (logdir/'DIAGNOSTICS_READY').write_text('Numerical endpoint diagnostics verified. Analysis, figures, review and commit remain.\n')
    except BaseException:
        failure=traceback.format_exc();(logdir/'failure.log').write_text(failure)
        write_status(logdir,phase='failed',postprocess_pid=os.getpid(),error=failure,elapsed_seconds=time.time()-start)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--producer-pid',type=int,required=True)
    p.add_argument('--launcher-pid',type=int,required=True);a=p.parse_args();finish(a.root,a.producer_pid,a.launcher_pid)
