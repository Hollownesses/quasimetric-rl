"""Attach to the existing E1 producer; evaluate only its verified fixed endpoint."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
SCRIPTS=Path(__file__).resolve().parent


def process_signature(pid):
    r=subprocess.run(['/bin/ps','-ww','-p',str(pid),'-o','lstart=','-o','command='],
                     capture_output=True,text=True)
    if r.returncode==0:
        return r.stdout.strip() or None
    if r.returncode==1 and not r.stderr.strip():
        return None
    raise RuntimeError(f'Cannot observe producer PID {pid}: {r.stderr.strip()}')


def write_status(path,**values):
    target=path/'status.json'
    temporary=path/'status.json.tmp'
    temporary.write_text(json.dumps(dict(checked_unix=time.time(),**values),indent=2)+'\n')
    temporary.replace(target)


def completed(training):
    path=training/'training_complete.json'
    if not path.exists():return False
    data=json.loads(path.read_text())
    assert data['steps']==10000
    for arm in ['pair','goal']:
        assert data['final'][arm]['adam_count']==110000
        assert (training/arm/'params_110000.pkl').is_file()
    return True


def wait_for_endpoint(training,pid,signature,logdir,interval=30):
    while True:
        try:
            if completed(training):return
        except json.JSONDecodeError:
            # A live writer may be in the middle of its short final JSON write.
            if process_signature(pid)!=signature:raise
        observed=process_signature(pid)
        if observed is None:
            # Recheck once: process exit can race the final completion marker.
            if completed(training):return
            raise RuntimeError('Producer exited without a verified 10k endpoint; training will NOT restart.')
        if observed!=signature:
            raise RuntimeError('Producer PID identity changed; training will NOT restart.')
        write_status(logdir,phase='waiting_for_fixed_endpoint',producer_pid=pid,
                     producer_signature=signature,postprocess_pid=os.getpid())
        time.sleep(interval)


def checked_step(logdir,label,script,*arguments):
    command=[sys.executable,str(SCRIPTS/script),*[str(a) for a in arguments]]
    write_status(logdir,phase=label,postprocess_pid=os.getpid(),command=command)
    with (logdir/f'{label}.log').open('x') as f:
        result=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,
                              env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',MPLBACKEND='Agg'))
    if result.returncode:
        raise RuntimeError(f'{label} failed with exit {result.returncode}; log preserved, no restart.')


def finish(root,pid):
    root=Path(root).resolve();training=root/'paired_10k';logdir=root/'postprocess_e1'
    logdir.mkdir(exist_ok=True)
    # Exclusive launch marker: accidental duplicate observers must stop.
    with (logdir/'LAUNCHED').open('x') as f:f.write(str(time.time())+'\n')
    start=time.time()
    try:
        protocol_bytes=(training/'protocol.json').read_bytes()
        expected=(training/'protocol.sha256').read_text().strip()
        assert hashlib.sha256(protocol_bytes).hexdigest()==expected
        protocol=json.loads(protocol_bytes)
        assert protocol['steps']==10000 and protocol['seed']==2001
        assert (root/'evaluation_bank.npz').is_file()
        signature=process_signature(pid)
        try:
            ready=completed(training)
        except json.JSONDecodeError:
            if signature is None:raise
            ready=False
        if not ready:
            assert signature is not None, 'Existing producer must be live; no training is launched here.'
            assert 'ogbench_critic/train.py' in signature
            assert str(training) in signature or str(training.relative_to(ROOT)) in signature
            assert '--steps 10000' in signature and '--seed 2001' in signature
        initial=dict(producer_pid=pid,producer_signature=signature,postprocess_pid=os.getpid(),
            training_protocol_sha256=expected,bank_sha256=hashlib.sha256((root/'evaluation_bank.npz').read_bytes()).hexdigest(),
            source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in SCRIPTS.glob('*.py')},
            start_unix=start)
        (logdir/'protocol.json').write_text(json.dumps(initial,indent=2)+'\n')
        wait_for_endpoint(training,pid,signature,logdir)
        assert hashlib.sha256((training/'protocol.json').read_bytes()).hexdigest()==expected
        for name in ['verify.py','evaluate.py','field.py','compare.py','common.py','critic.py']:
            assert hashlib.sha256((SCRIPTS/name).read_bytes()).hexdigest()==initial['source_sha256'][name],name
        checked_step(logdir,'verify','verify.py',training,'--steps',10000,'--bank',root/'evaluation_bank.npz')
        for arm in ['pair','goal']:
            checked_step(logdir,f'{arm}_greedy','evaluate.py','--bank',root/'evaluation_bank.npz',
                '--output',root/f'{arm}_greedy','--checkpoint',training/arm,'--epoch',110000)
            checked_step(logdir,f'{arm}_field','field.py','--output',root/f'{arm}_field',
                '--checkpoint',training/arm,'--epoch',110000)
        checked_step(logdir,'compare','compare.py',root)
        write_status(logdir,phase='analysis_ready_for_review',postprocess_pid=os.getpid(),
                     elapsed_seconds=time.time()-start,report=str(root/'e1_comparison/report.md'))
        (logdir/'ANALYSIS_READY').write_text('Fixed endpoint analysis generated; researcher review and commit remain.\n')
    except BaseException:
        failure=traceback.format_exc()
        (logdir/'failure.log').write_text(failure)
        write_status(logdir,phase='failed',postprocess_pid=os.getpid(),error=failure,
                     elapsed_seconds=time.time()-start)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--producer-pid',type=int,required=True)
    a=p.parse_args();finish(a.root,a.producer_pid)
