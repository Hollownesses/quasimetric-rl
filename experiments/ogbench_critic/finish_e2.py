"""Observe existing E2 training and consume its fixed endpoint; never train."""
import argparse
import json
import os
from pathlib import Path
import time
import traceback

from finish_e1 import ROOT,SCRIPTS,process_signature,write_status,checked_step
from compare_multistep import digest


def freeze_sources(training):
    """Archive the producer-written hashes without changing the running protocol."""
    protocol=json.loads((training/'protocol.json').read_text())
    expected=digest(training/'protocol.json');hashfile=training/'protocol.sha256'
    if hashfile.exists():assert hashfile.read_text().strip()==expected
    else:
        with hashfile.open('x') as f:f.write(expected+'\n')
    snapshot=training/'source_snapshot';snapshot.mkdir(exist_ok=True)
    sources={**protocol['source_sha256'],'MULTISTEP_PROTOCOL.md':protocol['protocol_sha256']}
    for name,sha in sources.items():
        assert Path(name).name==name
        target=snapshot/name
        if target.exists():assert digest(target)==sha,name
        else:
            assert digest(SCRIPTS/name)==sha,name
            with target.open('xb') as f:f.write((SCRIPTS/name).read_bytes())


def completed(training):
    marker=training/'training_complete.json'
    if not marker.exists():return False
    done=json.loads(marker.read_text())
    assert done['steps']==10000 and done['source_transitions_per_arm']==2560000
    for arm in ['one','multi']:
        assert done['final'][arm]['step']==done['final'][arm]['adam_count']==10000
        assert (training/arm/'critic_10000.pkl').is_file()
    return True


def wait_for_endpoint(training,pid,signature,logdir,interval=30):
    while True:
        try:
            if completed(training):return
        except json.JSONDecodeError:
            if process_signature(pid)!=signature:raise
        observed=process_signature(pid)
        if observed is None:
            if completed(training):return
            raise RuntimeError('E2 producer exited without a verified 10k endpoint; no restart.')
        if observed!=signature:
            raise RuntimeError('E2 producer PID identity changed; no restart.')
        write_status(logdir,phase='waiting_for_fixed_endpoint',producer_pid=pid,
                     producer_signature=signature,postprocess_pid=os.getpid())
        time.sleep(interval)


def finish(root,pid):
    root=Path(root).resolve();training=root/'multistep_10k';logdir=root/'postprocess_e2'
    logdir.mkdir(exist_ok=True)
    with (logdir/'LAUNCHED').open('x') as f:f.write(str(time.time())+'\n')
    start=time.time()
    try:
        freeze_sources(training)
        protocol=json.loads((training/'protocol.json').read_text())
        expected=(training/'protocol.sha256').read_text().strip()
        assert digest(training/'protocol.json')==expected
        assert protocol['steps']==10000 and protocol['seed']==4501 and protocol['training_pid']==pid
        assert (root/'evaluation_bank.npz').is_file()
        signature=process_signature(pid)
        try:ready=completed(training)
        except json.JSONDecodeError:
            if signature is None:raise
            ready=False
        if not ready:
            assert signature is not None,'Existing training must be live; this script cannot start it.'
            assert 'ogbench_critic/train_multistep.py' in signature
            assert str(training) in signature or str(training.relative_to(ROOT)) in signature
            assert '--steps 10000' in signature and '--seed 4501' in signature
        initial=dict(producer_pid=pid,producer_signature=signature,postprocess_pid=os.getpid(),start_unix=start,
                     training_protocol_sha256=expected,bank_sha256=digest(root/'evaluation_bank.npz'),
                     source_sha256={p.name:digest(p) for p in SCRIPTS.glob('*.py')})
        (logdir/'protocol.json').write_text(json.dumps(initial,indent=2)+'\n')
        # Critical methods must equal the already-running producer's snapshot.
        critical=['multistep.py','train_multistep.py','common.py','train.py','evaluate.py','field.py','verify_multistep.py']
        for name in critical:
            assert digest(SCRIPTS/name)==digest(training/'source_snapshot'/name)==protocol['source_sha256'][name],name
        wait_for_endpoint(training,pid,signature,logdir)
        assert digest(training/'protocol.json')==expected
        assert digest(root/'evaluation_bank.npz')==initial['bank_sha256']
        for name in critical+['finish_e2.py','compare_multistep.py','compare.py','finish_e1.py']:
            assert digest(SCRIPTS/name)==initial['source_sha256'][name],name
        checked_step(logdir,'verify','verify_multistep.py',training,'--steps',10000,'--bank',root/'evaluation_bank.npz')
        for arm in ['one','multi']:
            checked_step(logdir,f'{arm}_greedy','evaluate.py','--bank',root/'evaluation_bank.npz',
                '--output',root/f'e2_{arm}_greedy','--checkpoint',training/arm,'--epoch',10000,'--checkpoint-kind','multistep')
            checked_step(logdir,f'{arm}_field','field.py','--output',root/f'e2_{arm}_field',
                '--checkpoint',training/arm,'--epoch',10000,'--checkpoint-kind','multistep')
        checked_step(logdir,'compare','compare_multistep.py',root)
        write_status(logdir,phase='analysis_ready_for_review',postprocess_pid=os.getpid(),
                     elapsed_seconds=time.time()-start,report=str(root/'e2_comparison/report.md'))
        (logdir/'ANALYSIS_READY').write_text('Fixed endpoint generated. Researcher review and experiment commit remain.\n')
    except BaseException:
        failure=traceback.format_exc();(logdir/'failure.log').write_text(failure)
        write_status(logdir,phase='failed',postprocess_pid=os.getpid(),error=failure,elapsed_seconds=time.time()-start)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--producer-pid',type=int,required=True)
    a=p.parse_args();finish(a.root,a.producer_pid)
