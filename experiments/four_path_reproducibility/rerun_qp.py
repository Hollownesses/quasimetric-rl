"""Transparent common numerical correction: rerun all QPs, preserve training/results."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from experiments.four_path_factorial.run import file_digest
from experiments.four_path_reproducibility.run import SEEDS


def run(source,output):
    assert tuple(json.loads((source/'manifest.json').read_text())['seeds'])==SEEDS
    output.mkdir(parents=True,exist_ok=False)
    sources={}
    for name in ('experiments/four_path_refinement/repair.py',__file__):
        path=Path(name).resolve();relative=path.relative_to(Path.cwd())
        destination=output/'source_snapshot'/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,destination);sources[str(relative)]=file_digest(path)
    checkpoints={}
    for seed in SEEDS:
        directory=source/f'seed_{seed}'
        cfg=json.loads((directory/'config.json').read_text())
        assert cfg['seed']==seed and cfg['phr_steps']==20000 and cfg['decay_steps']==5000
        checkpoints[str(seed)]=file_digest(directory/'decay.pt')
    (output/'manifest.json').write_text(json.dumps(dict(seeds=SEEDS,source=str(source),
        reason='same mathematical QP; stable Jacobian-block Gram fallback for cancellation; all five rerun without retuning',
        sources=sources,decay_checkpoints=checkpoints,delta=.15,iterations=40,push_step=0.),indent=2)+'\n')
    jobs=[];results={};start=time.perf_counter()
    for seed in SEEDS:
        log=(output/f'seed_{seed}.log').open('w')
        cmd=[sys.executable,'-m','experiments.four_path_refinement.repair',
             '--checkpoint',str(source/f'seed_{seed}/decay.pt'),'--output',str(output/f'seed_{seed}'),
             '--delta','0.15','--iterations','40','--push-step','0']
        proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,
             env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MPLCONFIGDIR='/tmp/qrl-factorial-mpl'))
        jobs.append((seed,proc,log))
    while jobs:
        for job in list(jobs):
            seed,proc,log=job
            if proc.poll() is None:continue
            log.close();jobs.remove(job)
            results[str(seed)]=dict(exit_code=proc.returncode,seconds_since_suite_start=time.perf_counter()-start)
            (output/'process_results.json').write_text(json.dumps(results,indent=2)+'\n')
            print(json.dumps(dict(seed=seed,**results[str(seed)])),flush=True)
        if jobs:time.sleep(2)
    return int(any(r['exit_code']!=0 for r in results.values()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();raise SystemExit(run(Path(args.source),Path(args.output)))
