"""Preserve all five seeds, including process failures, without retuning."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

SEEDS=(101,202,303,404,505)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--smoke',action='store_true');args=p.parse_args()
    root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
    sources=list(Path('experiments/four_path_reproducibility').glob('*'))
    sources += [Path('experiments/four_path_factorial/run.py'),Path('experiments/four_path_refinement/run.py'),
                Path('experiments/four_path_refinement/repair.py'),
                Path('quasimetric_rl/modules/quasimetric_critic/losses/phr.py'),
                Path('quasimetric_rl/modules/quasimetric_critic/losses/feasible_direction.py')]
    manifest={}
    for path in sources:
        if not path.is_file():continue
        target=root/'source_snapshot'/path;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target);manifest[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    seeds=(9001,) if args.smoke else SEEDS
    (root/'manifest.json').write_text(json.dumps(dict(seeds=seeds,smoke=args.smoke,sources=manifest,
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()),indent=2)+'\n')
    jobs=[];results={};started=time.perf_counter()
    for seed in seeds:
        log=(root/f'seed_{seed}.log').open('w')
        command=[sys.executable,'-m','experiments.four_path_reproducibility.run','--seed',str(seed),'--output',str(root/f'seed_{seed}')]
        if args.smoke:command+=['--phr-steps','20','--decay-steps','20','--qp-iterations','2']
        proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
            env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MPLCONFIGDIR='/tmp/qrl-factorial-mpl'))
        jobs.append((seed,proc,log))
    while jobs:
        for job in list(jobs):
            seed,proc,log=job
            if proc.poll() is None:continue
            log.close();jobs.remove(job)
            results[str(seed)]=dict(exit_code=proc.returncode,seconds_since_suite_start=time.perf_counter()-started)
            (root/'process_results.json').write_text(json.dumps(results,indent=2)+'\n')
            print(json.dumps(dict(seed=seed,**results[str(seed)])),flush=True)
        if jobs:time.sleep(5)
    raise SystemExit(int(any(r['exit_code']!=0 for r in results.values())))
