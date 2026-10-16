"""Launch three matched arms; does not overwrite existing results."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--steps', type=int, default=5000)
    args = p.parse_args()
    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    jobs = []
    for arm in 'ABC':
        log = (root/f'{arm}.log').open('w')
        proc = subprocess.Popen([sys.executable,'-m','experiments.four_path_refinement.run',
            '--arm',arm,'--steps',str(args.steps),'--output',str(root/arm)],stdout=log,stderr=subprocess.STDOUT,
            env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MPLCONFIGDIR='/tmp/qrl-factorial-mpl'))
        jobs.append((arm,proc,log))
    failed = False
    while jobs:
        for job in list(jobs):
            arm,proc,log = job
            if proc.poll() is not None:
                log.close(); jobs.remove(job)
                print(f'{arm} exit={proc.returncode}',flush=True)
                failed |= proc.returncode != 0
        if jobs: time.sleep(5)
    raise SystemExit(1 if failed else 0)
