"""Bounded process concurrency; every arm result/error is retained."""
import argparse
import os
import subprocess
import sys
import time
from experiments.four_path_execution.common import *


def run(root,workers):
    manifest=read(root/'manifest.json');pending=list(manifest['arms']);jobs=[];results={};start=time.perf_counter()
    if (root/'process_results.json').exists():raise FileExistsError('Use a fresh frozen output tree')
    while pending or jobs:
        while pending and len(jobs)<workers:
            arm=pending.pop(0);log=(root/f'{arm}.log').open('w')
            command=[sys.executable,'-m','experiments.four_path_execution.run','--root',str(root),'--arm',arm]
            proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLCONFIGDIR='/tmp/qrl-execution-mpl'))
            jobs.append((arm,proc,log))
        for job in list(jobs):
            arm,proc,log=job
            if proc.poll() is None:continue
            log.close();jobs.remove(job)
            results[arm]=dict(exit_code=proc.returncode,suite_elapsed_seconds=time.perf_counter()-start)
            write(root/'process_results.json',results);print(json.dumps(dict(arm=arm,**results[arm])),flush=True)
        if jobs:time.sleep(2)
    return int(any(r['exit_code'] for r in results.values()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--workers',type=int,default=4)
    args=p.parse_args();raise SystemExit(run(Path(args.root),args.workers))
