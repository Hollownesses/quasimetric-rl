"""Bounded subprocess execution; preserve every outcome and fail loudly."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from experiments.full_replay_four_path.evaluate import ARMS,OLD,bind,model_path
from experiments.four_path_execution.common import write,read,file_digest


def run(root,workers):
    historical=read(OLD/'manifest.json')
    for path,sha in historical['sources'].items():
        if path.startswith(('minimal_qrl/','experiments/four_path_execution/')):
            assert file_digest(path)==sha,path
    for arm,entry in historical['checkpoints'].items():
        assert file_digest(model_path(root,arm))==entry['sha256'],arm
    bind(root);pending=[('original','merged_606')]+[('independent',a) for a in ARMS]
    jobs=[];results={};start=time.perf_counter()
    while pending or jobs:
        while pending and len(jobs)<workers:
            group,arm=pending.pop(0);name=f'{group}_{arm}';log=(root/f'{name}.log').open('w')
            command=[sys.executable,'-m','experiments.full_replay_four_path.evaluate','arm','--root',str(root),'--group',group,'--arm',arm]
            proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,MPLCONFIGDIR='/tmp/qrl-execution-mpl'))
            jobs.append((name,proc,log))
        for job in list(jobs):
            name,proc,log=job
            if proc.poll() is None:continue
            jobs.remove(job);log.close();results[name]=dict(exit_code=proc.returncode,seconds=time.perf_counter()-start)
            write(root/'process_results.json',results);print(name,results[name],flush=True)
        if jobs:time.sleep(2)
    if any(r['exit_code'] for r in results.values()):raise RuntimeError(results)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();run(a.root,a.workers)
