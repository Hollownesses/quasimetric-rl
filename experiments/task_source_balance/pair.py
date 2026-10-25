"""Prepare one seed, run its paired fixed budgets, then evaluate both endpoints."""
import argparse
from contextlib import ExitStack
from pathlib import Path
import subprocess
import sys

from .prepare import run as prepare


def run(output,seed,sampler_seed=2307):
    root=Path(output)
    prepare(str(root),seed,sampler_seed)
    processes=[]
    with ExitStack() as stack:
        try:
            for arm,mode in [('R0','record'),('R1','source')]:
                log=stack.enter_context((root/(arm+'.log')).open('w'))
                command=[sys.executable,'-u','-m','experiments.task_source_balance.train',
                    '--protocol',str(root/'protocol.json'),'--mode',mode,'--output',str(root/arm)]
                processes.append(subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT))
            codes=[process.wait() for process in processes]
            if any(codes):
                raise RuntimeError(f'Paired training failed (R0/R1 exit codes {codes}); inspect {root}/R*.log')
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
    for module in ('analyze','validate_results'):
        subprocess.run([sys.executable,'-m','experiments.task_source_balance.'+module,
                        '--root',str(root)],check=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--sampler-seed',type=int,default=2307)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    run(args.output,args.seed,args.sampler_seed)
