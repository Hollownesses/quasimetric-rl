"""Reproduce the registered suite in a new directory, with four CPU jobs."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import os
import subprocess
import sys


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--jobs',type=int,default=4)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    logs=args.output/'logs'; logs.mkdir()
    env=dict(os.environ,MPLCONFIGDIR='/tmp/qrl-factorial-mpl',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    script=Path(__file__).with_name('run.py')
    def cell(pair):
        parameterization,method=pair
        for critic,steps in ((0,20000),(1,5000)):
            name=f'{parameterization}_{method}_c{critic}'
            command=[sys.executable,str(script),'--output',str(args.output/'main'/name),
                '--parameterization',parameterization,'--method',method,'--critic',str(critic),
                '--steps',str(steps),'--lr','0.00002' if parameterization=='encoder' else '0.01',
                '--rho','100','--dual-step','1','--threads','2']
            with (logs/(name+'.log')).open('w') as log:
                subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            print('Completed '+name,flush=True)
    pairs=[(param,method) for param in ('encoder','latent') for method in ('barrier','phr')]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(cell,pairs))
    for name in ('summarize.py','audit.py','write_report.py'):
        subprocess.run([sys.executable,str(script.with_name(name)),str(args.output)],env=env,check=True)

if __name__=='__main__': main()
