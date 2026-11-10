"""Reproduce only the requested two-case gradient ablation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    output=args.output.resolve()
    if output.exists():
        raise FileExistsError('Choose a new output directory.')
    here=Path(__file__).resolve().parent
    root=here.parents[1]
    env=os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    start=time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='qrl-route-noise-') as temp:
        env['MPLCONFIGDIR']=str(Path(temp)/'matplotlib')
        for script in ('run.py','evaluate.py','finish.py'):
            subprocess.run([sys.executable,str(here/script),str(output)],cwd=root,env=env,check=True)
    seconds=time.perf_counter()-start
    assert seconds<600
    (output/'pipeline.json').write_text(json.dumps(dict(completed=True,seconds=seconds),indent=2)+'\n')


if __name__=='__main__':
    main()
