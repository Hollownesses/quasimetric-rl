"""Reproduce the frozen stage-one prototype in a new output directory."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    here = Path(__file__).resolve().parent
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('Choose a new output directory; existing results are never overwritten.')
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='qrl-gap-stage1-') as temporary:
        env['MPLCONFIGDIR'] = str(Path(temporary)/'matplotlib')
        preflight = Path(temporary)/'preflight.json'
        def execute(script, arg):
            subprocess.run([sys.executable, str(here/script), str(arg)], cwd=root, env=env, check=True)
        execute('audit.py', preflight)
        execute('train.py', output)
        shutil.copyfile(preflight, output/'preflight.json')
        execute('evaluate.py', output)
        execute('diagnose.py', output)
        execute('plot.py', output)
    seconds = time.perf_counter()-start
    assert seconds < 600
    (output/'pipeline.json').write_text(json.dumps(dict(completed=True, seconds=seconds), indent=2)+'\n')


if __name__ == '__main__':
    main()
