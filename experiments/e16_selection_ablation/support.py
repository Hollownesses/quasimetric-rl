"""Historical inputs are bound to the worktree recorded by the unified diagnosis."""
import hashlib
import json
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
UNIFIED=REPO/'results/unified_mechanism_20261102'

def read(path):return json.loads(Path(path).read_text())

def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

RAW=Path(os.environ.get('ABLATION_RAW') or read(UNIFIED/'audit.json')['raw_root']).resolve()
CODE=RAW/'experiments/ogbench_critic'
G=RAW/'results/ogbench_critic_20261029'
sys.path[:0]=[str(CODE),str(RAW)]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1<<20),b''):h.update(part)
    return h.hexdigest()

def check(path,expected):
    actual=sha(path)
    if actual!=expected:raise ValueError(f'{path}: expected {expected}, found {actual}')
    return actual
