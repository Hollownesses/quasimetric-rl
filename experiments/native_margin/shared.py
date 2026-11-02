"""Small, runtime-independent selection and audit helpers."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1 << 20), b''): h.update(part)
    return h.hexdigest()

def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')

def rows(path, records):
    keys = list(dict.fromkeys(k for r in records for k in r))
    with Path(path).open('w') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(records)

def select_pair(reference, fine=None, tol=1e-4):
    """Select without reading model scores. Stable order breaks reference ties."""
    reference = np.asarray(reference)
    good = np.flatnonzero(reference <= reference.min()+tol)
    bad = np.flatnonzero(reference > reference.min()+tol)
    bad = bad[np.argsort(reference[bad], kind='stable')]
    for b in bad:
        for a in good:
            if fine is None or fine[b] > fine[a]+tol: return int(a), int(b)
    return None

def source_weights(keys):
    _, inverse, count = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    return len(keys)/(len(count)*count[inverse]), inverse, count

def verify_inputs(raw):
    manifest = json.loads((REPO/'results/unified_mechanism_20261102/audit.json').read_text())
    required = ['results/connected_expert_utrap_20261021/main/verified_replay.pt',
                'results/connected_reference_fix_20261024/main/graph_reference_audit_only.npz',
                'results/task_source_balance_20261025/main/R0/final.pt',
                'results/ogbench_critic_20261029/official_counterfactual/official_task3_bank.npz',
                'experiments/ogbench_official_qrl/data/pointmaze-medium-navigate-v0.npz',
                'results/ogbench_official_qrl_20261026_seed0/main/OGBench/navigate_100k/sd000_20261026_115628/params_100000.pkl']
    result = {}
    for name in required:
        actual = sha(raw/name)
        assert actual == manifest['verified_sha256'][name], name
        result[name] = actual
    return result
