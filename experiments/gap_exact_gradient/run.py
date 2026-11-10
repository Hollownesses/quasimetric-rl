"""Two exact-gap runs using the frozen, unchanged stage-one Adam loop."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.gap_stochastic import train as base_train
from experiments.gap_stochastic.core import QUERIES
from experiments.gap_exact_gradient.audit import check
from experiments.gap_exact_gradient.estimator import ExactGapSampler, AUDIT_COLUMNS

HERE = Path(__file__).resolve().parent


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--baseline', type=Path, default=ROOT/'results/gap_stochastic_20261108')
    args = parser.parse_args()
    baseline = json.loads((args.baseline/'manifest.json').read_text())
    assert baseline['status'] == 'complete' and baseline['config'] == base_train.CONFIG
    assert (args.baseline/'PROTOCOL.md').read_bytes() == (base_train.HERE/'PROTOCOL.md').read_bytes()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'PROTOCOL.md').write_bytes((HERE/'PROTOCOL.md').read_bytes())
    (args.output/'stage1_protocol.md').write_bytes((args.baseline/'PROTOCOL.md').read_bytes())
    sources = list(HERE.glob('*.py'))+list(base_train.HERE.glob('*.py'))
    baseline_files = ['manifest.json','evaluation.json','PROTOCOL.md']
    baseline_files += [f'stochastic__{name}__unbiased.npz' for name in QUERIES]
    manifest = dict(config=base_train.CONFIG, intervention='exact G^2 gradient for distance AND flow; other estimators unchanged',
                    python=sys.version, numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform(),
                    queries={name:mu.tolist() for name,mu in QUERIES.items()},
                    baseline=str(args.baseline.resolve()), status='preflight', runs={},
                    protocol_sha256=sha(HERE/'PROTOCOL.md'),
                    sources={str(p.relative_to(ROOT)):sha(p) for p in sources},
                    audit_columns=AUDIT_COLUMNS,
                    baseline_files={name:sha(args.baseline/name) for name in baseline_files})
    start = time.perf_counter()
    deadline = start+base_train.CONFIG['wall_seconds']
    try:
        preflight = check()
        save(args.output/'preflight.json', preflight)
        manifest['preflight_seconds'] = time.perf_counter()-start
        manifest['status'] = 'running'
        save(args.output/'manifest.json', manifest)
        print('Preflight passed: both gap blocks exact; stochastic conservation/risk retained; population objective unchanged.', flush=True)
        for name, mu in QUERIES.items():
            old_key = f'stochastic__{name}__unbiased'
            baseline_path = args.baseline/(old_key+'.npz')
            sampler = ExactGapSampler(mu)
            arrays, info = base_train.run(mu, 'unbiased', 'stochastic', deadline, sampler=sampler)
            audit = sampler.arrays()
            assert audit.shape == (base_train.CONFIG['steps'],len(AUDIT_COLUMNS))
            assert info['sampling_sha256'] == baseline['runs'][old_key]['sampling_sha256']
            assert info['initial_sha256'] == baseline['runs'][old_key]['initial_sha256']
            with np.load(baseline_path) as prior:
                np.testing.assert_array_equal(arrays['x'][0], prior['x'][0])
                np.testing.assert_array_equal(arrays['step'], prior['step'])
            assert sha(baseline_path) == manifest['baseline_files'][baseline_path.name]
            errors = {key:float(np.max(audit[:,j])) for j,key in enumerate(AUDIT_COLUMNS) if j>=5}
            assert max(errors.values()) < 1e-10
            info.update(baseline_key=old_key, initial_and_sampling_match=True,
                        intervention='exact_gap', gradient_audit_maximum_errors=errors)
            np.savez_compressed(args.output/(name+'.npz'), **arrays, gradient_audit=audit)
            manifest['runs'][name] = info
            save(args.output/'manifest.json',manifest)
            print(json.dumps(dict(query=name, seconds=info['elapsed_seconds'],maximum_audit_error=max(errors.values()))),flush=True)
        for name, fingerprint in manifest['baseline_files'].items():
            assert sha(args.baseline/name) == fingerprint
        manifest['status']='complete'
    except Exception as exc:
        manifest['status']='failed'
        manifest['error']=repr(exc)
        raise
    finally:
        manifest['total_seconds']=time.perf_counter()-start
        save(args.output/'manifest.json',manifest)


if __name__ == '__main__':
    main()
