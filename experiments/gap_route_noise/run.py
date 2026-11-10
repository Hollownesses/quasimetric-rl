"""One pre-Adam gradient intervention, with the frozen stage-one trainer."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.gap_stochastic import train as base_train
from experiments.gap_stochastic.core import QUERIES
from experiments.gap_route_noise.audit import check
from experiments.gap_route_noise.estimator import ProjectedSampler, Z, AUDIT_COLUMNS

HERE = Path(__file__).resolve().parent


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


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
    sources = list(HERE.glob('*.py'))+[base_train.HERE/'train.py', base_train.HERE/'core.py']
    manifest = dict(config=base_train.CONFIG, intervention='project only raw conservation gradient before Adam',
                    direction=Z.tolist(), baseline=str(args.baseline.resolve()), status='preflight', runs={},
                    protocol_sha256=hashlib.sha256((HERE/'PROTOCOL.md').read_bytes()).hexdigest(),
                    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                    audit_columns=AUDIT_COLUMNS, baseline_files={})
    start = time.perf_counter()
    deadline = start+base_train.CONFIG['wall_seconds']
    try:
        # Exact expectation and every-enumerated-batch checks precede training.
        preflight = check()
        save(args.output/'preflight.json', preflight)
        manifest['preflight_seconds'] = time.perf_counter()-start
        manifest['status'] = 'running'
        save(args.output/'manifest.json', manifest)
        print('Preflight passed: projected conservation has zero route component; expectation preserved.', flush=True)
        for name, mu in QUERIES.items():
            old_key = f'stochastic__{name}__unbiased'
            baseline_path = args.baseline/(old_key+'.npz')
            baseline_hash = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
            manifest['baseline_files'][old_key] = baseline_hash
            sampler = ProjectedSampler()
            arrays, info = base_train.run(mu, 'unbiased', 'stochastic', deadline, sampler=sampler)
            audit = sampler.arrays()
            assert len(audit) == base_train.CONFIG['steps']
            assert info['sampling_sha256'] == baseline['runs'][old_key]['sampling_sha256']
            assert info['initial_sha256'] == baseline['runs'][old_key]['initial_sha256']
            with np.load(baseline_path) as prior:
                np.testing.assert_array_equal(arrays['x'][0], prior['x'][0])
                np.testing.assert_array_equal(arrays['step'], prior['step'])
            assert hashlib.sha256(baseline_path.read_bytes()).hexdigest() == baseline_hash
            info.update(baseline_key=old_key, initial_and_sampling_match=True,
                        maximum_raw_conservation_route=float(np.max(abs(audit[:,1]))),
                        rms_raw_conservation_route=float(np.sqrt(np.mean(audit[:,1]**2))),
                        maximum_projected_conservation_route=float(np.max(abs(audit[:,2]))),
                        maximum_gap_preservation_error=float(np.max(audit[:,5])),
                        maximum_modification_error=float(np.max(audit[:,6])),
                        maximum_cons_moment_route=float(np.max(abs(audit[:,7]))),
                        rms_adam_scaled_cons_route=float(np.sqrt(np.mean(audit[:,8]**2))))
            np.savez_compressed(args.output/(name+'.npz'), **arrays, gradient_audit=audit)
            manifest['runs'][name] = info
            save(args.output/'manifest.json',manifest)
            print(json.dumps(dict(query=name, seconds=info['elapsed_seconds'],
                                  maximum_projected_route=info['maximum_projected_conservation_route'])),flush=True)
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
