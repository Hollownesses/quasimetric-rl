"""Fixed-budget paired minibatch runs. This module imports no evaluator."""
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.gap_stochastic.core import (
    initial, project, QUERIES, sampled, population, CONE, EDGE, COST, PROJECTION_EVENTS)

HERE = Path(__file__).resolve().parent
CONFIG = dict(seed=2001, steps=20000, batch=256, lam=1., beta=1.,
              lr_start=.01, lr_end=.0001, decay_start=12000,
              qrl_cap=.05, qrl_dual_lr=.05, qrl_initial_multiplier=1.,
              save_every=500, wall_seconds=600)


def rate(step):
    progress = max(0, step-CONFIG['decay_start'])/(CONFIG['steps']-1-CONFIG['decay_start'])
    return CONFIG['lr_start']*(CONFIG['lr_end']/CONFIG['lr_start'])**progress


def run(mu, method, mode, deadline):
    x = initial()
    m, v = np.zeros(36), np.zeros(36)
    rng = np.random.default_rng(CONFIG['seed'])
    digest = hashlib.sha256()
    multiplier = CONFIG['qrl_initial_multiplier']
    snapshots = [x.copy()]
    steps, multipliers, losses = [0], [multiplier], [None]
    start = time.perf_counter()
    window_sum = 0.
    min_sample_loss = np.inf
    max_cone_violation = 0.
    max_flow_violation = 0.
    projection_events = []
    for step in range(CONFIG['steps']):
        if step % 500 == 0 and time.perf_counter() > deadline:
            raise TimeoutError('frozen total computation budget exceeded')
        if mode == 'stochastic':
            edges = rng.integers(0, 7, CONFIG['batch'])
            uniforms = rng.random(CONFIG['batch'])
            queries = np.searchsorted(np.cumsum(mu), uniforms)
            digest.update(edges.astype('<i8').tobytes())
            digest.update(uniforms.astype('<f8').tobytes())
            loss, gradient, risk = sampled(x, edges, queries, method,
                                          lam=multiplier if method == 'qrl' else CONFIG['lam'],
                                          beta=CONFIG['beta'])
        else:
            loss, gradient, _ = population(x, mu, lam=CONFIG['lam'], beta=CONFIG['beta'],
                                           batch=CONFIG['batch'],
                                           diagonal_gap=method == 'naive', diagonal_flow=method == 'naive')
        lr = rate(step)
        m = .9*m+.1*gradient
        v = .999*v+.001*gradient*gradient
        corrected_m = m/(1-.9**(step+1))
        metric = np.sqrt(v/(1-.999**(step+1)))+1e-8
        candidate = x-lr*corrected_m/metric
        previous_events = len(PROJECTION_EVENTS)
        x = project(candidate, metric)
        for event in PROJECTION_EVENTS[previous_events:]:
            projection_events.append(dict(step=step+1, **event))
        if method == 'qrl':
            multiplier = max(0., multiplier+CONFIG['qrl_dual_lr']*(lr/CONFIG['lr_start'])*(risk-CONFIG['qrl_cap']))
        window_sum += loss
        min_sample_loss = min(min_sample_loss, loss)
        max_cone_violation = max(max_cone_violation, float(-np.min(CONE@x[:30])))
        max_flow_violation = max(max_flow_violation, float(-x[30:].min()))
        if (step+1) % CONFIG['save_every'] == 0:
            snapshots.append(x.copy())
            steps.append(step+1)
            multipliers.append(multiplier)
            losses.append(window_sum/CONFIG['save_every'])
            window_sum = 0.
    return dict(x=np.array(snapshots), step=np.array(steps), multiplier=np.array(multipliers)), dict(
        elapsed_seconds=time.perf_counter()-start, method=method, mode=mode,
        sampling_sha256=digest.hexdigest() if mode == 'stochastic' else None,
        initial_sha256=hashlib.sha256(initial().tobytes()).hexdigest(),
        mean_sample_losses=losses, minimum_sample_loss=float(min_sample_loss),
        maximum_training_cone_violation=max_cone_violation,
        maximum_training_flow_nonneg_violation=max_flow_violation,
        projection_events=projection_events)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--resume', action='store_true', help='Only resume a numerical failure; completed runs are preserved.')
    args = parser.parse_args()
    current_sources = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}
    if args.resume:
        manifest = json.loads((args.output/'manifest.json').read_text())
        assert manifest['status'] == 'failed' and manifest['config'] == CONFIG
        assert (args.output/'PROTOCOL.md').read_bytes() == (HERE/'PROTOCOL.md').read_bytes()
        (args.output/'manifest_before_projection_fix.json').write_text(json.dumps(manifest, indent=2)+'\n')
        manifest['source_sha256_after_projection_fix'] = current_sources
        manifest['recovery_reason'] = 'Same weighted projection via SLSQP fallback for SciPy NNLS empty active set; no completed runs repeated.'
        manifest['recovered_from_error'] = manifest.pop('error', None)
        previous_seconds = manifest['total_training_seconds']
        manifest['status'] = 'running'
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output/'PROTOCOL.md').write_bytes((HERE/'PROTOCOL.md').read_bytes())
        manifest = dict(config=CONFIG, python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                    platform=platform.platform(), protocol_sha256=hashlib.sha256((HERE/'PROTOCOL.md').read_bytes()).hexdigest(),
                    source_sha256=current_sources,
                    status='running', runs={})
        previous_seconds = 0.
    start = time.perf_counter()
    deadline = start+CONFIG['wall_seconds']-previous_seconds
    try:
        for mode in ('stochastic', 'population'):
            for name, mu in QUERIES.items():
                for method in (('qrl', 'naive', 'unbiased') if mode == 'stochastic' else ('naive', 'unbiased')):
                    key = f'{mode}__{name}__{method}'
                    if key in manifest['runs']:
                        continue
                    arrays, info = run(mu, method, mode, deadline)
                    np.savez_compressed(args.output/(key+'.npz'), **arrays)
                    manifest['runs'][key] = info
                    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
                    print(json.dumps(dict(run=key, seconds=info['elapsed_seconds'], loss=info['mean_sample_losses'][-1])), flush=True)
        manifest['status'] = 'complete'
    except Exception as exc:
        manifest['status'] = 'failed'
        manifest['error'] = repr(exc)
        raise
    finally:
        manifest['total_training_seconds'] = time.perf_counter()-start+previous_seconds
        (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
