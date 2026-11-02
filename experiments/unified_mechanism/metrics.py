"""Shared signed diagnostics. Costs are positive and distances point forward."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

TOL = 1e-4


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def verified(path, expected):
    digest = sha(path)
    if digest != expected:
        raise ValueError(f'Fingerprint mismatch {path}: {digest}')
    return digest


def write_rows(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read(path):
    return json.loads(Path(path).read_text())


def stats(x):
    x = np.asarray(x, dtype=np.float64)
    assert len(x) and np.isfinite(x).all()
    return dict(n=x.size, mean=x.mean(), p05=np.quantile(x,.05), median=np.median(x),
                p95=np.quantile(x,.95), p99=np.quantile(x,.99), maximum=x.max(),
                positive_mean=np.maximum(x,0).mean(), positive_fraction=(x>TOL).mean(),
                negative_fraction=(x < -TOL).mean())


def path_metrics(cost, edge, value):
    cost, edge, value = (np.asarray(x, np.float64) for x in (cost, edge, value))
    assert len(cost) == len(edge) == len(value)-1 and (cost > 0).all()
    h, tau = edge-cost, edge+np.diff(value)
    # Check every prefix, including paths stopped before reaching the goal.
    prefix_error = value[0]-np.cumsum(cost)-value[1:]-np.cumsum(h)+np.cumsum(tau)
    assert np.max(np.abs(prefix_error)) < 1e-7
    return dict(steps=len(cost), cost=cost.sum(), start=value[0], end=value[-1],
                error_vs_path=value[0]-cost.sum(), sum_h=h.sum(), sum_tau=tau.sum(),
                h_over_cost=h.sum()/cost.sum(), tau_over_cost=tau.sum()/cost.sum(),
                tau_min=tau.min(), tau_first_half=tau[:len(cost)//2].sum(),
                prefix_identity_max=np.max(np.abs(prefix_error)))


def ranking(cost, value, ref):
    """Compare all strict reference pairs. Also detail chosen vs first reference winner."""
    c, j, v = (np.asarray(x, np.float64) for x in (cost, value, ref))
    assert len(c) and np.isfinite([c,j,v]).all()
    q, r = c+j, c+v
    i, k = np.triu_indices(len(q), 1)
    dr, dq = r[i]-r[k], q[i]-q[k]
    known = abs(dr)>TOL
    reverse = known & (np.sign(dr)*dq < -TOL)
    tied = known & (abs(dq)<=TOL)
    a, o = np.argmin(q), np.argmin(r)
    margin = r[a]-r[o]
    de = (j[a]-v[a])-(j[o]-v[o])
    assert abs(q[a]-q[o]-margin-de) < 1e-8
    return dict(candidates=len(q), strict_pairs=int(known.sum()), pairs=len(i),
                reversed_pairs=int(reverse.sum()), model_tied_strict=int(tied.sum()),
                inversion=reverse.sum()/known.sum() if known.any() else None,
                reference_margin=margin, chosen_error_difference=de, chosen_model_gap=q[a]-q[o],
                wrong=int(margin>TOL), ref_best_second=np.diff(np.sort(r)[:2])[0] if len(q)>1 else None,
                model_best_second=np.diff(np.sort(q)[:2])[0] if len(q)>1 else None)
