"""Read-only exact full-vector duplicate audit, independent of training RNG."""
import hashlib
import json
from pathlib import Path
import sys

from runtime import ROOT, prepare

prepare()
import numpy as np
from utils.env_utils import make_env_and_datasets


def duplicates(rows):
    rows = np.ascontiguousarray(rows)
    assert np.isfinite(rows).all()
    _, counts = np.unique(rows, axis=0, return_counts=True)
    n = len(rows)
    return dict(rows=n, unique_rows=len(counts), redundant_rows=int(n-len(counts)),
                redundant_fraction=float(1-len(counts)/n),
                rows_in_repeated_groups_fraction=float(counts[counts > 1].sum()/n),
                maximum_multiplicity=int(counts.max()))


def describe(ds, path):
    raw = np.load(path)
    ends = np.flatnonzero(raw['terminals'])
    lengths = np.diff(np.r_[-1, ends]) - 1
    valid = np.flatnonzero(ds['valids'])
    assert ends[-1] == len(raw['observations']) - 1
    assert int(lengths.sum()) == len(valid)
    assert np.array_equal(ds['valids'], 1-raw['terminals'])
    assert np.all(valid + 1 < ds.size)
    # Full transition includes all observation/action dimensions and terminal flag.
    transitions = np.concatenate([ds['observations'][valid], ds['actions'][valid],
                                 ds['observations'][valid+1], ds['terminals'][valid, None]], axis=1)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                shapes={k:list(v.shape) for k,v in ds.items()},
                dtypes={k:str(v.dtype) for k,v in ds.items()}, episodes=len(ends),
                valid_transitions=len(valid), invalid_final_rows=int((ds['valids']==0).sum()),
                episode_transition_lengths=dict(min=int(lengths.min()), max=int(lengths.max()),
                    mean=float(lengths.mean()), quantiles=np.quantile(lengths,[0,.25,.5,.75,1]).tolist()),
                full_observations_all_rows=duplicates(ds['observations']),
                full_observations_valid_sources=duplicates(ds['observations'][valid]),
                full_valid_transitions=duplicates(transitions))


if __name__ == '__main__':
    env, train, val = make_env_and_datasets('pointmaze-medium-navigate-v0')
    result = dict(definition='Exact numeric row equality, no rounding; transition=(full s, full a, full s_next, terminal). Redundant fraction=(N-unique)/N. No training modification.',
                  train=describe(train, ROOT/'data/pointmaze-medium-navigate-v0.npz'),
                  validation=describe(val, ROOT/'data/pointmaze-medium-navigate-v0-val.npz'),
                  tasks=env.unwrapped.task_infos)
    Path(sys.argv[1]).write_text(json.dumps(result, indent=2, default=lambda x: np.asarray(x).tolist())+'\n')
    env.close()
    print(json.dumps(result, indent=2, default=lambda x: np.asarray(x).tolist()))
