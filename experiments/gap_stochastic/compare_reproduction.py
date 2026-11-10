"""Compare every saved array of a same-stream replay, without selecting runs."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('original', type=Path)
    parser.add_argument('replay', type=Path)
    args = parser.parse_args()
    a = json.loads((args.original/'manifest.json').read_text())
    b = json.loads((args.replay/'manifest.json').read_text())
    assert a['config'] == b['config'] and set(a['runs']) == set(b['runs'])
    result = dict(all_equal=True, runs={}, same_seed_replay_not_an_independent_replication=True)
    for key in a['runs']:
        xa = np.load(args.original/(key+'.npz'))
        xb = np.load(args.replay/(key+'.npz'))
        equal = {field:bool(np.array_equal(xa[field],xb[field])) for field in xa.files}
        assert set(xa.files) == set(xb.files) and all(equal.values())
        assert a['runs'][key]['sampling_sha256'] == b['runs'][key]['sampling_sha256']
        result['runs'][key] = equal
    result['replay_pipeline'] = json.loads((args.replay/'pipeline.json').read_text())
    (args.original/'reproduction_check.json').write_text(json.dumps(result,indent=2)+'\n')
    (args.original/'replay_manifest.json').write_text(json.dumps(b,indent=2)+'\n')
    print(json.dumps(dict(all_equal=True, runs=len(result['runs']), replay_seconds=result['replay_pipeline']['seconds'])))


if __name__ == '__main__':
    main()
