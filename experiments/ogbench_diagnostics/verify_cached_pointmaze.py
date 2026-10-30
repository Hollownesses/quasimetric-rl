"""Full fixed-bank equality after adding exact frozen-score memoization."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/ogbench_critic'))
from common import load_base, load_data, distances, sha256, tree_digest
from exact_score_cache import ExactScoreCache
sys.path.insert(0, str(ROOT))
from experiments.ogbench_diagnostics import core
from experiments.ogbench_diagnostics.fast_pointmaze import FastPointmaze
import numpy as np


def run(root, output):
    started = time.monotonic(); root = Path(root).resolve(); output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False); snapshot = output / 'source_snapshot'; snapshot.mkdir()
    files = [Path(__file__), Path(__file__).with_name('CACHED_SCORE_PROTOCOL.md'), Path(__file__).with_name('fast_pointmaze.py'),
             Path(__file__).with_name('core.py'), ROOT / 'experiments/ogbench_critic/exact_score_cache.py',
             ROOT / 'experiments/ogbench_critic/test_exact_score_cache.py']
    for path in files: shutil.copy2(path, snapshot / path.name)
    env = engine = None
    try:
        def budget():
            if time.monotonic() - started > 120: raise TimeoutError('120-second cached-score verification exhausted')
        previous = root / 'fast_pointmaze_parallel_verification'
        assert sha256(Path(__file__).with_name('fast_pointmaze.py')) == sha256(previous / 'source_snapshot/fast_pointmaze.py')
        candidates = sorted(previous.glob('*_candidates.npz')); assert len(candidates) == 35
        agent, config = load_base(); digest = tree_digest(agent); scorer = ExactScoreCache(agent, audit=True)
        env, _, _ = load_data(config); engine = FastPointmaze(env, nthread=4)
        cases = core.read_bank(root / 'evaluation_bank.npz'); assert len(cases) == 25
        candidate_checks = {}
        for path in candidates:
            budget()
            with np.load(path) as saved:
                np.testing.assert_array_equal(saved['original'], saved['fast'])
                obs = saved['fast']; goals = np.broadcast_to(np.array([4.2055,12.0898]), obs.shape)
                expected = np.asarray(distances(agent, obs, goals))
                np.testing.assert_array_equal(scorer(obs, goals), expected)
                np.testing.assert_array_equal(scorer(obs.copy(), goals.copy()), expected)
            candidate_checks[path.stem] = True
        original = json.loads((root / 'baseline_greedy/summary.json').read_text())['cases']; results = {}
        for name, case in cases.items():
            budget(); before = scorer.statistics()
            def score(observations, goals):
                budget(); return scorer(observations, goals)
            info, trace = engine.rollout(case, score)
            assert info['success'] == original[name]['success'] and info['steps'] == original[name]['steps']
            with np.load(root / f'baseline_greedy/{name}.npz') as old:
                for key in ['observations','actions','goal','distances']: np.testing.assert_array_equal(trace[key], old[key])
            info['new_function_evaluations'] = scorer.evaluations - before['evaluations']
            info['new_cache_hits'] = scorer.hits - before['hits']
            np.savez_compressed(output / f'{name}.npz', **trace); results[name] = info
            (output / 'progress.json').write_text(json.dumps(results, indent=2) + '\n'); print(name, info, flush=True)
        budget(); assert tree_digest(agent) == digest
        result = dict(passed=True, elapsed_seconds=time.monotonic() - started, candidates=candidate_checks,
            cases=results, cache=scorer.statistics(), threads=4,
            bank_sha256=sha256(root / 'evaluation_bank.npz'),
            source_sha256={p.name: sha256(p) for p in snapshot.iterdir()},
            original_episode_seconds=sum(v['seconds'] for v in original.values()),
            cached_episode_seconds=sum(v['seconds'] for v in results.values()))
        (output / 'verification.json').write_text(json.dumps(result, indent=2) + '\n'); print('PASSED', result['elapsed_seconds'], flush=True)
    except BaseException as error:
        (output / 'failure.json').write_text(json.dumps(dict(error=repr(error), elapsed_seconds=time.monotonic() - started,
            traceback=traceback.format_exc(), passed=False), indent=2) + '\n'); raise
    finally:
        if engine is not None: engine.close()
        if env is not None: env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('root'); parser.add_argument('--output', required=True)
    args = parser.parse_args(); run(args.root, args.output)
