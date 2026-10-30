"""Reject any candidate/trajectory difference before allowing the fast evaluator."""
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
sys.path.insert(0, str(ROOT))
from experiments.ogbench_diagnostics import core
from experiments.ogbench_diagnostics.fast_pointmaze import FastPointmaze
import numpy as np


def read(path): return json.loads(Path(path).read_text())


def run(root, output, threads=4):
    started = time.monotonic(); root = Path(root).resolve(); output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False); snapshot = output / 'source_snapshot'; snapshot.mkdir()
    for name in ['FAST_POINTMAZE_PROTOCOL.md', 'PARALLEL_POINTMAZE_PROTOCOL.md', 'fast_pointmaze.py', 'verify_fast_pointmaze.py', 'core.py']:
        shutil.copy2(Path(__file__).with_name(name), snapshot / name)
    env = engine = None
    try:
        def budget():
            if time.monotonic() - started > 120: raise TimeoutError('120-second equivalence verification exhausted')
        agent, config = load_base(); digest = tree_digest(agent)
        env, _, _ = load_data(config); bank = root / 'evaluation_bank.npz'; cases = core.read_bank(bank)
        assert len(cases) == 25
        original_candidates = core.candidates
        original_teleport = env.unwrapped._teleport_info
        env.unwrapped._teleport_info = {}
        try:
            try: FastPointmaze(env)
            except AssertionError: pass
            else: raise AssertionError('Unsupported teleport environment accepted')
        finally: env.unwrapped._teleport_info = original_teleport
        engine = FastPointmaze(env, nthread=threads)
        assert engine.bound_rollout.__code__ is core.rollout.__code__
        assert core.candidates is original_candidates
        probes = [(name, case) for name, case in cases.items()]
        for arm in ['mean', 'tail']:
            for name, case in cases.items():
                if not name.startswith('task3_'): continue
                with np.load(root / f'tail_1000/{arm}_greedy/{name}.npz') as trace:
                    probes.append((f'{arm}_{name}_end', {**case, 'observation': trace['observations'][-1], 'integration': trace['final_integration']}))
        comparisons = {}; timing = dict(original=0., fast=0.)
        for name, case in probes:
            budget(); core.load_case(env, case); initial = core.save_integration(env)
            elapsed = [w._elapsed_steps for w in core.wrappers(env)]
            now = time.monotonic(); old, _ = core.candidates(env); timing['original'] += time.monotonic() - now
            now = time.monotonic(); new, _ = engine.candidates(env); timing['fast'] += time.monotonic() - now
            np.savez_compressed(output / f'{name}_candidates.npz', original=old, fast=new)
            np.testing.assert_array_equal(old, new)
            np.testing.assert_array_equal(initial, core.save_integration(env))
            assert [w._elapsed_steps for w in core.wrappers(env)] == elapsed
            comparisons[name] = dict(exact=True, candidates=len(new), max_abs_error=float(np.max(np.abs(old - new))))
        assert len(comparisons) == 35
        rollouts = {}; baseline = read(root / 'baseline_greedy/summary.json')['cases']
        for name, case in cases.items():
            budget()
            def score(observations, goals):
                budget(); return np.asarray(distances(agent, observations, goals))
            info, trace = engine.rollout(case, score)
            assert info['success'] == baseline[name]['success'] and info['steps'] == baseline[name]['steps']
            with np.load(root / f'baseline_greedy/{name}.npz') as old:
                for key in ['observations', 'actions', 'goal', 'distances']: np.testing.assert_array_equal(trace[key], old[key])
            np.savez_compressed(output / f'{name}_rollout.npz', **trace); rollouts[name] = info
            (output / 'progress.json').write_text(json.dumps(rollouts, indent=2) + '\n')
            print(name, info, flush=True)
        budget(); assert tree_digest(agent) == digest and core.candidates is original_candidates
        result = dict(passed=True, threads=threads, checkpoint_sha256=sha256(ROOT / 'results/ogbench_official_qrl_20261026_seed0/main/OGBench/navigate_100k/sd000_20261026_115628/params_100000.pkl'),
            bank_sha256=sha256(bank), source_sha256={p.name: sha256(p) for p in snapshot.iterdir()},
            candidates=comparisons, candidate_seconds=timing, cases=rollouts, teleport_rejected=True,
            unchanged_original_module=True, exact_episode_code=True, elapsed_seconds=time.monotonic() - started,
            original_episode_seconds=sum(v['seconds'] for v in baseline.values()),
            fast_episode_seconds=sum(v['seconds'] for v in rollouts.values()))
        (output / 'verification.json').write_text(json.dumps(result, indent=2) + '\n'); print('PASSED', result['elapsed_seconds'], flush=True)
    except BaseException as error:
        (output / 'failure.json').write_text(json.dumps(dict(error=repr(error), elapsed_seconds=time.monotonic() - started,
            traceback=traceback.format_exc(), passed=False), indent=2) + '\n'); raise
    finally:
        if engine is not None: engine.close()
        if env is not None: env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('root'); parser.add_argument('--output', required=True)
    parser.add_argument('--threads', type=int, choices=[0, 4], default=4)
    args = parser.parse_args(); run(args.root, args.output, args.threads)
