"""Fixed 1000-update mean/tail continuation; no reseeding or budget extension."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

from common import ROOT, BASE, BASE_SHA, load_base, load_data, tree_digest, tree_equal, sha256, distances
from critic import init_optimizer
from tail_critic import update, statistics
from train import freeze_proof
sys.path.insert(0, str(ROOT))
from experiments.ogbench_diagnostics import core
import jax
import numpy as np
from utils.flax_utils import save_agent, restore_agent

ARMS = ('mean', 'tail')


def budget(deadline):
    if time.monotonic() > deadline:
        raise TimeoutError('600-second fixed tail experiment exhausted; no automatic continuation')


def numbers(info): return {k: float(v) for k, v in info.items()}


def run(root, output, smoke=False):
    started = time.monotonic(); deadline = started + 600
    root = Path(root).resolve(); output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    steps = 2 if smoke else 1000; steps_done = {arm: 0 for arm in ARMS}
    snapshot = output / 'source_snapshot'; snapshot.mkdir()
    files = ['TAIL_PROTOCOL.md', 'tail_critic.py', 'test_tail_critic.py', 'run_tail.py', 'common.py', 'critic.py', 'train.py']
    for name in files: shutil.copy2(Path(__file__).with_name(name), snapshot / name)
    shutil.copy2(core.__file__, snapshot / 'diagnostic_core.py')
    env = None
    try:
        base, config = load_base(); env, train, val = load_data(config); frozen = freeze_proof(base)
        initial = tree_digest(base)
        np.random.seed(9001); validation = val.sample(2048)
        np.random.seed(2001); agents = {a: base for a in ARMS}; optimizers = {a: init_optimizer(base) for a in ARMS}
        protocol = dict(steps=steps, batch_size=256, tail_fraction=.1, tail_count=26,
            sampling_seed=2001, initialization='same original seed0 100k checkpoint; not a new independent seed',
            base_checkpoint=str(BASE / 'params_100000.pkl'), base_sha256=BASE_SHA,
            original_config=config, validation_sha256=tree_digest(validation),
            frozen=frozen, smoke=smoke, time_limit_seconds=600,
            source_sha256={p.name: sha256(p) for p in snapshot.iterdir()},
            devices=[str(d) for d in jax.devices()], jax=jax.__version__)
        (output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
        (output / 'initial_statistics.json').write_text(json.dumps(numbers(statistics(base, validation)), indent=2) + '\n')
        stream = hashlib.sha256()
        with (output / 'metrics.csv').open('w') as handle:
            writer = None
            for step in range(1, steps + 1):
                budget(deadline); batch = train.sample(256)
                for key in ['observations', 'next_observations', 'value_goals']: stream.update(batch[key].tobytes())
                info = {}
                for arm in ARMS:
                    budget(deadline)
                    agents[arm], optimizers[arm], info[arm] = update(agents[arm], optimizers[arm], batch, arm)
                    jax.block_until_ready(info[arm]['total_loss']); steps_done[arm] = step
                if step == 1 or step % 100 == 0 or step == steps:
                    for arm in ARMS:
                        row = dict(step=step, arm=arm, elapsed_seconds=time.monotonic() - started)
                        keys = ['total_loss', 'value_loss', 'lam_loss', 'd_neg_loss', 'd_pos_loss', 'lam']
                        row.update({f'train/{k}': float(info[arm][k]) for k in keys})
                        row.update({f'heldout/{k}': v for k, v in numbers(statistics(agents[arm], validation)).items()})
                        assert all(np.isfinite(v) for k, v in row.items() if k != 'arm')
                        if writer is None: writer = csv.DictWriter(handle, fieldnames=list(row)); writer.writeheader()
                        writer.writerow(row); handle.flush(); print(json.dumps(row), flush=True)
        verification = {}; epoch = 100000 + steps
        cases = core.read_bank(root / 'evaluation_bank.npz')
        obs = np.array([c['observation'] for c in cases.values()]); goals = np.array([c['goal'] for c in cases.values()])
        actor_reference = base.sample_actions(obs, goals, seed=jax.random.PRNGKey(123), temperature=0)
        for arm, agent in agents.items():
            budget(deadline); directory = output / arm; directory.mkdir()
            assert freeze_proof(agent) == frozen
            assert int(agent.network.opt_state[0].count) == epoch and int(agent.network.step) == epoch + 1
            assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(agent))
            assert tree_equal(actor_reference, agent.sample_actions(obs, goals, seed=jax.random.PRNGKey(123), temperature=0))
            save_agent(agent, str(directory), epoch); restored = restore_agent(base, str(directory), epoch)
            assert tree_equal(agent, restored), 'Full saved state roundtrip differs'
            assert tree_digest(base) == initial and sha256(BASE / 'params_100000.pkl') == BASE_SHA
            verification[arm] = dict(passed=True, actual_updates=steps_done[arm], adam_count=epoch,
                checkpoint_sha256=sha256(directory / f'params_{epoch}.pkl'), full_state_roundtrip=True,
                frozen_parameters_and_moments=True, frozen_bank_actor_outputs=True,
                critic_digest=tree_digest(agent.network.params['modules_value']))
        complete = dict(steps=steps, verification=verification, paired_stream_sha256=stream.hexdigest(),
                        train_and_verify_seconds=time.monotonic() - started)
        (output / 'training_complete.json').write_text(json.dumps(complete, indent=2) + '\n')
        if smoke:
            budget(deadline); print('SMOKE_COMPLETE', complete, flush=True); return
        # Formal evaluation only after both fixed 1000-update checkpoints verified.
        rollouts = {}; fields = {}
        for arm, agent in agents.items():
            directory = output / f'{arm}_greedy'; directory.mkdir(); result = {}
            for name, case in cases.items():
                budget(deadline)
                def score(observations, goals):
                    budget(deadline); return np.asarray(distances(agent, observations, goals))
                info, trace = core.rollout(env, case, score=score)
                trace['final_integration'] = core.save_integration(env)
                np.savez_compressed(directory / f'{name}.npz', **trace); result[name] = info
                (directory / 'progress.json').write_text(json.dumps(result, indent=2) + '\n')
                print(arm, name, info, flush=True)
            (directory / 'summary.json').write_text(json.dumps(result, indent=2) + '\n'); rollouts[arm] = result
        budget(deadline); walls, radius = core.geometry(env); nodes, graph = core.build_graph(walls, radius, .2)
        with np.load(root / 'baseline_field/field.npz') as reference: assert np.array_equal(nodes, reference['nodes'])
        for arm, agent in agents.items():
            budget(deadline); values = np.asarray(distances(agent, nodes, np.broadcast_to(core.GOAL, nodes.shape)))
            stats, field = core.graph_metrics(nodes, graph, values); fields[arm] = stats
            np.savez_compressed(output / f'{arm}_field.npz', **{**field, 'walls': walls})
        budget(deadline)
        summary = dict(protocol=protocol, training=complete, rollouts=rollouts, fields=fields,
                       bank_sha256=sha256(root / 'evaluation_bank.npz'), complete=True, elapsed_seconds=time.monotonic() - started)
        (output / 'analysis.json').write_text(json.dumps(summary, indent=2) + '\n')
        print('COMPLETE', summary['elapsed_seconds'], flush=True)
    except BaseException as error:
        (output / 'failure.json').write_text(json.dumps(dict(error=repr(error), steps_done=steps_done,
            elapsed_seconds=time.monotonic() - started, complete=False, traceback=traceback.format_exc()), indent=2) + '\n')
        raise
    finally:
        if env is not None: env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('root'); parser.add_argument('--output', required=True)
    parser.add_argument('--smoke', action='store_true'); args = parser.parse_args(); run(args.root, args.output, args.smoke)
