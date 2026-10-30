"""Bounded data-only graph correction of a fixed QRL critic; evaluation only."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

from common import load_base, load_data, distances, tree_digest, sha256, BASE, ROOT
from data_graph import DataGraph, valid_transitions
sys.path.insert(0, str(ROOT))
from experiments.ogbench_diagnostics import core
import numpy as np


def budget(deadline):
    if time.monotonic() > deadline:
        raise TimeoutError('360-second data-graph diagnostic exhausted; do not extend automatically')


def run(root, output):
    started = time.monotonic(); deadline = started + 360
    root = Path(root).resolve(); output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    snapshot = output / 'source_snapshot'; snapshot.mkdir()
    for name in ['DATA_GRAPH_PROTOCOL.md', 'data_graph.py', 'run_data_graph.py', 'test_data_graph.py', 'common.py']:
        shutil.copy2(Path(__file__).with_name(name), snapshot / name)
    agent, config = load_base(); frozen = tree_digest(agent.network.params)
    env, train, _ = load_data(config)
    source, target = valid_transitions(train.dataset)
    graph = DataGraph.build(source, target); graph.save(output / 'graph.npz')
    restored = DataGraph.load(output / 'graph.npz')
    assert np.array_equal(graph.nodes, restored.nodes) and np.array_equal(graph.source, restored.source)
    assert np.array_equal(graph.target, restored.target)
    metadata = dict(training_updates=0, base_checkpoint_sha256=sha256(BASE / 'params_100000.pkl'),
        graph_sha256=sha256(output / 'graph.npz'), source_transitions=len(source), nodes=len(graph.nodes),
        edges=graph.graph.nnz, spacing=graph.spacing, weight=.5, time_limit_seconds=360,
        graph_build_seconds=time.monotonic() - started,
        graph_inputs='official training valid transitions only; no evaluation bank or geometry',
        source_sha256={p.name: sha256(p) for p in snapshot.iterdir()})
    (output / 'protocol.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print('GRAPH', metadata, flush=True)
    budget(deadline)
    # The graph was constructed and serialized before accessing evaluation cases.
    bank = root / 'evaluation_bank.npz'; cases = core.read_bank(bank)
    assert len(cases) == 25; metadata['bank_sha256'] = sha256(bank)
    results = {}
    for mode in ['hybrid', 'graph']:
        directory = output / mode; directory.mkdir(); result = {}
        for name, case in cases.items():
            if mode == 'graph' and not name.startswith('task3_'): continue
            budget(deadline); goal = case['goal']
            def score(observations, goals):
                budget(deadline)
                assert np.array_equal(goals, np.broadcast_to(goal, goals.shape))
                table = graph.distances(observations, goal)
                if mode == 'graph': return table
                return .5 * table + .5 * np.asarray(distances(agent, observations, goals))
            info, trace = core.rollout(env, case, score=score)
            trace['final_integration'] = core.save_integration(env)
            np.savez_compressed(directory / f'{name}.npz', **trace); result[name] = info
            (directory / 'progress.json').write_text(json.dumps(result, indent=2) + '\n')
            print(mode, name, info, flush=True)
        (directory / 'summary.json').write_text(json.dumps(result, indent=2) + '\n'); results[mode] = result
    # Geometry is loaded only now for evaluation and never passed to the builder.
    with np.load(root / 'baseline_field/field.npz') as reference:
        walls = reference['walls'].copy(); old_nodes = reference['nodes'].copy()
    radius = core.geometry(env)[1]; nodes, reference_graph = core.build_graph(walls, radius, .2)
    assert np.array_equal(nodes, old_nodes)
    fields = {}; qrl = np.asarray(distances(agent, nodes, np.broadcast_to(core.GOAL, nodes.shape)))
    table = graph.distances(nodes, core.GOAL)
    for mode, value in [('graph', table), ('hybrid', .5 * table + .5 * qrl)]:
        budget(deadline)
        stats, field = core.graph_metrics(nodes, reference_graph, value)
        np.savez_compressed(output / f'{mode}_field.npz', **{**field, 'walls': walls})
        fields[mode] = stats
    env.close(); budget(deadline)
    assert tree_digest(agent.network.params) == frozen
    assert sha256(BASE / 'params_100000.pkl') == metadata['base_checkpoint_sha256']
    result = dict(protocol=metadata, results=results, fields=fields, complete=True,
                  elapsed_seconds=time.monotonic() - started)
    (output / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    print('COMPLETE', result['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('root'); parser.add_argument('--output', required=True)
    args = parser.parse_args(); run(args.root, args.output)
