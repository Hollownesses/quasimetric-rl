"""Post-hoc single-step probes at saved failed endpoints; never run new episodes."""
import argparse
import json
from pathlib import Path
import time

from common import ROOT, load_base, load_data, distances, sha256
import sys
sys.path.insert(0, str(ROOT))
from data_graph import DataGraph
from experiments.ogbench_diagnostics import core
import numpy as np


def run(root, probe_output):
    started = time.monotonic(); root = Path(root).resolve(); output = root / 'data_graph_critic'
    result = json.loads((output / 'analysis.json').read_text())
    assert result['complete'] and result['elapsed_seconds'] < 360
    assert sha256(output / 'graph.npz') == result['protocol']['graph_sha256']
    directory = Path(probe_output).resolve(); directory.mkdir(exist_ok=False)
    agent, config = load_base(); env, _, _ = load_data(config)
    graph = DataGraph.load(output / 'graph.npz'); bank = core.read_bank(root / 'evaluation_bank.npz'); probes = {}
    for mode in ['hybrid', 'graph']:
        for name, info in result['results'][mode].items():
            if not name.startswith('task3_') or info['success']: continue
            assert time.monotonic() - started < 30, 'Post-hoc probe budget exhausted; no continuation'
            with np.load(output / mode / f'{name}.npz') as trace:
                case = {**bank[name], 'observation': trace['observations'][-1], 'integration': trace['final_integration']}
            core.load_case(env, case); successors, _ = core.candidates(env)
            dg = graph.distances(successors, case['goal']); dq = np.asarray(distances(agent, successors, np.broadcast_to(case['goal'], successors.shape)))
            dh = .5 * dg + .5 * dq
            current = float(graph.distances(case['observation'][None], case['goal'])[0])
            node = int(graph.indices(case['observation'][None])[0]); start, end = graph.graph.indptr[node:node + 2]
            neighbors = graph.graph.indices[start:end]; values = graph.goal_distances(case['goal'])
            lower = neighbors[values[neighbors] < values[node]]
            if 0 < current < len(graph.nodes): assert len(lower)
            row = dict(current_graph_distance=current, minimum_successor_graph_distance=float(dg.min()),
                       successors_with_lower_graph_distance=int((dg < current).sum()),
                       graph_unreachable=bool(current == len(graph.nodes)),
                       graph_unreachable_nodes=int((values == len(graph.nodes)).sum()),
                       mapped_node=graph.nodes[node].tolist(), lower_graph_neighbors=graph.nodes[lower].tolist(),
                       lower_graph_neighbor_distances=values[lower].tolist(),
                       max_axis_displacement_to_lower_nodes=np.abs(graph.nodes[lower] - case['observation']).max(axis=1).tolist())
            index = int(np.argmin(dh if mode == 'hybrid' else dg))
            row.update(selected_action=core.ACTIONS[index].tolist(), selected_successor=successors[index].tolist(),
                       selected_displacement=float(np.max(np.abs(successors[index] - case['observation']))))
            np.savez_compressed(directory / f'{mode}_{name}.npz', observations=successors,
                                actions=core.ACTIONS, graph_distance=dg, qrl_distance=dq, hybrid_distance=dh)
            probes[f'{mode}_{name}'] = row
            print(mode, name, row, flush=True)
    env.close()
    (directory / 'analysis.json').write_text(json.dumps(dict(probes=probes, new_episodes=0,
        post_hoc=True, seconds=time.monotonic() - started, source_sha256=sha256(__file__)), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('root'); parser.add_argument('--output', required=True)
    args = parser.parse_args(); run(args.root, args.output)
