"""Evaluation-only graph references; never imported by the learner."""
import argparse
import json
from pathlib import Path
import numpy as np
from experiments.expert_utrap_network.audit import reverse_distances
from experiments.four_path_factorial.run import file_digest


def task_goals(data):
    # goal is indexed by all replay rows, but only physical entries are populated.
    goals = data.goal[data.original].unique().numpy()
    if len(goals) != 1 or not bool((data.nodes[goals, 41] > .5).all()):
        raise ValueError('Expected one canonical legal task goal')
    return goals


def validate_reference(data, distance):
    goals = task_goals(data)
    if distance.shape != (len(data.nodes),) or not np.isfinite(distance).all():
        raise ValueError('Invalid graph distances')
    if not np.array_equal(np.flatnonzero(distance == 0), np.sort(goals)):
        raise ValueError('Zero-distance nodes must be exactly the task goals')
    rows = data.original.numpy()
    src, dst, cost = data.src.numpy(), data.dst.numpy(), data.cost.double().numpy()
    minimum = np.full(len(distance), np.inf)
    np.minimum.at(minimum, src[rows], cost[rows]+distance[dst[rows]])
    mask = np.ones(len(distance), bool); mask[goals] = False
    if not np.allclose(minimum[mask], distance[mask], atol=1e-8, rtol=0):
        raise ValueError('Nonterminal reference fails Bellman minimum equality')
    return goals


def recompute(root, output):
    from .data import load_training_data
    root, output = Path(root), Path(output)
    data = load_training_data(root)
    before = file_digest(root/'verified_replay.pt')
    goal = task_goals(data)
    J, next_edge = reverse_distances(len(data.nodes), data.src.numpy(), data.dst.numpy(),
                                    data.cost.numpy(), goal)
    validate_reference(data, J)
    old = np.load(root/'audit/graph_reference_audit_only.npz')['distance']
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/'graph_reference_audit_only.npz', distance=J, next_edge=next_edge,
                        source=data.src.numpy(), target=data.dst.numpy(), physical=data.physical.numpy())
    changed = np.flatnonzero(abs(J-old)>1e-8)
    summary = dict(fingerprints=data.fingerprint, goals=goal.tolist(), old_zero_nodes=np.flatnonzero(old==0).tolist(),
        changed=[dict(node=int(i), old=float(old[i]), corrected=float(J[i])) for i in changed],
        replay_sha256=before, old_reference_sha256=file_digest(root/'audit/graph_reference_audit_only.npz'),
        reference_sha256=file_digest(output/'graph_reference_audit_only.npz'), passed=True)
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    assert file_digest(root/'verified_replay.pt') == before
    (output/'report.md').write_text('# 连通专家图参考修正\n\n'
        '仅修正事后评估，不重新建图、不训练模型、不改写旧参考。训练目标仍由物理行索引读取，未受此错误影响。\n\n'
        '`goal` 的非物理行默认值 0 被旧审计误当作目标。现在只从物理行提取目标，并检查零距离节点与合法目标一致、每个非终态满足 Bellman 最小值等式。\n\n'
        f'合法目标：{goal.tolist()}。变化节点数：{len(changed)}。\n\n'
        '| 节点 | 原参考 | 修正参考 |\n|---|---:|---:|\n'+''.join(f'| {i} | {old[i]:.9f} | {J[i]:.9f} |\n' for i in changed)+
        '\n原始 replay、状态编号、动作、成本、B1 与 checkpoint 均保留。新参考供后续统一评估；该修正不改变模型输出。\n\n'
        '复现：`python -m experiments.connected_expert_utrap.reference --root results/connected_expert_utrap_20261021/main --output results/NEW_REFERENCE`。输出目录必须不存在。\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();recompute(a.root,a.output)
