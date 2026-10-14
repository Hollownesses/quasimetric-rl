"""Read-only probes for the frozen-success-path B1 mechanism experiment.

Recorded suffix costs are computed here for reporting, never attached to replay
or passed to a loss. A path can be suboptimal, so an increasing V step is a
diagnostic observation, not by itself a violation of optimal-control theory.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch


def append_rows(path, rows):
    exists = path.exists()
    with path.open('a', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


class PathSupportDiagnostic:
    def __init__(self, raw, output_dir):
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.paths = []
        self.last_step = None
        # A fresh run replaces its own diagnostic tables, just as it replaces
        # train_metrics.csv. Checkpoint resume is rejected by this protocol.
        for name in ('values.csv', 'summary.csv'):
            (self.output / name).unlink(missing_ok=True)
        row, obs_start = 0, 0
        for length in raw.episode_lengths.tolist():
            if not raw.transition_infos['abstract_goal_edge'][row]:
                observations = raw.all_observations[obs_start:obs_start + length + 1].detach().cpu().clone()
                costs = -raw.rewards[row:row + length].detach().cpu().reshape(-1).clone()
                self.paths.append({
                    'observations': observations,
                    'goal': raw.transition_infos['task_goal_observations'][row].detach().cpu().clone(),
                    'costs': costs,
                    'suffix': np.r_[np.cumsum(costs.numpy()[::-1], dtype=np.float64)[::-1], 0.],
                    'context': int(raw.transition_infos['context_id'][row]),
                })
            row += length
            obs_start += length + 1
        if not self.paths:
            raise ValueError('Path diagnostic requires physical trajectories')

    @torch.no_grad()
    def snapshot(self, agent, step):
        if self.last_step == step:
            return
        if self.last_step is not None and step < self.last_step:
            raise ValueError('Path diagnostic steps must be increasing')
        modes = [(module, module.training) for module in agent.modules()]
        agent.eval()
        values_rows, summaries = [], []
        try:
            for critic_id, critic in enumerate(agent.critics):
                device = next(critic.parameters()).device
                for path_id, path in enumerate(self.paths):
                    obs = path['observations'].to(device)
                    goal = path['goal'].to(device).expand_as(obs)
                    values = critic(obs, goal).reshape(-1).cpu().numpy()
                    distances = critic(obs[:-1], obs[1:]).reshape(-1).cpu().numpy()
                    if not np.isfinite(values).all() or not np.isfinite(distances).all():
                        raise ValueError('Nonfinite path diagnostic prediction')
                    costs = path['costs'].numpy()
                    residual = distances - costs
                    delta = np.diff(values)
                    key = {'step': int(step), 'critic': critic_id, 'path': path_id, 'context': path['context']}
                    summaries.append({**key, 'physical_steps': len(costs),
                        'initial_value': float(values[0]), 'terminal_value': float(values[-1]),
                        'recorded_path_cost_upper_bound': float(path['suffix'][0]),
                        'value_increase_steps': int((delta > 1e-5).sum()),
                        'local_violation_count': int((residual > 1e-5).sum()),
                        'max_local_violation': float(np.maximum(residual, 0).max()),
                        'max_recorded_upper_bound_violation': float(np.maximum(values - path['suffix'], 0).max()),
                    })
                    for t, value in enumerate(values):
                        real = t < len(costs)
                        values_rows.append({**key, 'timestep': t, 'value': float(value),
                            'recorded_suffix_cost_upper_bound': float(path['suffix'][t]),
                            'next_value': float(values[t+1]) if real else None,
                            'value_delta': float(delta[t]) if real else None,
                            'physical_step_cost': float(costs[t]) if real else None,
                            'one_step_distance': float(distances[t]) if real else None,
                            'local_residual': float(residual[t]) if real else None,
                            'goal_inequality_residual': float(values[t]-costs[t]-values[t+1]) if real else None,
                        })
        finally:
            # Preserve every module's prior mode; probes do not train, consume
            # random samples, change the replay, or modify optimizer state.
            for module, training in modes:
                module.training = training
        append_rows(self.output / 'values.csv', values_rows)
        append_rows(self.output / 'summary.csv', summaries)
        self.last_step = step

    def write_report(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        summaries = list(csv.DictReader((self.output / 'summary.csv').open()))
        values = list(csv.DictReader((self.output / 'values.csv').open()))
        first, last = min(int(r['step']) for r in summaries), max(int(r['step']) for r in summaries)
        start = {(r['critic'], r['path']): r for r in summaries if int(r['step']) == first}
        final = [r for r in summaries if int(r['step']) == last]
        lines = ['# 仅成功路径的 B1 机制诊断', '',
                 f'记录 step={first} 至 {last}；只读逐路径探针，不增加训练 loss 或总成本标签。', '',
                 '记录路径的 suffix cost 是可行路径上界，不是连续环境的严格最优值。',
                 '沿记录路径的 V 上升次数仅作形状诊断，不应单独当成策略失败或最优性判定。', '',
                 '| critic | 路径 | 初始 V(start) | 最终 V(start) | 记录成本 | 最终 V(terminal) | 上升步数 | 局部违反数 |',
                 '|---|---|---:|---:|---:|---:|---:|---:|']
        for row in final:
            initial = start[(row['critic'], row['path'])]
            lines.append(f"| {row['critic']} | {row['path']} | {float(initial['initial_value']):.4f} | "
                         f"{float(row['initial_value']):.4f} | {float(row['recorded_path_cost_upper_bound']):.4f} | "
                         f"{float(row['terminal_value']):.4f} | {row['value_increase_steps']} | {row['local_violation_count']} |")
        lines += ['', '每一步的 V、ΔV、真实一步代价与局部残差见 [values.csv](values.csv)；',
                  '各次采样的汇总见 [summary.csv](summary.csv)。',
                  'push、barrier 及全 replay 审计随训练变化见 [train_metrics.csv](../train_metrics.csv)。',
                  '本诊断不测试保留起点泛化或执行成功率；通过后仍需独立评估。', '', '![逐路径估值](values.png)', '']
        (self.output / 'report.md').write_text('\n'.join(lines))
        critics = sorted({int(r['critic']) for r in final})
        fig, axes = plt.subplots(len(self.paths), len(critics), squeeze=False,
                                 figsize=(5 * len(critics), 2.7 * len(self.paths)))
        for p, path in enumerate(self.paths):
            for column, critic in enumerate(critics):
                ax = axes[p, column]
                for step, label in dict([(first, 'initial'), (last, 'final')]).items():
                    rows = [r for r in values if int(r['step']) == step and int(r['path']) == p and int(r['critic']) == critic]
                    ax.plot([int(r['timestep']) for r in rows], [float(r['value']) for r in rows], label=label)
                ax.plot(path['suffix'], '--', color='black', label='recorded cost upper bound')
                ax.set(title=f'Path {p}, critic {critic}', xlabel='Physical step', ylabel='Task value / cost')
                ax.legend(fontsize=7)
                ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(self.output / 'values.png', dpi=130)
        plt.close(fig)
