"""Independent checkpoint audit and publication-style diagnostics."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.four_path_refinement.run import (
    START, load_data, restore, diagnose, file_digest, tensor_digest, optimizer_digest,
)


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def audit(root):
    torch.set_num_threads(2)
    data = load_data(); result = {}
    configs = [json.loads((root/'main'/a/'config.json').read_text()) for a in 'ABC']
    for key in ('initial_model_sha256','initial_dual_sha256','initial_adam_sha256','checkpoint_sha256','fingerprints'):
        assert all(c[key] == configs[0][key] for c in configs)
    assert configs[0]['checkpoint_sha256'] == file_digest(START)
    for a in 'ABC':
        directory = root/'main'/a
        result[a] = {}
        curve = records(directory/'curve.jsonl')
        for step in (0,2000,5000):
            model, solver, opt, saved = restore(data,directory/f'checkpoint_{step:05d}.pt')
            record, _, arrays = diagnose(model,data,solver,step,0.)
            logged = next(r for r in curve if r['step'] == step)
            for key in ('push','max_violation','task_nrmse_percent','terminal_max','anchor_max','complementarity'):
                assert abs(record[key]-logged[key]) < 2e-5, (a,step,key,record[key],logged[key])
            assert record['violation_count'] == logged['violation_count']
            if step == 5000:
                stored_arrays=np.load(directory/'latest_edges.npz')
                for key in arrays:
                    np.testing.assert_allclose(arrays[key],stored_arrays[key],rtol=0,atol=2e-5)
            assert all(int(state['step']) == 20000+step for state in opt.state.values())
            if step == 0:
                assert tensor_digest(*model.state_dict().values()) == configs[0]['initial_model_sha256']
                assert optimizer_digest(opt) == configs[0]['initial_adam_sha256']
            result[a][str(step)] = record
        updates = records(directory/'updates.jsonl')
        assert len(updates) == 5000
        assert sum(u['dual_updated'] for u in updates) == (250 if a == 'C' else 5000)
        assert [u['step'] for u in updates] == list(range(1,5001))
        result[a]['update_count'] = len(updates)
        result[a]['dual_update_count'] = sum(u['dual_updated'] for u in updates)
    for name in ('repair','repair_push'):
        repair = root/name
        if not (repair/'COMPLETE').exists(): continue
        cfg = json.loads((repair/'config.json').read_text())
        model,solver,_,saved = restore(data,repair/'latest.pt')
        rec,_,arrays = diagnose(model,data,solver,saved['step'],0.)
        stored_arrays=np.load(repair/'latest_edges.npz')
        for key in arrays:
            np.testing.assert_allclose(arrays[key],stored_arrays[key],rtol=0,atol=2e-5)
        assert rec['push'] >= cfg['fixed_push_floor']
        for item in records(repair/'qp.jsonl'):
            for candidate in item['attempts']:
                if candidate['accepted']: assert candidate['push'] >= cfg['fixed_push_floor']
        logged = records(repair/'curve.jsonl')[-1]
        for key in ('push','max_violation','task_nrmse_percent','terminal_max','anchor_max'):
            assert abs(rec[key]-logged[key]) < 2e-5
        result[name] = rec
    return result


def summarize(root):
    out = root/'comparison';out.mkdir(exist_ok=False)
    audited = audit(root)
    (out/'independent_audit.json').write_text(json.dumps(audited,indent=2)+'\n')
    curves = {a:records(root/'main'/a/'curve.jsonl') for a in 'ABC'}
    updates = {a:records(root/'main'/a/'updates.jsonl') for a in 'ABC'}
    colors = dict(A='#cf5b46',B='#2675ba',C='#299866')
    metrics = [('push','Same-support push'),('task_nrmse_percent','Task NRMSE (%)'),
               ('max_violation','Max physical violation'),('violation_count','Violated physical edges /335'),
               ('feasible_scale','Conservative feasible scale'),('complementarity','Mean |lambda * h|'),
               ('complementarity_max','Max |lambda * h|'),('update_l2_mean','Actual parameter update L2')]
    fig,axes = plt.subplots(4,2,figsize=(12,14))
    for ax,(key,label) in zip(axes.flat,metrics):
        for a,curve in curves.items():
            filtered = [r for r in curve if key in r]
            ax.plot([r['step'] for r in filtered],[max(r[key],1e-9) for r in filtered],label=a,color=colors[a],linewidth=1)
        if key in ('max_violation','complementarity','complementarity_max','update_l2_mean'): ax.set_yscale('log')
        if key == 'max_violation': ax.axhline(1e-5,color='black',linestyle=':',label='tolerance')
        if key == 'task_nrmse_percent': ax.axhline(2,color='black',linestyle=':')
        if key == 'push': ax.axhline(curves['A'][0]['push_reference']*.99,color='black',linestyle=':',label='99% graph push')
        ax.set(xlabel='Additional primal updates',ylabel=label);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'curves.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(4,3,figsize=(14,12))
    for p in range(4):
        for a,curve in curves.items():
            x=[r['step'] for r in curve]
            axes[p,0].plot(x,[r['terminal_entry_edges'][p]['residual'] for r in curve],color=colors[a],label=a)
            axes[p,1].plot(x,[r['paths'][p]['positive_residual_sum'] for r in curve],color=colors[a])
            axes[p,2].plot(x,[r['paths'][p]['triangle_slack'] for r in curve],color=colors[a])
        axes[p,0].axhline(0,color='black',linestyle=':');axes[p,0].legend(fontsize=8)
        for ax,title in zip(axes[p],['Terminal-entry residual','Sum positive edge residuals','Triangle slack']):
            ax.set(title=f'Path {p}: {title}',xlabel='Additional primal updates');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'path_diagnostics.png',dpi=160);plt.close(fig)
    summary = {}
    for a,curve in curves.items():
        window=[r for r in curve if r['step'] >= 4000]
        u=[r for r in updates[a] if r['step'] > 4000]
        summary[a] = dict(final=curve[-1],at_2000=next(r for r in curve if r['step']==2000),
            last_1000=dict(max_violation_min=min(r['max_violation'] for r in window),
                max_violation_max=max(r['max_violation'] for r in window),
                max_violation_std=float(np.std([r['max_violation'] for r in window])),
                push_min=min(r['push'] for r in window),push_max=max(r['push'] for r in window),
                parameter_update_l2_mean=float(np.mean([x['parameter_update_l2'] for x in u])),
                parameter_update_l2_max=max(x['parameter_update_l2'] for x in u)),
            any_evaluated_joint_pass=any(r['joint_pass'] for r in curve))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    header=['arm','step','push','push_gap_percent','task_nrmse_percent','max_violation','violation_count','complementarity','seconds','train_seconds','joint_pass']
    with (out/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=header);writer.writeheader()
        for a in 'ABC':
            for key in ('at_2000','final'):
                writer.writerow(dict(arm=a,**{k:summary[a][key][k] for k in header[1:]}))
    fig,axes=plt.subplots(1,3,figsize=(13,3.7))
    for name in ('repair','repair_push'):
        if not (root/name/'curve.jsonl').exists(): continue
        repair=records(root/name/'curve.jsonl')
        for ax,key,title in zip(axes,['max_violation','push','task_nrmse_percent'],['Max physical violation','Push with fixed floor','Task NRMSE (%)']):
            measured=[r for r in repair if key in r]
            ax.plot([r['step'] for r in measured],[r[key] for r in measured],marker='o',label=name);ax.set(xlabel='Local QP iterations',title=title);ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[0].set_yscale('log');axes[0].axhline(1e-5,linestyle=':',color='black')
    axes[1].axhline(repair[0]['push_floor'],linestyle=':',color='black')
    fig.tight_layout();fig.savefig(out/'repair.png',dpi=160);plt.close(fig)
    print(json.dumps({a:{k:summary[a]['final'][k] for k in ('push','max_violation','joint_pass')} for a in 'ABC'},indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='results/four_path_refinement_20261016')
    summarize(Path(p.parse_args().root))
