"""Matched restart diagnostics; training never consumes graph references."""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.four_path_factorial.run import (
    IndexedIQE, TabularPHR, load_data, training_objective, evaluate,
    tensor_digest, file_digest,
)

START = ROOT/'results/four_path_factorial_20261016/main/encoder_phr_c0/latest.pt'


def restore(data, checkpoint=START):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    seed = saved.get('initialization_seed')
    model = IndexedIQE(data, 'encoder', 0, initialization_seed=seed)
    if seed is not None:
        data.fingerprint.pop('initial_checkpoint', None)  # legacy metadata, not used for fresh seeds
    model.load_state_dict(saved['model'])
    # The original c0 run used one group in model.parameters() order.
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.)
    optimizer.load_state_dict(saved['optimizer'])
    solver = TabularPHR(len(data.src), rho=100., dual_step=1.)
    solver.load_state_dict(saved['solver'])
    torch.set_rng_state(saved['rng'])
    return model, solver, optimizer, saved


def optimizer_digest(optimizer):
    return tensor_digest(*(v for state in optimizer.state.values()
                           for v in state.values() if torch.is_tensor(v)))


def learning_rate(arm, step, schedule_steps=5000):
    return 2e-5 if arm == 'A' else 2e-6 * .1**((step-1)/max(1, schedule_steps-1))


def advance(model, data, solver, optimizer, arm, step, schedule_steps=5000, inner=20):
    for group in optimizer.param_groups:
        group['lr'] = learning_rate(arm, step, schedule_steps)
    before = torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()
    optimizer.zero_grad(set_to_none=True)
    loss, old_residual, push = training_objective(model, data, solver, 'phr')
    if not torch.isfinite(loss):
        raise RuntimeError('nonfinite objective')
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        delta = torch.nn.utils.parameters_to_vector(model.parameters())-before
        magnitude = float(delta.norm())
        # C alone recomputes residuals AFTER the entire inner optimization block.
        updated = arm != 'C' or step % inner == 0
        if arm == 'C':
            solver.dual_step = float(inner)
            if updated:
                new_residual = model.distances(data)[0]-data.cost
                solver.update(new_residual)
        else:
            solver.update(old_residual.detach())
    return dict(step=step, lr=optimizer.param_groups[0]['lr'],
                parameter_update_l2=magnitude,
                parameter_update_relative=magnitude/float(before.norm()),
                parameter_update_linf=float(delta.abs().max()),
                dual_updated=updated, pre_primal_push=float(push.detach()),
                pre_primal_max_violation=float(old_residual.detach().relu().max()))


def diagnose(model, data, solver, step, elapsed):
    rec, values, arrays = evaluate(model, data, solver, step, elapsed)
    residual = torch.from_numpy(arrays['residual'])
    context = data.raw.transition_infos['context_id']
    entry = data.physical & torch.isin(data.dst, data.goal.unique())
    assert int(entry.sum()) == 4
    rec['terminal_entry_edges'] = [dict(row=int(i), path=int(context[i]),
        cost=float(data.cost[i]), distance=float(arrays['edge'][i]), residual=float(residual[i]))
        for i in entry.nonzero().flatten()]
    for p in rec['paths']:
        mask = data.physical & (context == p['path'])
        p['positive_residual_sum'] = float(residual[mask].relu().sum())
        p['max_violation'] = float(residual[mask].relu().max())
    rec['complementarity_max'] = float((solver.multiplier*residual).abs().max())
    rec['dual_stationarity_eta1_max'] = float(((solver.multiplier+residual).relu()-solver.multiplier).abs().max())
    rec['joint_pass'] = (rec['max_violation'] <= 1e-5 and rec['anchor_max'] <= 1e-6
                         and rec['terminal_max'] <= 1e-6 and rec['push_gap_percent'] <= 1.
                         and rec['task_nrmse_percent'] <= 2.)
    return rec, values, arrays


def run(args):
    torch.set_num_threads(2)
    data = load_data()
    model, solver, optimizer, saved = restore(data)
    assert saved['step'] == 20000
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    original_config = json.loads(START.with_name('config.json').read_text())
    assert data.fingerprint == original_config['fingerprints']
    config = dict(vars(args), checkpoint=str(START.relative_to(ROOT)), checkpoint_sha256=file_digest(START),
        fingerprints=data.fingerprint, rho=100., inner=20 if args.arm == 'C' else 1,
        dual_step=20. if args.arm == 'C' else 1.,
        dual_residual='post-inner' if args.arm == 'C' else 'pre-primal',
        initial_model_sha256=tensor_digest(*model.state_dict().values()),
        initial_dual_sha256=tensor_digest(*solver.state_dict().values()),
        initial_adam_sha256=optimizer_digest(optimizer), initial_adam_groups=optimizer.state_dict()['param_groups'],
        script_sha256=file_digest(__file__), torch=str(torch.__version__),
        push='unchanged full-support .6 task + .4 fixed B1',
        eval_tolerance=1e-5, push_gap_percent_tolerance=1., task_nrmse_percent_tolerance=2.)
    (out/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    curve = (out/'curve.jsonl').open('w'); updates = (out/'updates.jsonl').open('w')
    vf = (out/'values.csv').open('w'); writer = None
    started = time.perf_counter(); train_seconds = 0.; recent = []
    for step in range(args.steps+1):
        if step % args.eval_interval == 0 or step in (2000, 5000, args.steps):
            rec, values, arrays = diagnose(model, data, solver, step, time.perf_counter()-started)
            rec.update(train_seconds=train_seconds, total_primal_step=20000+step)
            if recent:
                rec['update_l2_mean'] = float(np.mean([x['parameter_update_l2'] for x in recent]))
                rec['update_l2_max'] = max(x['parameter_update_l2'] for x in recent)
                rec['lr'] = recent[-1]['lr']
            recent = []
            curve.write(json.dumps(rec)+'\n'); curve.flush(); updates.flush()
            if writer is None:
                writer = csv.DictWriter(vf, fieldnames=list(values[0])); writer.writeheader()
            writer.writerows(values); vf.flush()
            np.savez(out/'latest_edges.npz', **arrays)
            payload = dict(step=step,total_primal_step=20000+step,model=model.state_dict(),solver=solver.state_dict(),
                           optimizer=optimizer.state_dict(),rng=torch.get_rng_state())
            if step in (0,2000,5000,args.steps):
                torch.save(payload, out/f'checkpoint_{step:05d}.pt')
            torch.save(payload, out/'latest.pt')
            print(json.dumps({k:rec[k] for k in ('step','seconds','push','max_violation','violation_count','joint_pass')}), flush=True)
        if step == args.steps: break
        tick = time.perf_counter()
        update = advance(model,data,solver,optimizer,args.arm,step+1,args.schedule_steps)
        train_seconds += time.perf_counter()-tick
        updates.write(json.dumps(update)+'\n'); recent.append(update)
    curve.close(); updates.close(); vf.close()
    (out/'COMPLETE').write_text('completed\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['A','B','C'], required=True)
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--schedule-steps', type=int, default=5000)
    parser.add_argument('--eval-interval', type=int, default=25)
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
