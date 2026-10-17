"""Post-run Oracle cache equivalence and first-decision failure diagnostics."""
import argparse
import json
from pathlib import Path
import numpy as np
from experiments.four_path_execution.common import load_data,read,write,file_digest,torch
from experiments.four_path_oracle_execution.protocol import CONFIG,controller_factory
from minimal_qrl.baselines.mppi import simulate_action_sequences


def audit(root,historical_cache):
    out=root/'oracle_audit';out.mkdir(exist_ok=False);torch.set_num_threads(1)
    current=next((root/'oracle_cache').glob('*.npz'))
    old=np.load(historical_cache);new=np.load(current);a=old['values'];b=new['values']
    finite=np.isfinite(a)&np.isfinite(b)
    comparison=dict(historical_cache=str(historical_cache),historical_sha256=file_digest(historical_cache),
        current_cache=str(current),current_sha256=file_digest(current),
        signature_equal=bool(old['digest']==new['digest']),values_bitwise_equal=bool(np.array_equal(a,b)),
        finite_masks_equal=bool(np.array_equal(np.isfinite(a),np.isfinite(b))),
        maximum_finite_absolute_difference=float(abs(a[finite]-b[finite]).max()))
    write(out/'historical_cache_comparison.json',comparison)
    data=load_data();env=data.env;make=controller_factory(root)
    rows=[json.loads(x) for x in (root/'arms/oracle/episodes.jsonl').read_text().splitlines()]
    failures=[]
    for row in rows:
        if row['success']:continue
        env.reset(seed=row['planner_seed'],options={'device_id':'u_trap_target','start':row['pose']})
        c=make(data,'oracle');c.begin_episode(env,env.abstract_goal_observation(),row['planner_seed'])
        actions=c.sample_candidates(env);roll=simulate_action_sequences(env,env.state,actions,record_trajectory=True)
        costs,weights=c.score_candidates(env,roll);first=float(np.sum(weights[:,None]*actions,axis=0).astype(np.float32)[0])
        stored=np.load(root/'arms/oracle'/f"{row['state_id']}_{row['planner_seed']}.npz")
        assert first==float(stored['actions'][0,0])
        one=simulate_action_sequences(env,env.state,actions[:,:1])
        # Fixed legal control only demonstrates that initial collision is avoidable;
        # this is post hoc diagnostics, never a new candidate in the experiment.
        probe=simulate_action_sequences(env,env.state,np.full((1,2),-env.omega_max,np.float32))
        failures.append(dict(state_id=row['state_id'],planner_seed=row['planner_seed'],
            horizon_safe_candidates=int((~roll['invalid']).sum()),first_step_safe_candidates=int((~one['invalid']).sum()),
            first_control=first,negative_max_control_two_steps_safe=bool(not probe['invalid'][0]),
            negative_max_control_two_steps_final=probe['final_states'][0].tolist()))
        np.savez_compressed(out/f"{row['state_id']}_{row['planner_seed']}.npz",actions=actions,scored_costs=costs,weights=weights,**roll)
    write(out/'failure_first_decisions.json',failures)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--historical-cache',type=Path,required=True)
    a=p.parse_args();audit(a.root,a.historical_cache)
