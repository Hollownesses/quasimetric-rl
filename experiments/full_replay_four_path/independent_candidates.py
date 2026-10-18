"""Fixed candidate diagnostics on every independently frozen pose, after training."""
import argparse
from pathlib import Path
import numpy as np
import torch
from experiments.full_replay_four_path.evaluate import make_controller,ARMS
from experiments.four_path_execution.common import load_data,read,write,PLANNER_SEEDS
from experiments.four_path_oracle_execution.protocol import CONFIG
from minimal_qrl.baselines.mppi import simulate_action_sequences
from minimal_qrl.eval.u_trap_local_navigability import _rankdata,_correlation


def run(root):
    assert (root/'train/COMPLETE').exists();torch.set_num_threads(1)
    out=root/'independent_candidates';out.mkdir(exist_ok=False)
    data=load_data();env=data.env;controls={a:make_controller(root)(data,a) for a in ARMS}
    rows=[];paired=[]
    for record in read(root/'protocol.json')['independent_states']:
        pose=np.asarray(record['pose'],np.float32)
        for seed in PLANNER_SEEDS:
            name=f"{record['id']}_{seed}"
            env.reset(seed=seed,options={'device_id':'u_trap_target','start':pose})
            controls['none'].begin_episode(env,env.abstract_goal_observation(),seed)
            actions=controls['none'].sample_candidates(env)
            rollout=simulate_action_sequences(env,env.state,actions,record_trajectory=True)
            safe=~rollout['invalid'];final=rollout['final_states']
            useful=safe&(rollout['steps']==CONFIG.horizon)&(pose[0]-final[:,0]>=.25)&(np.cos(final[:,2])<0)
            scores={};arrays=dict(actions=actions,**rollout,useful=useful)
            for arm,control in controls.items():
                env.reset(seed=seed,options={'device_id':'u_trap_target','start':pose})
                control.begin_episode(env,env.abstract_goal_observation(),seed)
                cost,weight=control.score_candidates(env,rollout)
                first=float(np.sum(weight[:,None]*actions,axis=0).astype(np.float32)[0])
                _,_,_,_,info=env.step(np.array([first],np.float32))
                arrays[arm+'_costs']=cost;arrays[arm+'_weights']=weight;scores[arm]=(cost,weight,first)
                rows.append(dict(probe=name,arm=arm,safe_count=int(safe.sum()),useful_count=int(useful.sum()),
                    useful_weight=float(weight[useful].sum()),first_control=first,
                    first_control_safe=not(info['collision'] or info['out_of_bounds']),
                    effective_samples=float(1/np.square(weight).sum())))
            for arm in ARMS:
                if arm=='oracle':continue
                a,wa,ua=scores['oracle'];b,wb,ub=scores[arm]
                paired.append(dict(probe=name,arm=arm,safe_count=int(safe.sum()),
                    safe_rank_spearman=_correlation(_rankdata(a[safe]),_rankdata(b[safe])) if safe.sum()>1 else None,
                    weight_total_variation=float(abs(wa-wb).sum()/2),first_control_delta=ub-ua))
            np.savez_compressed(out/f'{name}.npz',**arrays)
    write(out/'scores.json',rows);write(out/'paired.json',paired)
    write(out/'COMPLETE',dict(banks=48,arms=len(ARMS),horizon=CONFIG.horizon,num_samples=CONFIG.num_samples))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
