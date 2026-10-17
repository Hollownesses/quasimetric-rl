"""One fixed candidate bank per development pose/RNG, scored by every model."""
import argparse
from experiments.four_path_execution.common import *
from minimal_qrl.baselines.mppi import simulate_action_sequences
from minimal_qrl.eval.u_trap_local_navigability import _rankdata,_correlation


def diagnostic(root,*,config=CONFIG,arms=ARMS,controller_factory=controller,comparison_pairs=None):
    if comparison_pairs is None:
        comparison_pairs=[(f'decay_{seed}',f'qp_{seed}') for seed in SEEDS]
    torch.set_num_threads(1);manifest=read(root/'manifest.json')
    assert asdict(config)==manifest['config']
    for name,sha in manifest['sources'].items():assert file_digest(ROOT/name)==sha,name
    out=root/'candidates';out.mkdir(exist_ok=False)
    data=load_data();env=data.env;controllers={a:controller_factory(data,a) for a in arms};rows=[];paired=[]
    for x in (4.8,4.0):
        for heading,theta in [('west',np.pi),('north',np.pi/2),('east',0.),('south',-np.pi/2)]:
            pose=np.array([x,3.65,theta],np.float32)
            for seed in PLANNER_SEEDS:
                probe=f'x{x}_{heading}_{seed}'
                env.reset(seed=seed,options={'device_id':'u_trap_target','start':pose})
                baseline=controllers['none'];baseline.begin_episode(env,env.abstract_goal_observation(),seed)
                actions=baseline.sample_candidates(env)
                rollout=simulate_action_sequences(env,env.state,actions,record_trajectory=True)
                final=rollout['final_states'];safe=~rollout['invalid']
                useful=safe&(rollout['steps']==config.horizon)&((pose[0]-final[:,0])>=.25)&(np.cos(final[:,2])<0.)
                states=rollout['trajectory'];exit_mask=((states[:,:,0]<=2.8)&(states[:,:,1]>=1.9)&(states[:,:,1]<=5.4)).any(axis=1)&safe
                arrays=dict(actions=actions,**rollout,useful=useful,west_exit=exit_mask)
                scores={}
                for arm,control in controllers.items():
                    env.reset(seed=seed,options={'device_id':'u_trap_target','start':pose})
                    control.begin_episode(env,env.abstract_goal_observation(),seed)
                    costs,weights=control.score_candidates(env,rollout)
                    first=float(np.sum(weights[:,None]*actions,axis=0).astype(np.float32)[0])
                    _,_,_,_,info=env.step(np.array([first],np.float32))
                    top=np.flatnonzero(safe)[np.argsort(costs[safe],kind='stable')[:10]]
                    scores[arm]=(costs,weights,top,first)
                    arrays[arm+'_costs']=costs;arrays[arm+'_weights']=weights
                    unfinished=~(rollout['success']|rollout['invalid'])
                    terminal=costs-rollout['costs']-rollout['invalid']*config.invalid_penalty
                    arrays[arm+'_weighted_terminal_contribution']=terminal
                    rows.append(dict(probe=probe,x=x,heading=heading,planner_seed=seed,arm=arm,
                        sampled=len(actions),safe_count=int(safe.sum()),useful_count=int(useful.sum()),
                        exit_count=int(exit_mask.sum()),useful_weight=float(weights[useful].sum()),exit_weight=float(weights[exit_mask].sum()),
                        top10_useful=int(useful[top].sum()),first_control=first,first_control_safe=not(info['collision'] or info['out_of_bounds']),
                        useful_terminal_mean=float(terminal[useful].mean()) if useful.any() else None,
                        other_safe_terminal_mean=float(terminal[safe&~useful&unfinished].mean()) if (safe&~useful&unfinished).any() else None,
                        effective_samples=float(1./np.square(weights).sum())))
                for before,after in comparison_pairs:
                    a,wa,ta,ua=scores[before];b,wb,tb,ub=scores[after]
                    paired.append(dict(probe=probe,before=before,after=after,
                        **({'seed':int(after.split('_')[1])} if before.startswith('decay_') else {}),safe_candidates=int(safe.sum()),useful_candidates=int(useful.sum()),
                        safe_rank_spearman=_correlation(_rankdata(a[safe]),_rankdata(b[safe])) if safe.sum()>1 else None,
                        top10_overlap=len(set(ta)&set(tb)),top10_size=len(ta),
                        weights_total_variation=float(abs(wa-wb).sum()/2),first_control_delta=float(ub-ua),
                        useful_weight_delta=float(wb[useful].sum()-wa[useful].sum())))
                np.savez_compressed(out/f'{probe}.npz',**arrays)
    write(out/'scores.json',rows);write(out/'paired.json',paired)
    write(out/'COMPLETE',dict(candidate_banks=16,unique_model_rollouts=16*config.num_samples,scoring_arms=len(arms)))
    print('candidate diagnostics complete',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);diagnostic(Path(p.parse_args().root))
