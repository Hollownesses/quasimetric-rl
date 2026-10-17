"""Success-conditioned costs and paired summaries; truncated failure costs excluded."""
import numpy as np


def mean(values):return float(np.mean(values)) if len(values) else None


def aggregate(rows):
    successes=[r for r in rows if r['success']]
    result=dict(episodes=len(rows),successes=len(successes),success_rate=len(successes)/len(rows),
        successful_cost_mean=mean([r['cost'] for r in successes]),successful_steps_mean=mean([r['steps'] for r in successes]))
    for key in ('collision','out_of_bounds','timeout'):
        result[key+'_count']=sum(r[key] for r in rows);result[key+'_rate']=result[key+'_count']/len(rows)
    for name in ('exit','wrap'):
        eligible=[r for r in rows if r[name+'_eligible']]
        result[name+'_eligible']=len(eligible)
        result[name+'_completed']=sum(bool(r[name+'_completed']) for r in eligible)
        result[name+'_rate']=result[name+'_completed']/len(eligible) if eligible else None
        result['cumulative_'+name+'_rate']=mean([r['cumulative_'+name] for r in rows])
    result.update(planning_seconds=sum(r['planning_seconds'] for r in rows),
        planning_ms_per_decision=1000*sum(r['planning_seconds'] for r in rows)/sum(r['steps'] for r in rows),
        model_rollouts=sum(r['model_rollouts'] for r in rows),model_transitions=sum(r['model_transitions'] for r in rows))
    return result


def paired(before,after):
    key=lambda r:(r['state_id'],r['planner_seed'])
    a={key(r):r for r in before};b={key(r):r for r in after}
    assert len(a)==len(before) and len(b)==len(after) and a.keys()==b.keys()
    pairs=[(a[k],b[k]) for k in a]
    both=[(x,y) for x,y in pairs if x['success'] and y['success']]
    return dict(pairs=len(pairs),both_success=len(both),
        improved=sum(not x['success'] and y['success'] for x,y in pairs),
        worsened=sum(x['success'] and not y['success'] for x,y in pairs),
        both_failure=sum(not x['success'] and not y['success'] for x,y in pairs),
        success_delta=mean([int(y['success'])-int(x['success']) for x,y in pairs]),
        joint_success_cost_delta=mean([y['cost']-x['cost'] for x,y in both]),
        joint_success_steps_delta=mean([y['steps']-x['steps'] for x,y in both]))
