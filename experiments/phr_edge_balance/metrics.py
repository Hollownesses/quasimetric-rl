"""Post-budget array arithmetic, isolated from all training and weight generation."""
import numpy as np
from experiments.constraint_exposure_diagnostic.core import score_error_identity


def distribution(values):
    v=np.asarray(values,dtype=float)
    if not len(v):return dict(count=0)
    return dict(count=len(v),mean=float(v.mean()),median=float(np.median(v)),
        p90=float(np.quantile(v,.9)),p95=float(np.quantile(v,.95)),p99=float(np.quantile(v,.99)),
        max=float(v.max()),min=float(v.min()),mean_absolute=float(abs(v).mean()),rms=float(np.sqrt(np.mean(v*v))))


def residual_stats(h):
    h=np.asarray(h,dtype=float);p=np.maximum(h,0);n=len(p)
    if not n:return dict(count=0)
    top=max(1,int(np.ceil(n*.01)));bad=h[h>1e-5]
    return dict(count=n,violations=int(len(bad)),mean_signed=float(h.mean()),mean_positive=float(p.mean()),
        p90_positive=float(np.quantile(p,.9)),p95_positive=float(np.quantile(p,.95)),p99_positive=float(np.quantile(p,.99)),
        top1pct_mean_positive=float(np.sort(p)[-top:].mean()),max_positive=float(p.max()),
        violating_p95=float(np.quantile(bad,.95)) if len(bad) else None)


def copy_groups(h,physical_rows,reps,ids,counts,bins):
    groups={}
    for spec in bins:
        mask=counts>=spec['min']
        if spec['max'] is not None:mask &= counts<=spec['max']
        unique=reps[mask];rows=physical_rows[mask[ids[physical_rows]]]
        if not len(unique):continue
        groups[spec['label']]=dict(unique_edges=len(unique),records=len(rows),
            record_mass=len(rows)/len(physical_rows),unique_edge_mass=len(unique)/len(reps),
            unique_edge_violations=residual_stats(h[unique]),record_violations=residual_stats(h[rows]))
    assert sum(g['records'] for g in groups.values())==len(physical_rows)
    assert sum(g['unique_edges'] for g in groups.values())==len(reps)
    return groups


def path_positive(h,episodes,unique=True):
    seen=set();values=[];details=[]
    for ep in episodes:
        start=int(ep['start'])
        if unique and start in seen:continue
        seen.add(start);rr=np.asarray(ep['rows'],dtype=int)
        positive=float(np.maximum(h[rr],0).sum());signed=float(h[rr].sum())
        values.append(positive);details.append(dict(start=start,steps=len(rr),sum_positive=positive,sum_signed=signed))
    return dict(**distribution(values),paths=details)


def successor_diagnostics(data,graph,reference,prediction,decision):
    cost=data.cost.double().numpy();dst=data.dst.numpy();error=prediction-reference
    branches=[];strict_pairs=[];contrasts=[];wrong_contrasts=[]
    for branch in decision['branches']:
        node=branch['source'];rows=np.array(graph.outgoing[node]);choice=branch['chosen_row']
        parts=score_error_identity(cost,prediction,reference,dst,rows,choice)
        assert abs(parts['regret']-branch['regret'])<1e-7
        item=dict(source=node,chosen_row=choice,chosen_successor=int(dst[choice]),
            best_successor=int(dst[parts['best_row']]),chosen_successor_error=float(error[dst[choice]]),
            best_successor_error=float(error[dst[parts['best_row']]]),**parts)
        contrasts.append(parts['successor_error_contrast'])
        if parts['regret']>1e-6:wrong_contrasts.append(parts['successor_error_contrast'])
        branches.append(item)
        # Orient every strict reference pair: the first candidate is truly worse.
        for i,a in enumerate(rows):
            for b in rows[i+1:]:
                truth_a=cost[a]+reference[dst[a]];truth_b=cost[b]+reference[dst[b]]
                if abs(truth_a-truth_b)<=1e-6:continue
                worse,better=(int(a),int(b)) if truth_a>truth_b else (int(b),int(a))
                gap=float(cost[worse]+reference[dst[worse]]-cost[better]-reference[dst[better]])
                contrast=float(error[dst[worse]]-error[dst[better]])
                predicted=float(cost[worse]+prediction[dst[worse]]-cost[better]-prediction[dst[better]])
                assert abs(predicted-gap-contrast)<1e-7
                strict_pairs.append(dict(source=node,worse_row=worse,better_row=better,reference_gap=gap,
                    successor_error_contrast=contrast,predicted_gap=predicted,correct=predicted>0))
    accuracy=sum(x['correct'] for x in strict_pairs)/len(strict_pairs)
    assert abs(accuracy-decision['summary']['pairwise_order_accuracy'])<1e-12
    return dict(branches=branches,strict_pairs=strict_pairs,summary=dict(
        all_chosen_contrast=distribution(contrasts),suboptimal_chosen_contrast=distribution(wrong_contrasts),
        strict_pair_contrast=distribution([x['successor_error_contrast'] for x in strict_pairs]),
        strict_pairs=len(strict_pairs),correct_pairs=sum(x['correct'] for x in strict_pairs),
        pairwise_order_accuracy=accuracy,suboptimal_branches=len(wrong_contrasts)))


def common_pair_changes(left,right):
    a={(x['source'],x['worse_row'],x['better_row']):x for x in left['strict_pairs']}
    b={(x['source'],x['worse_row'],x['better_row']):x for x in right['strict_pairs']}
    assert a.keys()==b.keys()
    changed=[dict(source=k[0],worse_row=k[1],better_row=k[2],R1_correct=a[k]['correct'],R2_correct=b[k]['correct'],
        R1_contrast=a[k]['successor_error_contrast'],R2_contrast=b[k]['successor_error_contrast']) for k in a if a[k]['correct']!=b[k]['correct']]
    return dict(correct_to_wrong=sum(a[k]['correct'] and not b[k]['correct'] for k in a),
        wrong_to_correct=sum(not a[k]['correct'] and b[k]['correct'] for k in a),
        contrast_delta=distribution([b[k]['successor_error_contrast']-a[k]['successor_error_contrast'] for k in a]),changed_pairs=changed)
