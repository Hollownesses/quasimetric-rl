"""Reference-only selection and signed, shared-suffix-safe counterfactuals."""
from collections import Counter
import numpy as np

def select_edges(good_path,counts,cost):
    if not len(good_path):return None,None
    rare=min(good_path,key=lambda e:(counts[e],e))
    candidates=[e for e in good_path if counts[e]>counts[rare] and abs(cost[e]-cost[rare])<=max(.05,.1*cost[rare])]
    control=min(candidates,key=lambda e:(abs(cost[e]-cost[rare]),-counts[e],e)) if candidates else None
    return rare,control

def counterfactual(margin,good_path,bad_path,h,selected,positive_only=False):
    """Remove the same physical edge component from every occurrence on both sides."""
    a,b=Counter(good_path),Counter(bad_path)
    delta=-sum((b[e]-a[e])*(max(h[e],0) if positive_only else h[e]) for e in selected)
    return float(margin+delta),float(delta)
