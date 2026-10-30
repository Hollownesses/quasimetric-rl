"""Fixed task-independent full-observation query expansion; no RNG changes."""
import numpy as np
from scipy.stats import qmc


class QueryAugmentation:
    def __init__(self):
        self.offsets=(2*qmc.Sobol(4,scramble=False).random_base2(18)-1).astype(np.float32)

    def __call__(self,batch,start):
        source=batch['observations'];goals=batch['value_goals']
        assert source.shape==goals.shape and source.ndim==2 and source.shape[1]==2
        assert source.dtype==goals.dtype==np.float32
        count=len(source)//2;assert start>=0 and start+count<=len(self.offsets)
        result=dict(batch);result['observations']=source.copy();result['value_goals']=goals.copy()
        offsets=self.offsets[start:start+count]
        result['observations'][1::2]+=offsets[:,:2]
        result['value_goals'][1::2]+=offsets[:,2:]
        return result
