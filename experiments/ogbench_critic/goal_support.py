"""Change only E12 odd-row goal centers; deterministic uniform support cycle."""
import math
import numpy as np
from query_augmentation import QueryAugmentation


class GoalSupport(QueryAugmentation):
    def __init__(self,nodes):
        super().__init__();self.nodes=np.asarray(nodes,np.float32)
        assert self.nodes.ndim==2 and self.nodes.shape[1]==2 and len(self.nodes)>0
        self.stride=math.ceil(len(self.nodes)*(math.sqrt(5)-1)/2)
        while math.gcd(self.stride,len(self.nodes))!=1:self.stride+=1

    def indices(self,start,count):
        assert start>=0 and count>=0
        return (np.arange(start,start+count,dtype=np.int64)*self.stride)%len(self.nodes)

    def __call__(self,batch,start):
        result=super().__call__(batch,start);count=len(batch['observations'])//2
        result['value_goals'][1::2]=self.nodes[self.indices(start,count)]+self.offsets[start:start+count,2:]
        return result
