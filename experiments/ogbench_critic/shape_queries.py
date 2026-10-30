"""E12 current queries unchanged; translate successor by the same offset."""
from query_augmentation import QueryAugmentation


class ShapeQueries(QueryAugmentation):
    def __call__(self,batch,start):
        result=super().__call__(batch,start);count=len(batch['observations'])//2
        result['next_observations']=batch['next_observations'].copy()
        result['next_observations'][1::2]+=self.offsets[start:start+count,:2]
        return result
