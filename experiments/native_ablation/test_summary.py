import pytest
from experiments.native_ablation.summarize import paired_counts,success

def test_boolean_history_and_numeric_new_records_pair_correctly():
    parent={'a':{'success':'True'},'b':{'success':'False'},'c':{'success':'0'},'d':{'success':'1'}}
    current={'a':{'success':'0'},'b':{'success':'1'},'c':{'success':'False'},'d':{'success':'True'}}
    assert paired_counts(current,parent)==(1,1)
    assert sum(success(r['success']) for r in current.values())==2

def test_missing_episode_is_rejected_instead_of_paired_by_row_order():
    with pytest.raises(ValueError,match='unmatched'):
        paired_counts({'new':{'success':1}},{'old':{'success':0}})

def test_unknown_flag_cannot_silently_count_as_success():
    with pytest.raises(ValueError,match='unrecognized'):success('timeout')
