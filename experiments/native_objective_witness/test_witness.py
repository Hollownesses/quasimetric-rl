import numpy as np
import pytest
from experiments.native_objective_witness.core import hard_boundary,mean_boundary,risk,push_loss,hoeffding_halfwidth,coordinate_scale

def test_boundaries_must_allow_expansion_not_only_feasibility_repair():
    a,b=hard_boundary([.25,.5],[1.,1.]);assert a==2 and b>1
    lo,hi,a=mean_boundary([.5,.5],.05)
    np.testing.assert_allclose(lo,(1+np.sqrt(.05))/.5,atol=1e-14,rtol=0)
    assert hi-lo<1e-14 and a>1 and risk([.5,.5],a)<.05

def test_native_shaping_and_multiple_comparison_bound():
    assert np.all(np.diff(push_loss([0,1,100,10000]))<0)
    assert push_loss(0)<501 and push_loss(100000)<1e-100
    np.testing.assert_allclose(hoeffding_halfwidth(4*1024),hoeffding_halfwidth(1024)/2)
    assert hoeffding_halfwidth(1024,4)>hoeffding_halfwidth(1024,1)

def test_zero_metric_cannot_claim_a_finite_optimum():
    with pytest.raises(ValueError):hard_boundary([0.],[1.])
    with pytest.raises(ValueError):mean_boundary([0.],.05)

def test_dynamics_coordinate_change_leaves_actions_unscaled_and_preserves_prediction():
    rng=np.random.default_rng(22)
    params={'modules_value':{'phi':{'Dense_0':{'kernel':rng.normal(size=(2,2)),'bias':np.ones(2)}}},
            'modules_dynamics':{'Dense_0':{'kernel':rng.normal(size=(3,4)),'bias':rng.normal(size=4)},
                                'Dense_1':{'kernel':rng.normal(size=(4,2)),'bias':rng.normal(size=2)}},
            'modules_actor':{'frozen':np.array([7.])}}
    alpha=.73;new=coordinate_scale(params,alpha,2)
    def dynamics(p,z,a):
        d=p['modules_dynamics'];x=np.c_[z,a];y=np.maximum(x@d['Dense_0']['kernel']+d['Dense_0']['bias'],0)
        return y@d['Dense_1']['kernel']+d['Dense_1']['bias']
    z=rng.normal(size=(5,2));a=rng.normal(size=(5,1))
    np.testing.assert_allclose(dynamics(new,alpha*z,a),alpha*dynamics(params,z,a),atol=1e-7,rtol=1e-7)
    np.testing.assert_array_equal(new['modules_dynamics']['Dense_0']['kernel'][-1],params['modules_dynamics']['Dense_0']['kernel'][-1])
    np.testing.assert_array_equal(new['modules_actor']['frozen'],params['modules_actor']['frozen'])

def test_reaggregation_excludes_all_512_historical_batches_and_is_repeatable(tmp_path):
    import csv,json
    from experiments.native_objective_witness.reaggregate import run
    names=['original_raw','E16_raw','original_boundary','E16_boundary']
    points={name:dict(feasible=1,D_raw=1. if name.startswith('E16') else 0.,D_extracted=.2 if name.startswith('E16') else .1) for name in names}
    def write(name,value):(tmp_path/name).write_text(json.dumps(value))
    write('audit.json',dict(points=points))
    write('query_audit.json',dict(three_field_sha256='three',full_field_sha256='full'))
    write('query_history.json',dict(historical_512_exact=True,prefix_sha256={'1024':'three'},full_fields_sha256='full'))
    for p in ['points.csv','comparisons.csv']:(tmp_path/p).write_text('initial\n')
    with (tmp_path/'queries.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['point','batch','P']);w.writeheader()
        for i in range(1024):
            for name in names:
                p=10 if name.startswith('original') else (0 if i<512 else 12)
                w.writerow(dict(point=name,batch=i,P=p))
    run(tmp_path)
    with (tmp_path/'comparisons.csv').open() as f:comparison=list(csv.DictReader(f))
    assert len(comparison)==4 and all(float(r['delta_P'])==2 for r in comparison)
    qa=json.loads((tmp_path/'query_audit.json').read_text());assert qa['new_query_batches']==512 and qa['new_queries']==524288
    before=(tmp_path/'comparisons.csv').read_bytes();run(tmp_path)
    assert (tmp_path/'comparisons.csv').read_bytes()==before
