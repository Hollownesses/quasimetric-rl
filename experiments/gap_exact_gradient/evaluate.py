"""Independent fixed-endpoint evaluation, using the original policy extractor."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.gap_stochastic.core import QUERIES
from experiments.gap_stochastic.evaluate import metrics, oracle


def criteria(row):
    return dict(all_queries_reachable=row['policy']['proper'],
                gap=abs(row['gap']) <= .005, risk=row['risk'] <= 1e-4,
                conservation=row['residual_linf'] <= .005,
                source_cost=row['policy']['source_cost'] is not None and row['policy']['source_cost'] <= 3.025,
                query_regret=row['policy_regret'] is not None and row['policy_regret'] <= .005)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    start = time.perf_counter()
    manifest = json.loads((args.folder/'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    baseline = Path(manifest['baseline'])
    previous = json.loads((baseline/'evaluation.json').read_text())
    results = dict(comparisons={}, cases={})
    histories = []
    for name,mu in QUERIES.items():
        truth,_,optimal = oracle(mu)  # Independent evaluation only, after both training runs.
        oldkey = manifest['runs'][name]['baseline_key']
        oldfile = baseline/(oldkey+'.npz')
        assert hashlib.sha256(oldfile.read_bytes()).hexdigest() == manifest['baseline_files'][oldfile.name]
        for method,path in (('unbiased',oldfile), ('exact_gap',args.folder/(name+'.npz'))):
            with np.load(path) as data:
                assert int(data['step'][-1]) == manifest['config']['steps']
                for step,x in zip(data['step'],data['x']):
                    row = metrics(x,mu,'unbiased',truth,optimal)
                    row.update(query=name, method=method, step=int(step), criterion_checks=criteria(row))
                    assert row['joint_pass'] == all(row['criterion_checks'].values())
                    histories.append(row)
            final = histories[-1].copy()
            if method == 'unbiased':
                prior = previous['runs'][oldkey]
                for field in ('gap','risk','residual_linf','residual_l2','policy_regret'):
                    assert abs(final[field]-prior[field]) < 1e-12
                assert final['policy'] == prior['policy']
            results['cases'][name+'__'+method] = final
        old = results['cases'][name+'__unbiased']
        new = results['cases'][name+'__exact_gap']
        results['comparisons'][name] = dict(
            source_cost_change=new['policy']['source_cost']-old['policy']['source_cost'] if new['policy']['source_cost'] is not None else None,
            before_pass=old['joint_pass'], after_pass=new['joint_pass'],
            before_all_queries_reachable=old['policy']['proper'], after_all_queries_reachable=new['policy']['proper'],
            conservation_linf_change=new['residual_linf']-old['residual_linf'],
            query_cost_change=new['policy']['query_cost']-old['policy']['query_cost'] if new['policy']['query_cost'] is not None else None,
            after_failed_criteria=[k for k,v in new['criterion_checks'].items() if not v])
    results['expensive_recovers_and_cheap_still_passes'] = bool(
        not results['comparisons']['expensive_suffix_queries']['before_pass'] and
        results['comparisons']['cheap_suffix_queries']['before_pass'] and
        all(row['after_pass'] for row in results['comparisons'].values()))
    results['evaluation_seconds'] = time.perf_counter()-start
    results['total_seconds'] = results['evaluation_seconds']+manifest['total_seconds']
    assert results['total_seconds'] < manifest['config']['wall_seconds']
    (args.folder/'evaluation.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    with (args.folder/'history.jsonl').open('w') as output:
        for row in histories:
            output.write(json.dumps(row,allow_nan=False)+'\n')
    fields = ('query','method','step','gap','risk','residual_l2','residual_linf','source_cost','query_cost',
              'policy_regret','all_queries_reachable','cheap_probability','joint_pass')
    with (args.folder/'summary.csv').open('w') as output:
        writer = csv.DictWriter(output,fieldnames=fields)
        writer.writeheader()
        for row in results['cases'].values():
            entry = {key:row.get(key) for key in fields}
            entry.update(source_cost=row['policy']['source_cost'],query_cost=row['policy']['query_cost'],
                         all_queries_reachable=row['policy']['proper'],cheap_probability=row['policy']['cheap_probability'])
            writer.writerow(entry)
    for key,row in results['cases'].items():
        print(key, json.dumps({k:row[k] for k in ('gap','risk','residual_l2','residual_linf','joint_pass')}),
              'source_cost',row['policy']['source_cost'],'query_cost',row['policy']['query_cost'],
              'hitting',row['policy']['hitting_probability'])
    print('comparison',json.dumps(results['comparisons']))
    print('expensive_recovers_and_cheap_still_passes',results['expensive_recovers_and_cheap_still_passes'])


if __name__ == '__main__':
    main()
