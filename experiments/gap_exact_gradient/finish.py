"""Plot saved trajectories and verify provenance; no additional training."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.gap_stochastic.core import QUERIES


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.folder/'manifest.json').read_text())
    assert manifest['status'] == 'complete' and len(manifest['runs']) == 2
    for name,fingerprint in manifest['sources'].items():
        assert sha(ROOT/name) == fingerprint
    baseline = Path(manifest['baseline'])
    for name,fingerprint in manifest['baseline_files'].items():
        assert sha(baseline/name) == fingerprint
    assert sha(args.folder/'PROTOCOL.md') == manifest['protocol_sha256']
    assert (args.folder/'stage1_protocol.md').read_bytes() == (baseline/'PROTOCOL.md').read_bytes()
    basemeta = json.loads((baseline/'manifest.json').read_text())
    errors, batches = {}, 0
    for name,info in manifest['runs'].items():
        old = basemeta['runs'][info['baseline_key']]
        assert info['sampling_sha256'] == old['sampling_sha256']
        assert info['initial_sha256'] == old['initial_sha256']
        with np.load(args.folder/(name+'.npz')) as data:
            audit = data['gradient_audit']
            assert data['x'].shape == (41,36)
            assert data['step'][-1] == 20000
            assert audit.shape == (20000,len(manifest['audit_columns']))
            np.testing.assert_array_equal(audit[:,0],np.arange(1,20001))
            assert np.all(np.isfinite(data['x'])) and np.all(np.isfinite(audit))
            assert np.max(audit[:,5:]) < 1e-10
            errors[name] = {key:float(np.max(audit[:,j])) for j,key in enumerate(manifest['audit_columns']) if j>=5}
            batches += len(audit)
    verification = dict(completed_runs=2,audited_training_batches=batches,
                        baseline_artifacts_unchanged=True, source_and_protocol_hashes_match=True,
                        initial_and_sampling_match=True, gradient_maximum_errors=errors)
    (args.folder/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    history = [json.loads(line) for line in (args.folder/'history.jsonl').read_text().splitlines()]
    fig,axs = plt.subplots(2,3,figsize=(12.5,7))
    for r,name in enumerate(QUERIES):
        for method,color in (('unbiased','#64748b'),('exact_gap','#0d9488')):
            rows = [x for x in history if x['query']==name and x['method']==method]
            steps = [x['step'] for x in rows]
            values = ([x['policy']['source_cost'] for x in rows],
                      [x['residual_linf'] for x in rows], [abs(x['gap']) for x in rows])
            for c,ys in enumerate(values):
                ys = [np.nan if y is None else max(y,1e-9) if c else y for y in ys]
                axs[r,c].plot(steps,ys,color=color,lw=1.5,label='Original unbiased' if method=='unbiased' else 'Exact gap gradient (both blocks)')
        for c,limit in enumerate((3.025,.005,.005)):
            axs[r,c].axhline(limit,color='#b45309',ls=':',lw=1,label='Frozen threshold')
            axs[r,c].grid(alpha=.2)
            axs[r,c].set_xlabel('Updates')
            if c:
                axs[r,c].set_yscale('log')
                axs[r,c].set_ylim(1e-6,2)
            else:
                axs[r,c].set_ylim(2.99,3.26)
        axs[r,0].set_ylabel('Cheap suffix heavy' if r==0 else 'Expensive suffix heavy')
    for ax,title in zip(axs[0],('Executed cost from s','True conservation L-inf','Absolute cost-distance gap')):
        ax.set_title(title,fontsize=11)
    fig.subplots_adjust(left=.07,right=.99,bottom=.15,top=.86,hspace=.33,wspace=.29)
    handles,labels = axs[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.01),ncol=3,fontsize=9)
    fig.suptitle('Exact gap gradient; original stochastic conservation and local risk\nSame seed 2001, batch 256, 20,000 updates; fixed endpoints',fontsize=12)
    fig.savefig(args.folder/'comparison.png',dpi=180)
    fig.savefig(args.folder/'comparison.pdf')
    plt.close(fig)
    print(json.dumps(verification,indent=2))


if __name__ == '__main__':
    main()
