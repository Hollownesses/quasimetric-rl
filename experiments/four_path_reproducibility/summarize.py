"""Audit all preregistered seeds, retaining original failures and corrected runs."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.four_path_factorial.run import load_data, file_digest, tensor_digest
from experiments.four_path_refinement.run import diagnose, restore
from experiments.four_path_reproducibility.run import SEEDS


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def table(headers,rows):
    esc=lambda x:str(x).replace('|',r'\|')
    return '| '+' | '.join(map(esc,headers))+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+''.join('| '+' | '.join(map(esc,r))+' |\n' for r in rows)


def source_audit(root):
    manifest=read(root/'manifest.json')
    for name,digest in manifest['sources'].items():
        assert file_digest(root/'source_snapshot'/name)==digest
    assert tuple(manifest['seeds'])==SEEDS
    return manifest


def summarize(root,corrected=None):
    torch.set_num_threads(2)
    out=root/'comparison';out.mkdir(exist_ok=False)
    data=load_data();data.fingerprint.pop('initial_checkpoint')
    manifest=source_audit(root);assert not manifest['smoke']
    fixed_manifest=source_audit(corrected) if corrected else manifest
    process=read(root/'process_results.json')
    final_process=read(corrected/'process_results.json') if corrected else process
    assert set(process)==set(final_process)==set(map(str,SEEDS))
    audited={};rows=[];configs=[];cost_rows=[];original_rows=[];failure_rows=[];path_rows=[];entry_rows=[];guard_rows=[];update_rows=[]
    original_success=[];success=[]
    for seed in SEEDS:
        directory=root/f'seed_{seed}';repair=(corrected/f'seed_{seed}') if corrected else directory/'repair'
        cfg=read(directory/'config.json');configs.append(cfg)
        assert cfg['fingerprints']==data.fingerprint
        assert cfg['script_sha256']==manifest['sources']['experiments/four_path_reproducibility/run.py']
        assert cfg['initial_adam_states']==cfg['initial_nonzero_multipliers']==0
        initial,initial_dual,initial_opt,initial_saved=restore(data,directory/'initial.pt')
        assert initial_saved['initialization_seed']==seed and not initial_opt.state and not initial_dual.multiplier.any()
        for key,module in [('initial_model_sha256',initial),('initial_encoder_sha256',initial.encoder),('initial_projector_sha256',initial.projector)]:
            assert tensor_digest(*module.state_dict().values())==cfg[key]
        assert tensor_digest(initial.project())==cfg['initial_projected_sha256']
        original_summary=read(directory/'summary.json') if (directory/'summary.json').exists() else None
        original_complete=process[str(seed)]['exit_code']==0 and (directory/'COMPLETE').exists()
        if original_complete:
            assert original_summary is not None
            old=original_summary['stages']['qp']
            if old['joint_pass']:original_success.append(seed)
            original_rows.append([seed,0,original_summary['repair_stop_reason'],f"{old['push_gap_percent']:.5f}",'通过' if old['joint_pass'] else '失败'])
        else:
            log=(root/f'seed_{seed}.log').read_text()
            original_rows.append([seed,process[str(seed)]['exit_code'],'Gram PSD断言' if 'AssertionError' in log else '进程失败','未完成','失败'])
        updates=lines(directory/'updates.jsonl');training=lines(directory/'training.jsonl')
        assert len(updates)==25000 and sum(u['dual_updated'] for u in updates)==25000
        assert [u['total_primal_step'] for u in updates]==list(range(1,25001))
        assert all(u['lr']==2e-5 for u in updates[:20000])
        np.testing.assert_allclose([u['lr'] for u in updates[20000:]],2e-6*.1**(np.arange(5000)/4999),rtol=1e-14,atol=0)
        phr_log=[r for r in training if r['stage']=='phr'][-1]
        decay_logs=[r for r in training if r['stage']=='decay']
        assert decay_logs[0]['push']==phr_log['push'] and decay_logs[0]['max_violation']==phr_log['max_violation']
        if original_summary:
            boundary=original_summary['stage_boundary']
            assert all(boundary[k]==original_summary['timing']['phr'][k] for k in boundary)
        for stage in ('phr','decay'):
            stage_updates=[u for u in updates if u['stage']==stage]
            update_rows.append([seed,stage,np.mean([u['parameter_update_l2'] for u in stage_updates]),
                max(u['parameter_update_l2'] for u in stage_updates),np.mean([u['parameter_update_l2'] for u in stage_updates[-250:]]),
                max(u['parameter_update_linf'] for u in stage_updates)])
        repair_cfg=read(repair/'config.json')
        assert repair_cfg['delta']==.15 and repair_cfg['push_step']==0. and repair_cfg['iterations']==40
        assert repair_cfg['fingerprints']==data.fingerprint
        assert repair_cfg['checkpoint_sha256']==file_digest(directory/'decay.pt')
        assert repair_cfg['script_sha256']==fixed_manifest['sources']['experiments/four_path_refinement/repair.py']
        if corrected:assert fixed_manifest['decay_checkpoints'][str(seed)]==repair_cfg['checkpoint_sha256']
        complete=final_process[str(seed)]['exit_code']==0 and (repair/'COMPLETE').exists()
        audit=dict(initial_checkpoint_sha256=file_digest(directory/'initial.pt'),original_complete=original_complete,original_process=process[str(seed)],final_process=final_process[str(seed)],stages={})
        audited[str(seed)]=audit
        stages=[('phr',directory/'phr.pt'),('decay',directory/'decay.pt')]
        if complete:stages.append(('qp',repair/'latest.pt'))
        for stage,path in stages:
            model,dual,opt,saved=restore(data,path)
            rec,values,arrays=diagnose(model,data,dual,saved['step'],0.)
            rec.update(strict_physical_feasible=rec['max_residual']<=0.,checkpoint_sha256=file_digest(path),
                       model_sha256=tensor_digest(*model.state_dict().values()))
            assert saved['initialization_seed']==seed
            if original_summary and (stage!='qp' or not corrected):
                expected=original_summary['stages'][stage]
                for key in ('push','max_residual','task_nrmse_percent','terminal_max','anchor_max','complementarity'):
                    assert abs(rec[key]-expected[key])<2e-5,(seed,stage,key)
                assert rec['joint_pass']==expected['joint_pass']
                assert rec['model_sha256']==expected['model_sha256'] and rec['checkpoint_sha256']==expected['checkpoint_sha256']
            if stage!='qp':
                count=20000 if stage=='phr' else 25000
                assert saved['step']==count and all(int(v['step'])==count for v in opt.state.values())
                if stage=='decay':
                    assert abs(repair_cfg['initial_push']-rec['push'])<2e-5
                    assert abs(repair_cfg['fixed_push_floor']-(repair_cfg['initial_push']-.15))<1e-9
                    assert repair_cfg['initial_model_sha256']==rec['model_sha256']
            else:
                assert rec['push']>=repair_cfg['fixed_push_floor']
                assert saved['stop_reason']==read(repair/'evaluation.json')['stop_reason']
                stored=np.load(repair/'latest_edges.npz')
                for key in arrays:np.testing.assert_allclose(stored[key],arrays[key],rtol=0,atol=2e-5)
                if corrected and original_complete:
                    old_model=torch.load(directory/'repair/latest.pt',weights_only=True)['model']
                    differences=torch.cat([(value.double()-old_model[key].double()).flatten()
                        for key,value in model.state_dict().items() if value.is_floating_point()])
                    audit['qp_repeat_comparison']=dict(identical_model=rec['model_sha256']==original_summary['stages']['qp']['model_sha256'],
                        parameter_difference_l2=float(differences.norm()),parameter_difference_linf=float(differences.abs().max()),
                        push_difference=rec['push']-original_summary['stages']['qp']['push'],
                        same_joint_result=rec['joint_pass']==original_summary['stages']['qp']['joint_pass'])
            audit['stages'][stage]=rec
            np.savez(out/f'seed_{seed}_{stage}_edges.npz',**arrays)
            write(out/f'seed_{seed}_{stage}_values.json',values)
            rows.append(dict(seed=seed,stage=stage,**{k:rec[k] for k in ('push','push_gap_percent','task_nrmse_percent','max_residual','max_violation','violation_count','strict_physical_feasible','joint_pass','complementarity','complementarity_max','anchor_max','terminal_max')}))
            for p in rec['paths']:
                path_rows.append([seed,stage,p['path'],f"{p['start']:.5f}",f"{p['reference']:.5f}",f"{p['positive_residual_sum']:.7f}",f"{p['triangle_slack']:.7f}"])
            for e in rec['terminal_entry_edges']:
                entry_rows.append([seed,stage,e['path'],e['row'],f"{e['residual']:+.8f}"])
        trace=lines(repair/'qp.jsonl');curve=lines(repair/'curve.jsonl')
        assert len(trace)<=40
        assert all(not any(k in r for k in ('joint_pass','task_nrmse_percent','push_reference')) for r in (curve[:-1] if complete else curve))
        for q in trace:
            for candidate in q['attempts']:
                if candidate['accepted']:assert candidate['push']>=repair_cfg['fixed_push_floor']
        audit['success']=bool(complete and audit['stages']['qp']['joint_pass'])
        if audit['success']:success.append(seed)
        if complete:
            final=audit['stages']['qp'];reason=curve[-1]['stop_reason'];failure=[]
            assert len(trace)==curve[-1]['step']
            guard_rows.append([seed,f"{repair_cfg['initial_push']:.6f}",f"{repair_cfg['fixed_push_floor']:.6f}",f"{final['push']:.6f}",
                f"{repair_cfg['initial_push']-final['push']:.6f}",f"{sum(q['actual_parameter_update_l2'] for q in trace):.6f}"])
            if final['max_violation']>1e-5:failure.append('逐边可行性')
            if final['push_gap_percent']>1:failure.append('push gap')
            if final['task_nrmse_percent']>2:failure.append('task NRMSE')
            if max(final['terminal_max'],final['anchor_max'])>1e-6:failure.append('终态/anchor')
            failure_rows.append([seed,reason,'通过' if not failure else '、'.join(failure)])
        else:failure_rows.append([seed,'进程失败','失败，未替换seed'])
        cost_rows.append([seed,20000,5000,len(trace),sum(q['accepted'] for q in trace),
            sum(1+any(c['mode']=='elastic' for c in q['attempts']) for q in trace),sum(len(q['attempts']) for q in trace),
            sum(q.get('gram_numerics',{}).get('stable_block_fallback',False) for q in trace),
            f"{phr_log['seconds']:.1f}",f"{decay_logs[-1]['seconds']:.1f}",f"{curve[-1]['seconds']:.1f}"])
    keys=('initial_model_sha256','initial_encoder_sha256','initial_projector_sha256','initial_projected_sha256')
    assert all(len({c[key] for c in configs})==5 for key in keys)
    write(out/'independent_audit.json',audited)
    write(out/'initialization_hashes.json',[dict(seed=c['seed'],**{k:c[k] for k in keys}) for c in configs])
    write(out/'data_fingerprints.json',data.fingerprint)
    with (out/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(rows)
    stage_counts={stage:sum(r['joint_pass'] for r in rows if r['stage']==stage) for stage in ('phr','decay','qp')}
    write(out/'summary.json',dict(seeds=SEEDS,denominator=5,original_successful_seeds=original_success,
        original_success_count=len(original_success),numerical_correction=bool(corrected),successful_seeds=success,
        success_count=len(success),stage_success_counts=stage_counts,stages=rows,failures=failure_rows,compute=cost_rows))
    plot(root,out,rows)
    display=[[r['seed'],r['stage'],f"{r['push']:.5f}",f"{r['push_gap_percent']:.4f}",f"{r['task_nrmse_percent']:.4f}",
        f"{r['max_residual']:+.8f}",r['violation_count'],'是' if r['strict_physical_feasible'] else '否','通过' if r['joint_pass'] else '失败'] for r in rows]
    correction_text=''
    if corrected:
        correction_text=f'''原版冻结运行的联合通过率为 **{len(original_success)}/5**，两个seed（202、303）在QP第0轮触发Gram半正定断言，完整失败日志保留。排查发现，四项展开式在相减时产生数值误差：最小/最大特征值比为−1.49×10⁻¹⁰、−1.60×10⁻¹⁰，略超原1e-10阈值。没有放宽阈值，而是在检查失败时直接累加显式Jacobian分块的PSD内积。与原展开Gram相差的相对范数为7.37×10⁻¹⁰、7.81×10⁻¹⁰，独立参数空间二次型检查相对误差小于2.2×10⁻¹³。详见[数值诊断](../../gram_diagnosis.json)。

统一修复后，对**全部五个seed**都从各自原始decay.pt重新执行完全相同预算的QP；没有重新训练、改δ、改容差、加预算或替换seed。原正常完成的404、505得到逐位相同模型；101未触发回退，但重跑从3轮变为2轮，最终push差−3.05×10⁻⁵、参数差L2=1.83×10⁻⁴，联合失败结论不变。这里没有假定多线程CPU及近奇异Gram分解逐位可复现，已记录这一数值敏感性。数值修复是看到运行错误后作出的实现修正，因此上报原版与修正版两组结果，不声称修正版结果是完全未经事后修正的预登记结果。修正版源码、来源checkpoint哈希及日志保存在[数值修复清单](../../qp_numerical_fix/manifest.json)。
'''
    report=f'''# 四路径固定流程的五个新初始化复现（2026-10-17）

最终审计：**{len(success)}/5通过联合门槛**，成功seed为{success}。PHR后、降步长后、QP后分别通过{stage_counts['phr']}/5、{stage_counts['decay']}/5、{stage_counts['qp']}/5。分母始终是101、202、303、404、505，不替换失败seed。本次结果不支持“固定流程可稳定重复找到通过解”；旧checkpoint的成功仍是存在性证据。

{correction_text}
{table(['seed','原进程exit','原QP结果/停止原因','原push gap %','原联合门槛'],original_rows)}

## 三个固定阶段的完整结果

{table(['seed','阶段','push','push gap %','task NRMSE %','最大有符号h','违反边/335','严格h≤0','联合门槛'],display)}

门槛保持不变：物理边最大正残差≤1e-5、终态与anchor≤1e-6、同采样支持push gap≤1%、task NRMSE≤2%。严格可行单独按float32前向max h≤0判断，容差内正残差不能称严格可行，浮点检查也不是实数算术证明。记录图参考push=81.617389992，只用于结束后的评估，不是连续环境最优距离。详见[CSV](summary.csv)、[独立审计](independent_audit.json)。

{table(['seed','最终模型自身停止原因','未通过项'],failure_rows)}

![三个阶段](stages.png)

违反量对数轴把零画在1e-8处，仅用于显示；有符号原值见表。

## 固定价值保护与阶段作用

{table(['seed','降步长终点push','固定push下限','QP终点push','实际push损失','Σ接受步参数L2'],guard_rows)}

δ始终为绝对0.15。价值下限只来自各模型的降步长终点，既不读取图参考，也不按seed重选。本次降步长后的五个模型均已满足push gap≤1%与task NRMSE≤2%，但局部边仍不可行；QP后五个模型都满足可行容差，其中202、303、404严格非正，另外两个保留容差内微小正残差。全部task NRMSE仍通过，四个失败均来自push gap。

独立评估给出的1% push门槛为80.801216092。五个修复起点距离这一门槛的余量依次仅0.037338、0.056053、0.115395、0.204483、0.131470，实际修复损失约0.136–0.150；只有404有足够余量。因此主要暴露的是当前恢复步骤的价值消耗与起点余量之间的耦合，而不是单纯“修不到局部可行”。R0最小化参数改动，并不最小化push损失；0.15模型保护条件也不保证独立参考的1%门槛。该参考余量仅在结束后分析，未用于停止、修改δ或重跑选择。不能据此断言更小损失的可行解不存在。

建议下一轮把“降步长阶段已展开多少价值”与“修复消耗多少模型push”作为两个问题。首先验证统一预算下更充分的primal/PHR子问题优化或固定的模型push上升策略，再冻结另一套流程，在新的独立初始化上检验。当前5个seed保留为诊断集；不要用它们逐个调δ或修复预算后重新宣称本次成功。局部修复停滞也不证明不存在高价值可行解。

![训练曲线](training.png)

训练曲线只记录模型push、一步约束残差与互补性，没有图参考驱动停止。全边乘子互补性均值/最大值见CSV与审计JSON；QP保留训练乘子，这些量是诊断，不是QP重新求得的KKT证书。

{table(['seed','阶段','平均参数更新L2','最大L2','末250步平均L2','最大L∞'],[[r[0],r[1]]+[f'{v:.7g}' for v in r[2:]] for r in update_rows])}

## 冻结支持、初始化与停止隔离

[原始预登记协议](../source_snapshot/experiments/four_path_reproducibility/PROTOCOL.md)：同一四路径bank、完整42D规范状态的336节点、335物理边+4 anchor、终态quotient、task/B1配对与一步成本。B1数据seed固定42，与网络seed无关。[数据指纹](data_fingerprints.json)全部一致。

每个seed通过生产QuasimetricCritic构造器从头初始化encoder、projector、IQE；没有加载旧critic，Adam为空，339个逐边乘子为零。[完整模型/encoder/projector/表示哈希](initialization_hashes.json)均五份不同。测试在禁止torch.load时仍创建五份新模型，同seed可重建相同参数。

PHR固定20k全量primal/dual更新，AdamW lr2e-5、wd0、ρ100、λ←[λ+h_pre]+；接着5k仅把全部primal学习率按2e-6×0.1^((j−1)/4999)降至2e-7，保留Adam状态及乘子。QP固定最终projector整层1,050,624个参数，全339边加模型push下限，R0方向目标为零；最多40轮、每次2000 sweeps、信赖半径0.1、内部余量5e-5、13级回溯、elastic fallback100。真实非线性前向检查所有候选，允许不可行起点恢复，连续3轮无接受步则停止。

停止调整已经先提交`7bdabd0`。QP只使用全边可行性、固定模型push下限和迭代/停滞预算选择终点，保存模型后才评估图参考；低价值但可行的模型正常停止，随后记为联合失败。原版失败进程在错误结束后进行独立审计，不借参考挑选中间模型；数值修复重跑统一从既定decay终点开始。三阶段评估均在运行结束后进行。没有使用图距离、累计成本或LP multiplier作为训练标签。

## 每条路径及进入终态的边

{table(['seed','阶段','路径','起点估值','记录图参考','Σ正残差','Σ三角余量'],path_rows)}

四条进入终态的物理边成本均0.1；终态自距离为零不代替入口边检查。终态/anchor残差另保存在CSV，逐节点路径值及全339边数组在本目录`seed_*_values.json`、`seed_*_edges.npz`。

{table(['seed','阶段','路径','replay row','入口有符号残差'],entry_rows)}

## 实际预算与复现

{table(['seed','PHR步','降步长步','QP轮','接受步','QP求解次数','候选前向','Gram回退次数','PHR秒','降步长秒','最终QP秒'],cost_rows)}

五个seed共执行125000次primal、125000次dual更新；没有训练重跑。各进程CPU torch线程2，并发执行，表内秒数是各阶段墙钟、不能直接当独占CPU耗时。原版suite墙钟{max(x['seconds_since_suite_start'] for x in process.values()):.1f}秒；最后一列仅计最终QP，原版QP/失败日志仍保留。{'统一数值修复重跑suite墙钟'+format(max(x['seconds_since_suite_start'] for x in final_process.values()),'.1f')+'秒。' if corrected else ''} 独立smoke seed9001执行20+20更新与2轮QP，不计成功率，也未用于调参。

审计重载所有初始化、PHR、decay及正常完成的QP终点，核对数据/代码/checkpoint哈希、Adam步数20k/25k、25k更新序列/学习率、阶段边界模型指标连续、固定push下限、所有接受候选的价值保护、最终全边数组与联合门槛。原三个完整进程还核对已记录的边界Adam/乘子哈希；两个原中断进程没有生成summary，不虚构这项记录，状态保留由共同训练代码及连续性测试验证。相关测试23项通过，见[测试输出](../../validation_final.txt)。旧实验产物均未覆盖。

当前实现可直接从头复现，输出目录须不存在：

```sh
export MPLCONFIGDIR=/tmp/qrl-factorial-mpl OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
PY=../quasimetric-rl/.venv/bin/python
$PY -m pytest experiments/four_path_reproducibility experiments/four_path_refinement/test_refinement.py experiments/four_path_factorial/test_factorial.py -q
$PY -m experiments.four_path_reproducibility.suite --smoke --output results/NEW_RUN/smoke
$PY -m experiments.four_path_reproducibility.suite --output results/NEW_RUN/main
$PY -m experiments.four_path_reproducibility.summarize --root results/NEW_RUN/main
```

本次在保留原运行基础上的数值修复/审计命令：

```sh
$PY -m experiments.four_path_reproducibility.rerun_qp --source results/four_path_reproducibility_20261016/main --output results/four_path_reproducibility_20261016/qp_numerical_fix
$PY -m experiments.four_path_reproducibility.summarize --root results/four_path_reproducibility_20261016/main --corrected results/four_path_reproducibility_20261016/qp_numerical_fix
```

仅检验固定四路径经验图，不外推到新成本尺度或连续环境最优性。源代码与轻量结果进入本次提交；大体积checkpoint和逐步更新日志留在上述本地结果目录，可按命令重新生成。
'''
    (out/'report.md').write_text(report)
    print(json.dumps(dict(original_successful_seeds=original_success,successful_seeds=success,stage_counts=stage_counts),indent=2))


def plot(root,out,rows):
    colors=plt.cm.tab10(np.arange(5))
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for color,seed in zip(colors,SEEDS):
        selected=[r for stage in ('phr','decay','qp') for r in rows if r['seed']==seed and r['stage']==stage]
        for ax,key,label in zip(axes.flat,('max_violation','push_gap_percent','task_nrmse_percent','violation_count'),
                               ('Max physical violation','Graph push gap (%)','Task NRMSE (%)','Violated edges /335')):
            ax.plot(range(len(selected)),[max(r[key],1e-8) if key=='max_violation' else r[key] for r in selected],marker='o',label=str(seed),color=color)
            ax.set(xticks=range(3),xticklabels=['PHR 20k','Decay +5k','QP'],ylabel=label);ax.grid(alpha=.2)
    axes[0,0].set_yscale('log');axes[0,0].axhline(1e-5,color='black',linestyle=':')
    axes[0,1].axhline(1,color='black',linestyle=':');axes[1,0].axhline(2,color='black',linestyle=':')
    for ax in axes.flat:ax.legend(title='Seed',fontsize=8)
    fig.tight_layout();fig.savefig(out/'stages.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for color,seed in zip(colors,SEEDS):
        data=lines(root/f'seed_{seed}/training.jsonl')
        for ax,key,label in zip(axes,('push','max_violation','complementarity'),('Model push','Max physical violation','Mean |lambda * h|')):
            ax.plot([r['total_primal_step'] for r in data],[max(r[key],1e-9) for r in data],label=str(seed),color=color,linewidth=.8)
            ax.set(xlabel='Total primal updates',ylabel=label);ax.axvline(20000,color='black',linestyle=':');ax.grid(alpha=.2)
    axes[1].set_yscale('log');axes[2].set_yscale('log')
    for ax in axes:ax.legend(title='Seed',fontsize=8)
    fig.tight_layout();fig.savefig(out/'training.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='results/four_path_reproducibility_20261016/main')
    parser.add_argument('--corrected')
    args=parser.parse_args();summarize(Path(args.root),Path(args.corrected) if args.corrected else None)
