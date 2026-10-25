"""Only after BOTH fixed budgets complete: common corrected-reference evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from experiments.connected_expert_utrap.compare import evaluate,graph_inputs,graph_structure,decisions
from experiments.connected_expert_utrap.reference import validate_reference
from experiments.connected_expert_utrap.data import load_training_data
from experiments.connected_phr_solver.diagnostic import decompose
from experiments.connected_phr_solver.objective import all_distances
from experiments.four_path_factorial.run import IndexedIQE,file_digest,tensor_digest
from experiments.full_replay_four_path.train import EpochSampler
from .train import equal,write,task_weights,load_initial


def verify(root,protocol,data):
    initial={};final={};hist=Path(protocol['historical_initial']).parent
    history_initial=torch.load(hist/'initial.pt',weights_only=True)
    history_final=torch.load(hist/'final.pt',weights_only=True)
    common_initial=load_initial(protocol)
    audits={};configs={}
    for arm,mode in [('R0','record'),('R1','source')]:
        folder=root/arm;done=json.loads((folder/'COMPLETE').read_text())
        assert done['step']==50000 and not done['reference_loaded']
        assert done['final_sha256']==file_digest(folder/'final.pt')
        cfg=json.loads((folder/'config.json').read_text());configs[arm]=cfg
        signature=dict(protocol_sha256=file_digest(root/'protocol.json'),task_weight_mode=mode,
            weights_sha256=protocol['weights']['weights_sha256'],fingerprints=data.fingerprint)
        assert cfg['training_signature']==signature and done['training_signature']==signature
        initial[arm]=torch.load(folder/'initial.pt',weights_only=True)
        final[arm]=torch.load(folder/'final.pt',weights_only=True)
        for key in ('model','optimizer','dual','visits','sampler','rng','push_counters'):
            assert equal(initial[arm][key],common_initial[key]),(arm,key)
        assert initial[arm]['step']==0 and final[arm]['step']==50000
        assert initial[arm]['initialization_seed']==final[arm]['initialization_seed']==protocol['seed']
        assert final[arm]['training_signature']==signature
        assert final[arm]['fingerprints']==data.fingerprint
        assert tensor_digest(final[arm]['task_weights'])==protocol['weights']['weights_sha256']
        assert torch.equal(final[arm]['model']['nodes'],data.nodes)
        assert final[arm]['sampling_chain_sha256']==done['sampling_chain_sha256']
        stage=torch.load(folder/'phr.pt',weights_only=True);assert stage['step']==45000
        lines=[json.loads(s) for s in (folder/'training.jsonl').read_text().splitlines()]
        audits[arm]=[s for s in lines if 'connected_expert' in s]
        assert [s['step'] for s in audits[arm]]==list(range(0,50001,1000))
        updates=[s for s in lines if 'lr' in s]
        assert [s['step'] for s in updates]==list(range(100,50001,100))
        assert all(s['connected_expert_components']['optimized_pair_rows']==s['connected_expert_components']['pair_rows'] for s in updates)
    for key in ('sampler','visits','rng','push_counters','sampling_chain_sha256'):
        assert equal(final['R0'][key],final['R1'][key]),key
    history_budget_equal=all(equal(final['R0'][key],history_final[key])
        for key in ('sampler','visits','push_counters'))
    if protocol.get('sampler_seed',2307)==2307:
        assert history_budget_equal, 'historical final sampling budget mismatch'
    # Independently reconstruct every ordered batch, including the original partial batches.
    sampler=EpochSampler(len(data.original),protocol.get('sampler_seed',2307));visits=torch.zeros_like(final['R0']['visits']);chain='0'*64
    for _ in range(50000):
        ids=sampler.next(256);visits[ids]+=1;rows=data.original[ids]
        chain=hashlib.sha256(bytes.fromhex(chain)+rows.numpy().tobytes()).hexdigest()
    assert chain==final['R0']['sampling_chain_sha256']
    assert torch.equal(visits,final['R0']['visits']) and equal(sampler.state_dict(),final['R0']['sampler'])
    same_cfg=[k for k in configs['R0'] if configs['R0'][k]!=configs['R1'][k]]
    assert set(same_cfg)=={'training_signature','task_weight_mode'}
    fields=['model','optimizer','dual','visits','sampler','rng','push_counters']
    initial_history_equal=all(equal(initial['R0'][key],history_initial[key]) for key in fields)
    assert initial_history_equal==protocol.get('initial_exact_to_history',True)
    return dict(initial_exact_to_history=initial_history_equal,paired_initial_exact=True,initial_exact_fields=fields,
        final_sampler_visits_rng_counters_exact_between_arms=True,
        final_sampling_budget_exact_to_history=history_budget_equal,
        final_sampler_visits_rng_counters_exact_between_arms_and_history=history_budget_equal and equal(final['R0']['rng'],history_final['rng']),
        entire_ordered_sequence_independently_verified=True,sampling_chain_sha256=chain,
        steps=50000,visits_sum=int(visits.sum()),visits_min=int(visits.min()),visits_max=int(visits.max()),
        audits_per_arm=len(audits['R0']),final_model_exact_between_arms=equal(final['R0']['model'],final['R1']['model']),
        final_dual_exact_between_arms=equal(final['R0']['dual'],final['R1']['dual']),
        r0_final_model_exact_to_history=equal(final['R0']['model'],history_final['model'])),audits


@torch.no_grad()
def decomposition(data,ref,next_edge,path,prediction,starts):
    saved=torch.load(path,weights_only=True)
    model=IndexedIQE(data.small,'encoder',0,initialization_seed=saved['initialization_seed'])
    model.load_state_dict(saved['model']);model.eval()
    pred,edge=all_distances(model,data)
    assert np.array_equal(pred,prediction)
    pieces=decompose(ref,next_edge,data.dst.numpy(),data.cost.double().numpy(),pred,edge)
    unique=data.src[data.original].unique().numpy()
    groups=dict(records=data.src[data.original].numpy(),unique_sources=unique,starts=starts,
        unique_starts=np.unique(starts),longest_quartile=unique[ref[unique]>=np.quantile(ref[unique],.75)])
    aggregate={g:{k:float(v[idx].mean()) for k,v in pieces.items()} for g,idx in groups.items()}
    for g,idx in groups.items():
        aggregate[g]['negative_edge_slack_fraction']=float((pieces['edge_slack'][idx]<0).mean())
    return dict(groups=aggregate,identity_max_error=float(abs(pieces['gap']-pieces['edge_slack']-pieces['triangle_slack']).max()),
        minimum_triangle_slack=float(pieces['triangle_slack'].min())),pieces


def render_report(out,summary):
    results=summary['results'];r0=results['R0'];r1=results['R1'];pair=summary['pairing']
    seed=summary.get('seed',606);sampler_seed=summary.get('sampler_seed',2307)
    conclusion=(
        f"本轮R1的去重源NRMSE为{r1['distance']['unique_sources']['nrmse_percent']:.2f}%（R0为{r0['distance']['unique_sources']['nrmse_percent']:.2f}%），"
        f"最远四分位为{r1['distance']['longest_quartile']['nrmse_percent']:.2f}%（R0为{r0['distance']['longest_quartile']['nrmse_percent']:.2f}%）。"
        f"165独立起点平均路径regret为{r1['decisions']['unique_start_rollouts']['mean_successful_regret']:.2f}（R0为{r0['decisions']['unique_start_rollouts']['mean_successful_regret']:.2f}），"
        f"但最优分支数为{r1['decisions']['optimal_branch_choices']}/58（R0为{r0['decisions']['optimal_branch_choices']}/58），"
        f"最大局部违反为{r1['local']['unique_edges']['max_positive']:.2f}（R0为{r0['local']['unique_edges']['max_positive']:.2f}）。"
        '距离、决策与可行性必须分别判断，不能用单一改善指标概括。')
    lines=['# task push 去重源均衡配对实验（50k 正式终点）','',
        conclusion,'','本轮仅将 task 的经验目标分布从物理记录均匀改为 canonical 源状态均匀。'
        '所有重复记录、轨迹、B1 配对、原始一步成本、PHR 与 dual 时钟均保留。'
        f'R0/R1 都从本次 seed{seed} 的共同零步状态重新训练。historical 为 seed606 的旧模型，仅作背景参考。'
        '训练行为按e06e2ad核对，复用与50452ef相同的原训练器；未继承e162dd3或b7c940f的训练改动。'
        '完成两组预算后才读取保留eb2967d修正的 `results/connected_reference_fix_20261024/main` 参考。','',
        '## 配置、测试与配对核验','',
        f'配置：seed{seed}，batch256，sampler seed{sampler_seed}，B1 seed42，task_mode=all，AdamW weight decay=0，'
        '逐记录 PHR rho100，primal 前残差 eta1 逐访问更新乘子。45k×2e-5，随后5k从2e-6几何下降至2e-7；'
        '每组4线程，每1000步全量审计，固定50k。原采样器保留每epoch最后40条的小批次。',
        '',f"加载得到 N={summary['weights']['N']}、M={summary['weights']['M']}；平均权重={summary['weights']['mean']:.10f}。"
        '全局权重为 N/(M n(s))；minibatch 不除权重和。float32源总权重差异小于6e-7。','',
        f"R0/R1初始模型、优化器、零乘子、访问计数、sampler、RNG 和计数器与本次共同初始状态逐项一致。"
        f"与历史seed606初始状态一致：{pair['initial_exact_to_history']}。"
        f"完整有序采样序列已独立重建核验；两组每行访问{pair['visits_min']}–{pair['visits_max']}次，"
        f"总访问{pair['visits_sum']:,}次，均完成{pair['audits_per_arm']}次全量审计（含0步）。"
        '最终乘子允许独立演化；两组最终sampler/visits/RNG/counters一致，采样预算另由独立重建核验。','',
        '源码测试入口见 `experiments/task_source_balance/test_balance.py`；'
        '包含历史R0更新对照、加权值与梯度、B1/PHR/dual公式、完整预算、初始化及恢复配置拒绝检查。'
        '本轮训练和结果核验记录见本目录输出；不自动沿用其他运行的测试通过声明。','',
        '## 任务距离','',
        '误差与 bias 均相对修正后的图参考，bias=预测−参考；NRMSE=RMSE/RMS(参考)。records 使用物理记录自然重复权重，'
        'unique_sources 每源一次；最远四分位阈值只用于事后评价。','',
        '| 分组 | 模型 | MAE | RMSE | NRMSE % | bias |','|---|---|---:|---:|---:|---:|']
    names={'records':'42,536记录','unique_sources':'6,766去重源','longest_quartile':'最远四分位','unique_starts':'165独立起点','starts':'307原始起点'}
    for group,label in names.items():
        for arm in ('historical','R0','R1'):
            d=results[arm]['distance'][group]
            lines.append(f"| {label} | {arm} | {d['mae']:.4f} | {d['rmse']:.4f} | {d['nrmse_percent']:.4f} | {d['bias']:.4f} |")
    lines+=['','## 分支与完整路径决策','','所有模型使用相同真实候选边，按 cost+预测后继距离贪心选择；'
        '严格参考排序排除参考并列，预测并列不算正确。分支最优容差1e-6。','',
        '| 模型 | 最优分支/58 | 排序正确率 | 分支平均regret | 165起点路径regret | 165最优路径数 | 307起点路径regret | 307到达数 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ('historical','R0','R1'):
        d=results[arm]['decisions'];u=d['unique_start_rollouts']
        lines.append(f"| {arm} | {d['optimal_branch_choices']} | {d['pairwise_order_accuracy']:.4%} | {d['mean_branch_regret']:.4f} | {u['mean_successful_regret']:.4f} | {u['optimal_rollouts']} | {d['mean_successful_regret']:.4f} | {d['graph_successes']} |")
    lines+=['','该图为DAG，唯一无出边节点是目标，最长可选路径275步，小于评价上限300。'
        '307/307到达主要由结构保证，仅为完整性检查，不能作为学会导航的证据。','',
        '## 局部约束违反','','局部残差为 d(s,s′)−原始一步成本；严格违反阈值为1e-5，报告的最大/平均正违反取relu残差、不截断该容差。','',
        '| 模型 | 分布 | 违反数/总数 | 最大正违反 | 平均正违反 |','|---|---|---:|---:|---:|']
    for arm in ('historical','R0','R1'):
        for group in ('records','unique_edges'):
            d=results[arm]['local'][group]
            lines.append(f"| {arm} | {group} | {d['violations']}/{d['count']} | {d['max_positive']:.6f} | {d['mean_positive']:.6f} |")
    lines+=['','## 长程误差分解','','沿固定图参考最短路径，以实际模型边距离构造：'
        '`D*−J = Σ(cost−d_edge) + [Σd_edge−J]`。前项是有符号累计边项，可能因局部违反为负；'
        '后项是累计三角松弛。误差变小不能自动解释为三角松弛变小。此恒等式分解有符号的逐节点误差及其均值，不是RMSE的可加分解。','',
        '| 分组 | 模型 | 平均 D*−J | 累计边项 | 累计三角松弛 |','|---|---|---:|---:|---:|']
    for group in ('records','unique_sources','longest_quartile','unique_starts','starts'):
        for arm in ('historical','R0','R1'):
            d=results[arm]['decomposition']['groups'][group]
            lines.append(f"| {names[group]} | {arm} | {d['gap']:.4f} | {d['edge_slack']:.4f} | {d['triangle_slack']:.4f} |")
    lines+=['','## R1−R0 的直接判读','']
    for group in ('unique_sources','longest_quartile','unique_starts','records'):
        a=r0['distance'][group];b=r1['distance'][group];delta=b['nrmse_percent']-a['nrmse_percent']
        lines.append(f"- {names[group]}任务距离 NRMSE {'下降' if delta<0 else '上升'} {abs(delta):.4f} 个百分点；MAE变化 {b['mae']-a['mae']:+.4f}，bias变化 {b['bias']-a['bias']:+.4f}。")
    a=r0['decisions'];b=r1['decisions']
    lines.append(f"- 最优分支 {a['optimal_branch_choices']}→{b['optimal_branch_choices']}；排序正确率变化 {(b['pairwise_order_accuracy']-a['pairwise_order_accuracy'])*100:+.4f} 个百分点；分支平均regret变化 {b['mean_branch_regret']-a['mean_branch_regret']:+.4f}。")
    lines.append(f"- 165独立起点完整路径平均regret变化 {b['unique_start_rollouts']['mean_successful_regret']-a['unique_start_rollouts']['mean_successful_regret']:+.4f}；不能以距离误差代替路径决策指标。")
    lines.append(f"- 最大局部正违反 {r0['local']['unique_edges']['max_positive']:.6f}→{r1['local']['unique_edges']['max_positive']:.6f}；去重边平均正违反 {r0['local']['unique_edges']['mean_positive']:.6f}→{r1['local']['unique_edges']['mean_positive']:.6f}。")
    for group in ('longest_quartile','unique_starts'):
        a=r0['decomposition']['groups'][group];b=r1['decomposition']['groups'][group]
        lines.append(f"- {names[group]}的平均低估 D*−J 变化 {b['gap']-a['gap']:+.4f} = 累计边项变化 {b['edge_slack']-a['edge_slack']:+.4f} + 累计三角松弛变化 {b['triangle_slack']-a['triangle_slack']:+.4f}。"
            +('三角松弛确有下降；同时须结合边项变化判断贡献。' if b['triangle_slack']<a['triangle_slack'] else '三角松弛未下降，不能将误差改善归因为三角松弛下降。'))
    a=r0['decomposition']['groups']['longest_quartile'];b=r1['decomposition']['groups']['longest_quartile']
    lines+=['',f"最远四分位累计三角松弛从{a['triangle_slack']:.4f}变为{b['triangle_slack']:.4f}，累计边项从{a['edge_slack']:.4f}变为{b['edge_slack']:.4f}。"
        f"两项变化分别为{b['edge_slack']-a['edge_slack']:+.4f}与{b['triangle_slack']-a['triangle_slack']:+.4f}，"
        '需结合本seed的bias与局部违反判断改善或恶化；不能预设其他seed复现历史606的方向。'
        '这不是对RMSE的可加归因，也不等于独立识别了优化过程中的因果中介。',
        '',f"R1的去重源bias为{r1['distance']['unique_sources']['bias']:+.4f}，最远四分位bias为{r1['distance']['longest_quartile']['bias']:+.4f}。"
        f"记录MAE为{r1['distance']['records']['mae']:.4f}（R0为{r0['distance']['records']['mae']:.4f}），165起点最优完整路径数为{r1['decisions']['unique_start_rollouts']['optimal_rollouts']}（R0为{r0['decisions']['unique_start_rollouts']['optimal_rollouts']}）。"
        '因此平均regret、排序正确率与精确最优选择数给出的结论并不相同。']
    lines+=['','## 45k 事后过程与历史复核','','50k始终是正式终点，下表不用于选择checkpoint。','',
        '| 模型/步数 | 去重源 NRMSE % | 最远四分位 NRMSE % | 最优分支 | 最大违反 |','|---|---:|---:|---:|---:|']
    for arm in ('historical','R0','R1'):
        for step,result in [(45000,summary['stage45000'][arm]),(50000,results[arm])]:
            lines.append(f"| {arm}/{step} | {result['distance']['unique_sources']['nrmse_percent']:.4f} | {result['distance']['longest_quartile']['nrmse_percent']:.4f} | {result['decisions']['optimal_branch_choices']} | {result['local']['unique_edges']['max_positive']:.6f} |")
    lines+=['',f"R1的45k→50k去重源NRMSE为{summary['stage45000']['R1']['distance']['unique_sources']['nrmse_percent']:.4f}%→{r1['distance']['unique_sources']['nrmse_percent']:.4f}%，"
        f"最优分支数为{summary['stage45000']['R1']['decisions']['optimal_branch_choices']}→{r1['decisions']['optimal_branch_choices']}；无论过程指标是否更好，均保留50k为正式终点。",
        '',f"R0最终模型与历史逐位一致：{pair['r0_final_model_exact_to_history']}。"
        '新seed不要求与历史seed606参数一致；四线程也不承诺同seed逐位复现。'
        '原50k仅作历史复核，配对比较以本轮同seed的R0/R1为准。','',
        '## 范围与复现','',f'本报告为单seed{seed}、有限固定训练图、4线程；结果不证明多seed稳定性、连续环境最优性、'
        '未见状态泛化或MPPI闭环能力。不得把共享后缀记录视为独立统计样本。'
        '不追加预算、不搜索配置、不改变研究问题。','',
        '可复现命令与公式见仓库 `experiments/task_source_balance/PROTOCOL.md`；'
        '冻结配置 `../protocol.json`，源码快照 `../source_snapshot/`，完整数值 `summary.json`，'
        '逐分支/逐轨迹决策 `R0_decisions.json` / `R1_decisions.json`，节点分解及预测 `arrays.npz`。',
        '', '![统一比较](comparison.png)','']
    (out/'report.md').write_text('\n'.join(lines))


def run(root):
    root=Path(root);protocol=json.loads((root/'protocol.json').read_text());torch.set_num_threads(1)
    # Gate all graph/reference access on both completed fixed budgets.
    for arm in ('R0','R1'):
        assert json.loads((root/arm/'COMPLETE').read_text())['step']==50000
    for path,digest in protocol['source_files'].items():assert file_digest(path)==digest,path
    data=load_training_data(protocol['data']);weights,stats=task_weights(data)
    assert data.fingerprint==protocol['fingerprints'] and stats==protocol['weights']
    pairing,logs=verify(root,protocol,data)
    graph,starts=graph_inputs(data);structure=graph_structure(data,graph,starts)
    assert structure['acyclic'] and structure['longest_possible_recorded_rollout']==275
    assert structure['branches']==58 and structure['unique_starts']==165 and len(starts)==307
    refroot=Path('results/connected_reference_fix_20261024/main');info=json.loads((refroot/'summary.json').read_text())
    assert info['passed'] and info['fingerprints']==data.fingerprint
    assert file_digest(refroot/'graph_reference_audit_only.npz')==info['reference_sha256']
    with np.load(refroot/'graph_reference_audit_only.npz') as f:
        ref=f['distance'].copy();next_edge=f['next_edge'].copy()
        for key,expected in [('source',data.src),('target',data.dst),('physical',data.physical)]:assert np.array_equal(f[key],expected.numpy())
    validate_reference(data,ref)
    oracle,_=decisions(data,graph,ref,ref,starts,300)
    assert oracle['summary']['optimal_branch_choices']==58 and oracle['summary']['mean_successful_regret']<1e-8
    out=root/'comparison';out.mkdir(exist_ok=False)
    results={};stages={};arrays={};details={};hist=Path(protocol['historical_initial']).parent
    for arm,folder in [('historical',hist),('R0',root/'R0'),('R1',root/'R1')]:
        for step,name in [(50000,'final.pt'),(45000,'phr.pt')]:
            path=folder/name;before=file_digest(path)
            result,pred,detail=evaluate(data,graph,ref,starts,path)
            result['decisions']['strict_reference_pairs']=sum(b['strict_reference_pairs'] for b in detail['branches'])
            result['decisions']['correctly_ordered_pairs']=sum(b['correctly_ordered_pairs'] for b in detail['branches'])
            diag,pieces=decomposition(data,ref,next_edge,path,pred,starts);result['decomposition']=diag
            assert file_digest(path)==before
            if step==50000:
                results[arm]=result;details[arm]=detail;write(out/(arm+'_decisions.json'),detail)
                arrays[arm+'_prediction']=pred;arrays.update({arm+'_'+k:v for k,v in pieces.items()})
            else:stages[arm]=result
            print(json.dumps(dict(arm=arm,step=step,distance=result['distance']['unique_sources'],decisions=result['decisions'])),flush=True)
    branch_changes=[]
    for a,b in zip(details['R0']['branches'],details['R1']['branches']):
        assert a['source']==b['source']
        if a['chosen_row']!=b['chosen_row']:
            branch_changes.append(dict(source=a['source'],R0_choice=a['chosen_row'],R1_choice=b['chosen_row'],R0_regret=a['regret'],R1_regret=b['regret']))
    summary=dict(seed=protocol['seed'],sampler_seed=protocol.get('sampler_seed',2307),
        results=results,stage45000=stages,pairing=pairing,weights=stats,graph_structure=structure,
        fingerprints=data.fingerprint,protocol_sha256=file_digest(root/'protocol.json'),reference_sha256=info['reference_sha256'],
        reference_correction=info['changed'],branch_changes=branch_changes,oracle=oracle['summary'],
        evaluation_source_sha256=file_digest(__file__),training=logs,
        official_endpoint=50000,scope='single seed; finite fixed graph; four-thread training; changed task empirical measure')
    write(out/'summary.json',summary);np.savez_compressed(out/'arrays.npz',reference=ref,starts=starts,**arrays)
    shutil.copy2(__file__,out/'analysis_source.py')
    plot(out,summary);render_report(out,summary)
    write(out/'COMPLETE',dict(summary_sha256=file_digest(out/'summary.json'),report_sha256=file_digest(out/'report.md'),pairing=pairing))


def plot(out,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    results=summary['results'];arms=['historical','R0','R1'];x=np.arange(3)
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for j,key in enumerate(('unique_sources','longest_quartile','unique_starts')):
        axes[0,0].bar(x+(j-1)*.25,[results[a]['distance'][key]['nrmse_percent'] for a in arms],width=.25,label=key)
    axes[0,0].set(xticks=x,xticklabels=arms,ylabel='Task NRMSE (%)');axes[0,0].legend(fontsize=8)
    axes[0,1].bar(x,[results[a]['decisions']['optimal_branch_choices'] for a in arms]);axes[0,1].set(xticks=x,xticklabels=arms,ylabel='Optimal branches / 58',ylim=(0,58))
    axes[0,2].bar(x,[results[a]['decisions']['unique_start_rollouts']['mean_successful_regret'] for a in arms]);axes[0,2].set(xticks=x,xticklabels=arms,ylabel='165-start mean path regret')
    for a in ('R0','R1'):
        logs=summary['training'][a];axes[1,0].plot([r['step'] for r in logs],[r['connected_expert']['max_violation'] for r in logs],label=a)
    axes[1,0].set(xlabel='Update',ylabel='Maximum local violation');axes[1,0].legend()
    for j,key in enumerate(('edge_slack','triangle_slack','gap')):
        axes[1,1].bar(x+(j-1)*.25,[results[a]['decomposition']['groups']['longest_quartile'][key] for a in arms],width=.25,label=key)
    axes[1,1].axhline(0,color='k',lw=.6);axes[1,1].set(xticks=x,xticklabels=arms,ylabel='Farthest-quartile signed decomposition');axes[1,1].legend(fontsize=8)
    for a in ('R0','R1'):
        logs=summary['training'][a];axes[1,2].plot([r['step'] for r in logs],[r['connected_expert']['mean_positive'] for r in logs],label=a)
    axes[1,2].set(xlabel='Update',ylabel='Record mean positive violation');axes[1,2].legend()
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=180);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);run(p.parse_args().root)
