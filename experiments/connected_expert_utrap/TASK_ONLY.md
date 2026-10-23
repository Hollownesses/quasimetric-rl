# 联通专家图 task-only QRL 冻结协议（2026-10-24）

目的：区分 B1 共同优化的影响与 task-only PHR 自身的困难。唯一训练变量是
B1 系数从 0.4 变为 0；task 保持 0.6，不能重分配 B1 权重。保留原 B1
索引、距离查询和采样流程；日志 pair_rows 是查询数，不代表非零梯度。

固定数据 `results/connected_expert_utrap_20261021/main`，42536 条物理记录、
6958 条去重边；共享 encoder/projector/IQE、seed606、batch256、4线程、
记录无放回采样 seed2307，重复片段保留。逐记录 PHR rho100、访问时用
更新前残差以 eta1 更新乘子；AdamW、零 weight decay；45k 步 lr2e-5，
5k 步 lr2e-6 几何下降到2e-7。共50k步固定停止，无 QP、Oracle 或图距离标签。

```sh
MPLCONFIGDIR=/tmp/qrl-oct24 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
../quasimetric-rl/.venv/bin/python -m experiments.full_replay_four_path.train \
  --connected-expert-data results/connected_expert_utrap_20261021/main \
  --output results/connected_expert_task_only_20261024/main/train \
  --seed 606 --phr-steps 45000 --decay-steps 5000 --batch 256 \
  --threads 4 --audit-interval 5000 --task-mode all --push-mode task-only
```

每5000步扫描全边、检查数值有限性、乘子、访问次数与日志更新；不按参考误差
早停或选 checkpoint。参考修正单独完成；训练不读取参考。审计频率相较既有
1000步改为5000步，只读扫描不消耗训练 RNG。初始模型与基线逐位比较，
终点校验采样器及访问次数；多线程训练不承诺跨运行逐位一致。

结束后统一评估现有 task+B1 QRL、task-only QRL 和另一 worktree 的 optimality
TD。统一使用第0项修正参考，报告记录/去重 task NRMSE、起点误差、局部违反、
58个分支的选择/成本后悔及记录图贪心 rollout。图内恢复不等于连续状态或 MPPI
闭环成功。TD 有后继评估和 target network，更新预算相同但不声称算力相同。

测试：`python -m pytest experiments/connected_expert_utrap/test_task_only.py
experiments/connected_expert_utrap/test_connected.py
experiments/expert_utrap_network/test_training_adapter.py
experiments/witnessed_task_push/test_witnessed.py -q`。
仅提交代码、测试、协议、最终报告及引用图片；原始训练产物保留本地。

## 监督和统一评估

监督器每20秒读取已写日志，遇到新5000步扫描时记录监督时间和完整扫描。
它检查数值有限性、QRL 的 B1 优化行数为零、记录访问次数差不超过1，以及
两组终态/anchor为零；不会改变训练配置。原始监督记录保留本地。

```sh
python -m experiments.connected_expert_utrap.monitor \
  --qrl-task results/connected_expert_task_only_20261024/main/train \
  --td ../quasimetric-rl-td-comprare/results/connected_expert_td_20261024/main/train \
  --output results/connected_expert_task_only_20261024/main/supervision.jsonl --watch
```

两组固定预算均完成后执行（输出目录必须不存在）：

```sh
MPLCONFIGDIR=/tmp/qrl-oct24 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
../quasimetric-rl/.venv/bin/python -m experiments.connected_expert_utrap.compare \
  --data results/connected_expert_utrap_20261021/main \
  --reference results/connected_reference_fix_20261024/main \
  --qrl-b1 results/connected_expert_qrl_20261023/main/train \
  --qrl-task results/connected_expert_task_only_20261024/main/train \
  --td ../quasimetric-rl-td-comprare/results/connected_expert_td_20261024/main/train \
  --output results/connected_expert_task_only_20261024/main/comparison
```

评估前严格核对三组初始模型/优化器/采样器/RNG、最终记录访问与采样器，
以及TD全部候选图哈希。NRMSE定义为RMSE/RMS(reference)，局部违反阈值1e-5，
分支最优容差1e-6；相同分数按最早代表记录破平局。额外报告当前task Bellman
残差与task单边不等式，区分任务potential和全部IQE边可行性。45k及50k都在
训练结束后评估，不据此选择停止时间。参考贪心应有58/58最优分支和307/307
零regret图rollout，作为评估器自检。

补充图结构审计：检查有向环和最长可能记录路径，报告307条轨迹起点中165个
不重复节点的独立统计。该图无环且最长路径275步，因此300步图rollout的到达率
不能单独证明学习成功；分支regret、完整路径regret及距离误差才有区分力。

最终评估后可运行 `python -m experiments.connected_expert_utrap.diagnose
--data results/connected_expert_utrap_20261021/main
--comparison results/connected_expert_task_only_20261024/main/comparison`，复算task-only
45k→50k分支翻转，并以参考单步advantage沿实际所选记录路径求和，核对完整路径
regret的精确分解。此分析不改变模型、停止点或训练协议。
