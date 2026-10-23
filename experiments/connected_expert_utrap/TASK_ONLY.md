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
