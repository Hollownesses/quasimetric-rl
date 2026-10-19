# 专家网络接入既有完整 replay 训练器

入口仍为 `experiments.full_replay_four_path.train`。新增可选参数 `--expert-data`：传入上一轮冻结的数据目录时，只加载该专家网络；不传时保持原始 200,000 replay＋四路径实验的行为。此前构造报告与 README 是数据构造阶段的记录，当前训练接入以本文为准；原报告、数据及源码快照不改写。

仓库根目录启动正式训练：

```sh
MPLCONFIGDIR=/tmp/qrl-expert-mpl ../quasimetric-rl/.venv/bin/python \
  -m experiments.full_replay_four_path.train \
  --expert-data results/expert_utrap_network_20261019/main \
  --output results/expert_utrap_qrl_20261019/main/train \
  --seed 606 --phr-steps 20000 --decay-steps 5000 \
  --batch 256 --threads 4 --audit-interval 1000 --task-mode all
```

输出目录必须尚不存在。恢复时重复相同命令并追加：

```sh
--resume results/expert_utrap_qrl_20261019/main/train/latest.pt
```

预算、batch、threads、seed、task mode 和数据指纹必须保持一致；新专家 checkpoint 还校验完整训练目标签名，禁止与旧 union checkpoint 混用。恢复包含模型、AdamW、逐记录 dual、采样顺序/游标/RNG、访问次数和 push 计数。4 线程继承旧实验的浮点非确定性；单线程恢复测试可逐位比较，不能据此保证 4 线程逐位复现。

| 配置 | 旧 full_replay_four_path_20261017 | 本次专家网络 |
|---|---|---|
| 初始化 | seed=606，生产共享 encoder/projector/IQE | 相同，从头初始化 |
| 优化器 | AdamW，weight_decay=0，默认 betas/eps | 相同 |
| 主阶段 | 20,000 步，lr=2e-5 | 相同 |
| 降学习率 | 5,000 步，2e-6 指数降至 2e-7 | 相同 |
| PHR | rho=100，逐记录独立 dual，eta=1，更新前残差 | 相同 |
| task/B1 | 0.6 task(all)＋0.4 固定同轨迹 B1，B1 seed=42 | 相同；本数据 all/witnessed 等价 |
| minibatch | 256 条原始物理边，无放回，轮末较小 batch | 256 条专家物理记录，规则相同 |
| 来源权重 | 0.5 原始 batch＋0.5 全部四路径，每步都含旧四路径 | 单一专家来源，权重 1.0；不混入旧 replay/四路径 |
| 终态 anchor | 原始 anchor quotient 恒等，扫描保留 | 64 个合法 anchor，恒等扫描保留、不采样优化 |
| 全量扫描 | 每 1,000 步及阶段/最终边界 | 相同，扫描本数据全部 12,013 行 |
| 线程/停止 | CPU 4 线程，固定预算，不按参考选模型 | 相同 |

学习率、rho、dual step、push 权重沿用原脚本内的固定实现，不需要也不支持另传不存在的 CLI 参数。

唯一必要的目标结构变化是把原来两个数据来源的 50/50 混合换成单来源专家数据。训练保留 11,949 条物理记录（2,537 条去重物理边），不会静默去重或把共享片段加权改掉。较小数据在相同 25,000 更新预算下，每条记录被访问更频繁，因此这是相同优化设置与更新预算，不是相同单边曝光预算。

`training_adapter.py` 将构造加载器按物理行保存的 task/B1 索引，映射为原训练器使用的全 replay 行索引，防止 interleaved anchors 导致错位。启动审计验证文件指纹、动力学、一步成本、合法 anchor、每个 context 的成功链及 B1 边界；不会读取 `audit/graph_reference_audit_only.npz` 或其他图距离/Oracle 标签。训练日志使用 `expert`、`expert_visits` 和 `expert_*` push 计数，避免把单来源数据误标成旧 original/four-path 分层。

结果包括 `config.json`、`data_audit.json`、`initial.pt`、`latest.pt`、`phr.pt`、`final.pt`、`training.jsonl`、`final_constraints.npz` 和 `COMPLETE`。COMPLETE 只表示固定训练预算完成，不表示约束可行或距离/导航恢复。环境成本、任务终态规范、网络架构和 MPPI 未变。

接入验证：

```sh
MPLCONFIGDIR=/tmp/qrl-expert-mpl ../quasimetric-rl/.venv/bin/python -m pytest -q \
  experiments/expert_utrap_network/test_training_adapter.py \
  experiments/full_replay_four_path/test_full_replay.py \
  experiments/witnessed_task_push/test_witnessed.py
```

测试覆盖真实 11,949 行数据索引/启动审计、单来源完整权重、更新前残差 dual、短步训练/精确恢复、配置拒绝，以及旧 union 与冻结旧代码的逐位更新回归。短步测试使用临时目录，不启动上面的正式 25,000 步实验。
