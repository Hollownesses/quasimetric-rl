**精确缺口梯度：距离与流同时替换，保留原随机守恒与局部风险**

在仓库目录运行，输出目录需尚不存在：

```sh
env PYTHONDONTWRITEBYTECODE=1 ../quasimetric-rl/.venv/bin/python experiments/gap_exact_gradient/run_all.py /private/tmp/qrl-exact-gap-reproduction
```

依赖沿用第一阶段 Python / NumPy / SciPy / Matplotlib 环境，CPU FP64。该命令只新增两组各 20000 步的训练；原无偏基线读取 `results/gap_stochastic_20261108`。不运行上一轮守恒梯度投影，不增加其它训练对照。

在新检出中，先生成第一阶段基线。使用本目录的 `run.py <消融输出目录> --baseline <基线输出目录>`，再依次将同一个消融输出目录传给 `evaluate.py` 和 `finish.py`。三个命令都用第一阶段的 Python 环境，并设置 `OPENBLAS_NUM_THREADS=1` 和 `OMP_NUM_THREADS=1`。基线生成命令见 [第一阶段复现说明](../gap_stochastic/README.md)。

`estimator.py` 计算 `G = COST[:6] @ f - mu @ d[QUERY]`，从原无偏完整梯度中减去原 G² 的两个随机梯度块，再加上 `[-2G mu, 2G COST[:6]]`（距离块仅查询坐标非零）。精确 G 使用已知图成本、既定查询分布、当前模型参数，训练完全不调用最短路或最优流。守恒和风险在每批独立重算核验，确保原随机估计保留。

`audit.py` 在训练之前做有限差分和 batch=2 的 35² 全枚举，检查原总体目标的期望梯度恒等式。`run.py` 将新估计器注入已有 `gap_stochastic.train.run` 回调；原 Adam、投影、学习率、采样器和初始化代码不改动。原随机标量损失中的 gap² 也替换为精确值，使日志与实际梯度一致；损失日志不影响优化器或停止条件。

`evaluate.py` 只在两组训练结束后调用原评估：真实缺口、风险、守恒残差、各查询到达概率、流归一化策略的实际成本和全部冻结判据。`finish.py` 核验来源/初始化/采样摘要、40000 批梯度审计并生成 PNG/PDF 曲线。`run_all.py` 串行执行上述三阶段，限制总墙钟 600 秒。

主结果目录：`results/gap_exact_gradient_20261110`。`manifest.json` 保存环境版本、源文件和基线哈希；`preflight.json` 是训练前核验；两个 `.npz` 保存每 500 步的参数和逐批 `gradient_audit`，列名在 manifest 中；`summary.csv` / `evaluation.json` 为固定终点；`verification.json` / `pipeline.json` 为完整性检查。

唯一预先指定的成功解释见 `PROTOCOL.md`。这是机制对照，未提出新的可扩展随机估计器，未分离距离侧与流侧各自的贡献。
