# 原生 MQE 对照：公共环境一致性与固定预算

2026-10-29制定。目的为将已有多步准度量方法纳入比较，不将E2的IQE消融称为MQE复现，也不将“多步”作为新贡献。保持pointmaze-medium-navigate-v0；暂不扩展stitch或其他算法。

## 固定代码与环境

- MQE官方仓库WJ2003B/mqe-release，commit 04c2697405233e4313fab43e05bd4d95f1b3e111。直接运行该版impls/main.py、agents/mqe.py及utils/datasets.py；保留双MRN、完整B×B备份、动作不变性、目标/waypoint采样和actor联合学习。没有套用E2的IQE或16目标估计。
- 用同一官方OGBench环境commit a977253ce5b88434f4e4f531c50047cc95993a51及原始数据；MQE的requirements.txt本就建议安装外部OGBench。其仓库内附带的ogbench/locomaze/maze.py额外加入动作噪声、初始化强制渲染，不用于本公平对照。这是显式环境依赖选择，不能把两套环境的结果混称相同协议。
- 首次Git clone连接重置，改用GitHub连接器读取固定tree和源码。SOURCE_BLOBS锁定90个运行相关文件的Git blob SHA1，68个与现有官方源码逐字节相同的文件复用，22个从固定官方ref读取；所有90个均核验。未获取其他环境的二进制/纹理资源，不支持据此宣称已安装全仓库的所有任务。标准git clone同commit亦可通过相同校验。
- 复用现有独立CPU JAX虚拟环境，只使用依赖，不安装、升级或修改包。运行时显式导入MQE impls与上述官方OGBench，两者来源均断言；每轮保留pip freeze与实际设备。没有CUDA。
- 必要适配仅为本地dataset_dir、离线W&B、video_episodes=0及macOS CGL预加载。原生main硬编码Linux EGL，第一次smoke在MuJoCo导入前失败，原目录/日志保留；使用全新目录重试。CGL仅处理渲染绑定初始化，不改动力学或评价成功条件。

## 参数与预算

- seed0，从头初始化。alpha=0.03取论文arXiv:2511.07730v1附录表3的pointmaze值，非代码默认0.1；gamma0.995、lambda0.95、next-state概率0.2、batch256、512维三层网络及8个MRN组件与表2一致。
- diag_backup固定代码默认0.5，t=5.0，lr3e-4；论文表2/3未提供pointmaze的diag_backup，代码注释建议locomotion 0.2–0.4但没有逐任务脚本。该差异透明记录，不依据task3结果搜索参数，也不宣称完整论文超参数已全部确定。
- 先做独立两次更新smoke：完整batch256，每步记录损失，第二步保存checkpoint；第1/2步均调用原生评价器，全五任务每任务1个完整episode，视频0。通过后验证优化器计数、全状态有限性、完整保存加载、actor输出与原生采样边界。
- 正式方法比较预定100,000更新，每1000步日志，每10,000步全五任务各50episode评价并保存。只将100k终点作为结果，不挑最好、按成功率停止或延长。100k与当前QRL短预算一致，但batch不同，源转移数和耗时分别报告，不能称同计算量。
- 此为原生MQE在当前公共环境的100k对照，不是完整论文结果复现。原论文状态版为1M并对末三次评价跨8seed汇总；完整1M命令另提供，不因中间成功率自行改变预算。本协议并不替代已运行的官方QRL 1M核验。

## 分析与后续

正式终点需同时保留官方actor全五任务50episode结果、同固定25例真实后继V贪心与距离场（分别报告两ensemble及预先规定的保守聚合max距离），避免只以actor改变来解释距离收益。原生Q控制可作另列诊断，但不能替换V下降通道主要问题。尺度采用log-discount单位，除以-log(gamma)只作单位转换，不能直接证明最短步数校准。

MQE对比改变了表示、价值目标、目标采样及策略联合学习等多个变量，是方法参照；只有E2内部one/multi可以分离备份长度。稳定改善仍需独立训练采样/初始化和actor提取核验。失败必须保留日志与实际阻塞，不缩数据、换任务、改batch或静默修改算法。

来源：[固定官方源码](https://github.com/WJ2003B/mqe-release/tree/04c2697405233e4313fab43e05bd4d95f1b3e111)、[论文附录B.3/B.4](https://arxiv.org/html/2511.07730v1#A2.SS3)。
