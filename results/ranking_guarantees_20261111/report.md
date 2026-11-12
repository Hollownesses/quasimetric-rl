**平方松弛下的必然反序、安全条件与统一保证的不可能性**

2026-11-11。本文给出新的推导与有限凸问题核验，没有新增神经网络训练或环境回合。目标是区分某个可行场允许反序、原目标的全局最优解必然反序，以及足以排除反序的条件。

核心结果有三项。在一个四节点分叉图上，可以得到所有全局最优场必然反序的闭式充要条件；在一般有限图上，可以得到依赖动作间隔和最优后缀覆盖的安全条件；缺少统一间隔或边覆盖下界时，任何固定正平方松弛预算都不能保证所有问题的动作正确。这些结论可以同时成立，并不意味着每一个问题只要有正松弛就一定失败。

**问题、符号与适用范围。**

考虑有限强连通确定性图，边成本 $c_e>0$，边采样概率 $p_e>0$ 且 $\sum_ep_e=1$。$\mathcal Q$ 是全部有向准度量函数，允许不同点距离为零。线性推远问题为

$$
V_\varepsilon=\max_{d\in\mathcal Q}\Phi(d),\qquad
\Phi(d)=\sum_{u,g}\mu_{ug}d(u,g),\qquad
\sum_ep_e[d(e)-c_e]_+^2\le\varepsilon^2.
\tag{1}
$$

查询质量非负并归一化。这里 $\varepsilon$ 是预算平方根，OG 当前实现的预算 0.05 对应 $\varepsilon=\sqrt{0.05}$。

决策使用真实首步成本和真实后继

$$
\widehat Q(s,a,g)=c(s,a)+d(s'_a,g).
\tag{2}
$$

讨论的是式 (2) 的动作排序，不包含潜在动力学误差、actor 拟合误差或 MPPI 候选缺失。原论文的硬约束恢复定理、逐边松弛近似定理和实践平均平方约束不是同一个问题。[QRL 原论文，定理 2、3 与式 (12)](https://proceedings.mlr.press/v202/wang23al/wang23al.pdf)

令 $D_w(u,g)$ 为边成本 $w$ 下的最短路距离，$D_c$ 为真值。若动作 $a$ 的真实成本比 $b$ 低，定义

$$
\Delta=\big[c(s,b)+D_c(s'_b,g)\big]-\big[c(s,a)+D_c(s'_a,g)\big]>0.
$$

严格反序指 $\widehat\Delta=\widehat Q(b)-\widehat Q(a)<0$。$\widehat\Delta=0$ 是平局，它不算严格反序，但任意平局处理规则不再保证选择真实最优动作。

**引理。完整函数类的全局最优查询值等于某个膨胀图的距离。**

对任意可行 $d$，令 $x_e=[d(e)-c_e]_+$。三角不等式给出 $d\le D_{c+x}$，并且 $D_{c+x}(e)\le c_e+x_e$，所以 $D_{c+x}$ 仍满足同一预算。若 $d$ 是全局最优，且 $\mu_{ug}>0$，则必有

$$
d(u,g)=D_{c+x}(u,g)\ge D_c(u,g).
\tag{3}
$$

否则用 $D_{c+x}$ 替换 $d$ 会严格提高推远目标。这一步依赖完整准度量类，也依赖被比较查询有正质量。它对严格递增的 shaped 推远同样成立。

由此前的流对偶推导，有

$$
V_\varepsilon
=\min_{f\in\mathcal F_\mu}\left[c^\top F+\varepsilon\|F/\sqrt p\|_2\right],
\qquad
x_e^*=\frac{\varepsilon F_e^*}{p_e\|F^*/\sqrt p\|_2}.
\tag{4}
$$

$f$ 是逐目标守恒流，$F$ 为聚合流。对给定非零最优流，椭球的支撑点 $x^*$ 唯一。任意最优 $x$ 和该最优流都必须达到极小极大等式，所以所有最优距离场在正质量查询上等于同一个 $D_{c+x^*}$。下面的“所有全局最优场”只要求相关查询值唯一，不声称整个距离矩阵唯一。[前序对偶推导](/Users/z./Documents/博士/本科毕设/quasimetric-rl-mechanism-diagnostics/results/objective_flow_20261105/report.md)

**命题 1。固定后缀路径时的精确反序关系。**

若两个动作后的路径被图结构强制沿 $P_a,P_b$ 到共同汇合点，之后共用同一后缀，则膨胀图上的共同后缀完全抵消，得到

$$
\widehat\Delta
=\Delta-\frac{\varepsilon}{N^*}
\left(\sum_{e\in P_a}\frac{F_e^*}{p_e}
-\sum_{e\in P_b}\frac{F_e^*}{p_e}\right),
\qquad N^*=\|F^*/\sqrt p\|_2.
\tag{5}
$$

因此，括号内的便宜后缀相对流量负载超过 $\Delta N^*/\varepsilon$ 时，必然严格反序。它说明要比较的是两个后缀的膨胀差，不是某一条路径上存在正违反就足够。

式 (5) 仍包含最优流，主要适合作为求解后的机制诊断。它假设后缀没有可绕行分支，不能对一般多路径图固定一条真实最短路后直接声称等号成立。下一命题把最优流消掉，得到训练前可计算的阈值。

**命题 2。四节点分叉图上的闭式充要条件。**

图有节点 $s,a,b,g$，边为 $s\to a$、$a\to g$、$s\to b$、$b\to g$、$g\to s$。成本依次为 $1,1,1,1+\Delta,1$。最后一条边只用于强连通，到达目标即停止。因此两个首动作的真实总成本为 $2$ 和 $2+\Delta$。

目标固定为 $g$，查询质量为

$$
\mu_{sg}=r>0,\quad\mu_{ag}=A>0,\quad\mu_{bg}=B>0,\qquad r+A+B=1.
$$

边频率先取完全均匀 $p_e=1/5$。定义

$$
\kappa(r,A,B)=
\frac{\sqrt5\,(A-B)}
{\sqrt{r^2/2+(A+r/2)^2+(B+r/2)^2}}.
\tag{6}
$$

则对于式 (1) 的每一个全局最优距离场，严格有

$$
\boxed{
\begin{array}{lll}
\Delta<\varepsilon\kappa &\Longleftrightarrow& \widehat\Delta<0,\\
\Delta=\varepsilon\kappa &\Longleftrightarrow& \widehat\Delta=0,\\
\Delta>\varepsilon\kappa &\Longleftrightarrow& \widehat\Delta>0.
\end{array}}
\tag{7}
$$

当 $A\le B$ 时，$\kappa\le0$，所以对 $\Delta>0$ 不会出现反序，无论预算有多大。当 $A>B$ 时，临界平方预算为

$$
\boxed{\eta_{\rm crit}=\varepsilon_{\rm crit}^2=\Delta^2/\kappa^2.}
\tag{8}
$$

超过这个阈值，全部全局最优场都反序；等于它时平局；低于它时排序正确。这比“可以构造一个错误可行场”更强。

证明如下。令 $t\in[0,1]$ 表示从 $s$ 出发的需求沿便宜分支的比例。其余两个源的路线没有选择。按上述五条边的顺序，全部无环可行流为

$$
F(t)=\big(rt,\ A+rt,\ r(1-t),\ B+r(1-t),\ 0\big).
$$

最优流是以下一维问题的解

$$
\min_{t\in[0,1]}H(t),\qquad
H(t)=C_0-r\Delta t+\varepsilon N(t),\qquad
N(t)=\sqrt{\sum_eF_e(t)^2/p_e}.
\tag{9}
$$

因为两条首边流量和恒为 $r>0$，不同 $t$ 对应的流不可能互为正比例，故 $N(t)$ 严格凸，$\varepsilon>0$ 时 $t^*$ 唯一。

暂时允许一般正边频率，并定义

$$
D(t)=\frac{A+rt}{p_{ag}}-\frac{B+r(1-t)}{p_{bg}},\qquad
K(t)=\frac{rt}{p_{sa}}-\frac{r(1-t)}{p_{sb}}.
$$

则

$$
H'(t)=r\left[-\Delta+\varepsilon\frac{D(t)+K(t)}{N(t)}\right],\qquad
\widehat\Delta(t)=\Delta-\varepsilon\frac{D(t)}{N(t)}.
\tag{10}
$$

若最优流在区间内部，$H'(t^*)=0$，从而

$$
\widehat\Delta(t^*)=\varepsilon K(t^*)/N(t^*).
\tag{11}
$$

令 $\bar t=p_{sa}/(p_{sa}+p_{sb})$，则 $K(\bar t)=0$。严格凸性给出 $t^*<\bar t$ 当且仅当 $H'(\bar t)>0$。因此内部解反序恰好等价于

$$
\Delta<\varepsilon\frac{D(\bar t)}{N(\bar t)}.
\tag{12}
$$

端点也保持这一结论。若 $t^*=0$，则 $H'(0)\ge0$，代入式 (10) 得 $\widehat\Delta\le-\varepsilon r/(p_{sb}N(0))<0$。若 $t^*=1$，同理得 $\widehat\Delta\ge\varepsilon r/(p_{sa}N(1))>0$。这完成了对所有解的证明。均匀边频率下 $\bar t=1/2$，式 (12) 正好化为式 (6)–(7)。$\varepsilon=0$ 时则直接恢复真实排序。

注意式 (11) 中保留了首边膨胀的影响。对偶流依据膨胀后的完整路径成本分配，而实际控制使用真实首步成本加学习后缀。不能仅由“对偶流使用了昂贵路径”就断言实际贪心一定反序；以上证明正是为了解决这个差别。

**数值例和对前序反例的进一步定位。**

取 $r=0.2,A=0.6,B=0.2,\Delta=0.2$，此时

$$
\kappa=2/\sqrt3,\qquad\varepsilon_{\rm crit}=\sqrt{0.03},\qquad\eta_{\rm crit}=0.03.
$$

| 平方预算 $\varepsilon^2$ | 真实动作差 | 全局最优场的预测差 | 便宜分支流量比例 $t^*$ |
|---|---:|---:|---:|
| 0.0075 | +0.2 | +0.063069 | 1.000000 |
| 0.0300 | +0.2 | 0 | 0.500000 |
| 0.0675 | +0.2 | −0.052947 | 0.326910 |

因此平方预算 0.05 已经落在严格反序区间。所有转移都被采样、所有相关查询都有至少 0.2 的质量，函数类完整，求解是全局的。

对 2026-11-05 的六节点反例，同样令第一处分流比例为 $t$，每条路线内的其它源需求固定。将 $D(t)$ 改为两条首动作后缀的 $F_e/p_e$ 之和之差，证明完全相同。该例得到

$$
\kappa_{\rm old}=2.51320673994,\qquad
\Delta_{\rm old}=0.25,\qquad
\eta_{\rm crit,old}=0.00989517743.
$$

原来使用的 0.05 超过临界值，故不只是一次数值解出现了 −0.261996 的间隔，解析条件也保证该线性原问题的所有全局最优场在此处反序。

本次核验了 336 个需求、频率、预算和间隔组合，其中 146 个严格反序、48 个平局，解析符号全部与最优流构造出的准度量一致。另用独立的凸势函数求解核验阈值两侧与边界，并计算原始值与对偶上界差。随后补充了全部状态—目标对具有固定正质量的例子。数值核验用于发现推导或实现错误，不替代上述证明。[完整数值](/Users/z./Documents/博士/本科毕设/quasimetric-rl-mechanism-diagnostics/results/ranking_guarantees_20261111/verification_full.json) · [核验程序](/Users/z./Documents/博士/本科毕设/quasimetric-rl-mechanism-diagnostics/experiments/objective_flow_analysis/ranking_threshold.py)

**命题 3。一般有限图上的无反序充分条件。**

固定状态与目标，设 $a_*$ 是真实唯一最优动作，$P_*$ 是从其后继 $s'_{a_*}$ 到目标的一条真实最短路。记

$$
L_p(P_*)=\sqrt{\sum_{e\in P_*}1/p_e},\qquad
\Delta_b=Q^*(s,b,g)-Q^*(s,a_*,g)>0.
$$

假定完整函数类、精确全局最优，且所有相关后继到目标的查询质量为正。那么

$$
\boxed{\Delta_b>\varepsilon L_p(P_*)\ \ \text{对所有 }b\ne a_*}
\quad\Longrightarrow\quad
\widehat Q(s,b,g)>\widehat Q(s,a_*,g).
\tag{13}
$$

如果有多条真实最短路，可以取其中 $L_p$ 最小的一条。后继已经是目标时取空路径，$L_p=0$。

证明令 $\alpha_u=D_{c+x}(u,g)-D_c(u,g)\ge0$。沿 $P_*$ 求和并用加权 Cauchy–Schwarz 不等式得

$$
\alpha_{s'_{a_*}}\le\sum_{e\in P_*}x_e
\le\sqrt{\sum_ep_ex_e^2}\sqrt{\sum_{e\in P_*}1/p_e}
\le\varepsilon L_p(P_*).
$$

因此

$$
\widehat Q(b)-\widehat Q(a_*)
=\Delta_b+\alpha_{s'_b}-\alpha_{s'_{a_*}}
\ge\Delta_b-\varepsilon L_p(P_*).
\tag{14}
$$

这里不需要两个绝对误差界相加。完整类全局解在相关查询上只会相对真实距离上移，坏动作后缀的膨胀反而有利于保留正确排序，真正危险的是好动作后缀被抬高。

若要保护所有动作的完整排序，而不仅是最佳动作，对每一对真实成本严格不同的动作，把真实较好的动作作为上述 $a_*$ 使用同一论证即可。式 (14) 不要求它在全部动作中最优，只要求取其真实最短后缀。

当各相关真实最短后缀不超过 $H$ 条边，边频率至少为 $p_{\min}>0$，动作间隔至少为 $\Delta_{\min}>0$，得到统一充分条件

$$
\boxed{\varepsilon\sqrt{H/p_{\min}}<\Delta_{\min}.}
\tag{15}
$$

这也是一个完整回答，原目标并非在任何条件下都没有决策保证。它需要把松弛预算与动作间隔、路径长度和覆盖程度联系起来。这个条件是保守充分条件，不是一般图的必要条件。

**共同后缀可进一步收紧安全界，但不能把允许反序当成最优解必反序。**

在式 (5) 的强制路径结构下，令 $z=\chi_{P_a}-\chi_{P_b}$ 为后缀边计数差。则

$$
\max_{x\ge0,\ \sum_ep_ex_e^2\le\varepsilon^2} z^\top x
=\varepsilon\|z_+/\sqrt p\|_2.
\tag{16}
$$

所以 $\Delta>\varepsilon\|z_+/\sqrt p\|_2$ 足以保护排序。简单路径时，只需计入便宜后缀独有的边，共同后缀自动消去。

若严格反向，即 $\Delta<\varepsilon\|z_+/\sqrt p\|_2$，只能由式 (16) 推出存在一个预算可行的膨胀场使其反序。原推远目标是否真的选择它，仍由查询需求决定。式 (7) 才是这类特定图中有关目标最优解的结论。

**命题 4。有限求解精度下的保证。**

对于满足同一预算的任意 $\widehat d\in\mathcal Q$，设其相对于完整函数类最优值的线性目标差距为

$$
V_\varepsilon-\Phi(\widehat d)\le\eta.
$$

用 $x=[\widehat d(e)-c_e]_+$ 构造 $D_{c+x}$，并定义非负的饱和缺口 $\beta_{ug}=D_{c+x}(u,g)-\widehat d(u,g)$。因为 $D_{c+x}$ 同样可行，

$$
\sum_{u,g}\mu_{ug}\beta_{ug}\le\eta,
\qquad \beta_{ug}\le\eta/\mu_{ug}\quad(\mu_{ug}>0).
$$

于是对于好动作 $a_*$ 与坏动作 $b$，

$$
\boxed{
\widehat Q(b)-\widehat Q(a_*)
\ge\Delta_b-\varepsilon L_p(P_*)-\frac{\eta}{\mu_{s'_b g}}.
}
\tag{17}
$$

证明把误差写成 $\widehat d-D_c=\alpha-\beta$，得到预测差为 $\Delta_b+\alpha_b-\alpha_{a_*}+\beta_{a_*}-\beta_b$，只需控制两个不利项。

因此，若相关查询质量至少为 $\mu_{\min}>0$，统一充分条件变成

$$
\boxed{
\varepsilon\sqrt{H/p_{\min}}+\eta/\mu_{\min}<\Delta_{\min}.
}
\tag{18}
$$

$\eta$ 必须相对于完整函数类的最优值定义，包含有限表示误差与优化误差。神经网络训练损失收敛并不等价于该 $\eta$ 很小。若函数类无法包含饱和场，式 (3) 的单侧上移结论不再成立，故不能直接把式 (13) 用于任意原生网络。前序硬可行有限类反例正属于这个区别。

**命题 5。没有统一正间隔下界时，不存在普适的正松弛安全预算。**

严格表述为，对于任意固定 $\varepsilon>0$，都存在四节点、五条边的确定性问题，满足全部边频率为 $1/5$、全部相关查询质量至少为 0.2、边成本在 $[1,1.1]$ 内、使用完整准度量类且精确求解，但所有全局最优场都会把真实次优动作排在最佳动作之前。

证明固定命题 2 的 $r=0.2,A=0.6,B=0.2$，取

$$
\Delta_\varepsilon=\min\{0.1,\varepsilon/\sqrt3\}>0.
$$

由于 $\kappa=2/\sqrt3$，总有 $\Delta_\varepsilon<\varepsilon\kappa$，直接由式 (7) 得严格反序。

这里图规模、数据覆盖、相关查询覆盖和成本范围都没有退化，变化的只是两个动作的真实间隔。因此不能提出一个独立于动作间隔的固定正预算，保证所有任务排序正确。

这一命题还能加强到全部状态—目标对都有固定正质量。仍用同一四节点图，令状态顺序为 $s,a,b,g$，取相互独立的源与目标分布

$$
p_{\rm source}=(0.15,0.6,0.2,0.05),\qquad
p_{\rm goal}=(0.1,0.1,0.1,0.7),\qquad
\mu=p_{\rm source}\otimes p_{\rm goal}.
$$

每个状态—目标对的质量至少为 0.005，所有边仍以 $1/5$ 采样。除 $s\to g$ 查询以外，每个查询的简单路径唯一。$s\to g$ 的质量为 $r=0.105$，故聚合流为

$$
F(t)=(0.04+0.105t,\ 0.54+0.105t,\ 0.185-0.105t,\ 0.285-0.105t,\ 0.175).
$$

首边膨胀相等的位置是 $\bar t=29/42$，此时两条首动作后缀的负载差 $D(\bar t)=2$，$N(\bar t)^2=381/160$。式 (10)–(12) 的证明仍然适用，因为 $\bar t\in(0,1)$，所以

$$
\kappa_{\rm full}=\frac{2}{\sqrt{381/160}}=1.296067138\ldots,\qquad
\widehat\Delta<0\quad\Longleftrightarrow\quad\Delta<\varepsilon\kappa_{\rm full}.
$$

对任意 $\varepsilon>0$，取 $\Delta=\min\{0.1,\varepsilon\kappa_{\rm full}/2\}$ 即得全查询覆盖的反例。这里源分布、目标分布、图规模、边频率和成本范围都固定，只改变真实动作间隔。例如 $\varepsilon=0.05,\Delta=0.02$ 时，全部最优场的预测差为 −0.023719。因而该不可能性并不依赖遗漏某些状态—目标查询。

这不是说对每一个固定问题，任意小的正预算都会失败。固定该图和一个正间隔后，低于式 (8) 的预算就可以保证正确。

**命题 6。即使间隔固定，缺少边覆盖下界也没有统一正预算保证。**

在同一四节点图中固定 $r,A,B>0$ 和 $\Delta>0$，令便宜后缀边 $a\to g$ 的概率为 $q$，其它四条边均为 $(1-q)/4$。于是 $\bar t=1/2$，一般频率版本的系数满足

$$
\frac{D(\bar t)}{N(\bar t)}
=\frac{(A+r/2)/q-(B+r/2)/p_0}
{\sqrt{(A+r/2)^2/q+\{r^2/2+(B+r/2)^2\}/p_0}}
\sim\frac1{\sqrt q},
\quad p_0=(1-q)/4.
\tag{19}
$$

所以对任意固定正预算和固定真实间隔，取足够小但仍为正的 $q$，就能使式 (12) 成立，从而强制所有全局最优场反序。这与一般安全界的 $1/\sqrt{p_{\min}}$ 依赖一致，该覆盖依赖不只是证明技巧造成的松界。

例如固定 $\Delta=1,\varepsilon=0.05$、$r=0.2,A=0.6,B=0.2$，$q=0.01$ 时预测差仍为 +0.506296，$q=0.001$ 时变为 −0.575167。后者平方风险预算只有 0.0025。

**严格递增的 softplus shaping 能否排除这个问题。**

命题 2 的显式阈值针对线性推远，不能直接宣称同一数值阈值适用于 OG shaping。但是一般安全界式 (13) 对严格递增 shaping 仍成立，因为其证明只需要饱和引理和平方预算。

更强地，统一决策保证的不可能性也适用于任意可微、凹、严格递增且 $U'>0$ 的推远变换，包括

$$
U(z)=-100\log(1+\exp(5-z/100)).
$$

证明仍取均匀边频率的四节点图，固定 $r,A,B>0$ 且 $A>B$，先令 $\Delta=0$。图和预算关于 $a,b$ 完全对称。如果某个全局最优场满足 $J_a<J_b$，交换 $a,b$ 可将目标提高 $(A-B)[U(J_b)-U(J_a)]>0$，矛盾。所以 $J_a\ge J_b$。

若 $J_a=J_b$，利用凸可行域上一阶最优条件，这个解也必须最大化切线的线性目标，其有效需求为

$$
\widetilde r=rU'(J_s),\quad
\widetilde A=AU'(J_a),\quad
\widetilde B=BU'(J_b).
$$

此时 $\widetilde A>\widetilde B$。命题 2 对未归一化正需求也成立，因为 $\kappa$ 对整体需求缩放不变。将 $\Delta=0$ 代入式 (7)，线性目标的每一个全局最优解都必须满足 $J_a>J_b$，再次矛盾。因此 shaped 目标在零间隔时的所有全局最优解都严格满足 $J_a>J_b$。

有限图和有限预算给出统一有界的距离可行域。让 $\Delta\downarrow0$，任何最优解序列的极限都是零间隔问题的最优解；这是因为增加 $b\to g$ 成本只放宽约束，旧可行解仍可用，且约束和目标连续。若存在趋于零的正间隔仍能取得 $J_a\le J_b$ 的最优解，其极限会违反上一段结论。因此存在 $\delta_0>0$，使所有 $0<\Delta<\delta_0$ 的 shaped 全局最优解仍然严格反序。

这给出对 shaped 目标的存在性不可能性定理，尚未给出其同样简洁的显式 $\delta_0$。本次也独立数值核验了 OG 同形 softplus，取预算 0.05 时，$\Delta=0,0.01,0.2$ 对应预测差为 −0.133640、−0.128625、−0.030933。

对于近似 shaped 最优解，若可行距离范围内 $U'(z)\ge m>0$，目标差距为 $\eta$，则式 (17) 的最后一项改为 $\eta/(m\mu_{s'_b g})$。该导数下界可能很小，不能不加检查地当作常数 1。

**固定平方惩罚乘子的对应结论。**

如果研究的是固定 $\lambda>0$ 的目标 $\Phi(d)-\lambda\sum_ep_e[d(e)-c_e]_+^2$，其流正则为 $\sum_eF_e^2/(4\lambda p_e)$。同样推导可得一般边频率分叉图的严格反序条件

$$
\Delta<\frac{D(\bar t)}{2\lambda}.
\tag{20}
$$

均匀频率时为 $\Delta<5(A-B)/(2\lambda)$。有限乘子也不能在没有间隔下界时提供统一排序保证。这个固定乘子问题与给定正预算的约束问题不同，不能混用二者的阈值。

**可以写入研究论证的结论。**

平均平方违反预算对完整准度量目标引入依赖查询需求和转移频率的路径膨胀。在明确的分叉结构中，该膨胀具有可计算的临界值，超过后全部全局最优距离场严格反序。因而即使排除数据缺失、函数逼近和求解误差，固定正预算仍不具有与动作间隔、转移覆盖无关的统一决策正确性保证。与此同时，加入正间隔、覆盖、路径长度和全局目标差距条件后，可以得到式 (18) 的充分保证。

这些是对优化问题的数学结论，不能据此直接认定现有 U 或 OG 保存模型已经位于相应全局反序解。实际 IQE、随机目标、潜在动力学耦合、有限优化和连续状态支持都需要另外检查。本文没有把可求解图上的结论替代为真实神经训练的全局认证。

复现新增数值核验，输出文件需尚不存在。

```bash
env PYTHONDONTWRITEBYTECODE=1 ../quasimetric-rl/.venv/bin/python experiments/objective_flow_analysis/ranking_threshold.py /private/tmp/qrl-ranking-threshold-verification.json
```
