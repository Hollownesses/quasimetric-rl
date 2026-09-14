# MQE-inspired two-sided waypoint consistency

This experiment keeps the repository's state-only IQE critic, softplus
GlobalPush, local constraint, temporal path constraint, goal-return constraint,
and evaluation pipeline unchanged. It adapts only MQE's multistep waypoint
sampling idea.

For a physical future goal `g` and a same-episode waypoint `s_{t+k}`, the loss
uses a frozen EMA target critic and regresses both sides with Huber loss:

```text
d_online(s_t, g) = C[t:t+k] + stop_grad(d_target(s_{t+k}, g))
```

`k=1` is retained with an explicit Bernoulli component; otherwise `k` is drawn
from a truncated geometric distribution. `C[t:t+k]` is the accumulated
nonnegative engineering cost, not a fixed step count or an oracle label.

## Abstract-goal anchors

The previous implementation sampled an abstract-goal anchor only when a replay
sample already came from a successful episode and then passed another Bernoulli
test. With a 3.58% successful-transition rate and probability 0.1, only about
0.358% of MQE samples reached the abstract goal.

The v2 implementation reserves `ceil(batch_size * terminal_anchor_fraction)`
MQE-only slots per batch and independently samples their sources from complete,
naturally successful episodes. The main replay batch and every other loss keep
their original sampling distribution. With the default fraction 0.1, a batch of
256 therefore contains 26 terminal anchors rather than roughly one.

For these samples, the waypoint is the physical terminal state, the goal is the
matching abstract task goal, and `d(s_terminal, G)=0` is applied exactly by the
task definition. The target is the observed remaining trajectory cost. No
Oracle, Hybrid A*, or Dijkstra value is used.

Training logs expose `terminal_anchor_count`, `terminal_anchor_fraction`,
`physical_huber`, and `terminal_anchor_huber` separately.
Every checkpoint and `timing.json` also record the experiment variant, the
softplus GlobalPush objective and its parameters, and all MQE hyperparameters.

## Commands

```bash
PHASE=train_qrl_nstep_upper_bound DEVICE=mps \
  OUTPUT_ROOT=./results/diagnostic_u_shadow_corridors_topology_v2 \
  bash minimal_qrl/run_comm_inspection_diagnostic.sh

PHASE=train_qrl_mqe_waypoint_consistency DEVICE=mps \
  OUTPUT_ROOT=./results/diagnostic_u_shadow_corridors_topology_v2 \
  bash minimal_qrl/run_comm_inspection_diagnostic.sh
```

The MQE phase defaults to a terminal-anchor fraction of 0.1. Override it with
`QRL_MQE_TERMINAL_ANCHOR_FRACTION` when running an explicit ablation.
