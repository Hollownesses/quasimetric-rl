# KKT-aligned functional-dual QRL v3

This branch exposes the first neural implementation of the KKT-aligned QRL
ablation.  The legacy squared-hinge constraint remains the default so existing
experiments and checkpoints keep their original behavior.

## Objective

For transition edge `e=(s,a,s')`, define

```text
h_e(theta) = d_theta(s, s') - c_e.
```

The critic minimizes

```text
-E[d] + E[stopgrad(lambda_psi(e)) h_e]
      + 0.5 rho E[relu(h_e)^2].
```

Once the observed violation fraction reaches a configured wake-up threshold,
each sampled edge receives a projected dual target

```text
lambda_target(e) = clip(stopgrad(lambda_psi(e)) + eta h_e, 0, lambda_max).
```

The functional dual minimizes a Huber regression loss from its effective raw
output to the inverse-softplus form of this target.  This is a fitted version
of tabular projected dual ascent: violated edges request a larger multiplier,
while every slack edge, including deeply slack edges, requests a smaller one.
It therefore preserves per-edge complementarity targets without separately
normalizing positive and slack populations.  `lambda_psi(e)` is an
edge-conditioned positive field with numerical safety bounds.  Its inputs are
detached source observation, detached destination observation, and `c_e`,
passed through a fixed signed-log transform.  It does not share a gradient
path with the critic.

The softplus raw output has a straight-through lower trust bound so a transient
negative excursion cannot remove the recovery gradient.  Every critic update
is preceded by `dual_steps` dual updates.  The default v3 ablation uses one
dual step, dual learning rate `1e-4`, and `rho=1`.

## Run the isolated diagnostic arm

```bash
PHASE=train_qrl_kkt_functional \
DEVICE=mps \
OUTPUT_ROOT=./results/diagnostic_u_shadow_corridors_topology_v2 \
bash minimal_qrl/run_comm_inspection_diagnostic.sh
```

The phase also selects the theoretical linear Global Push objective and turns
off the optional temporal, n-step, and MQE losses, isolating the optimizer
change.

Useful overrides include:

```bash
QRL_KKT_AUGMENTED_LAGRANGIAN_RHO=0.3
QRL_KKT_DUAL_STEPS=1
QRL_KKT_DUAL_LR=0.0001
QRL_KKT_DUAL_START_VIOLATION_FRACTION=0.05
QRL_KKT_DUAL_PROJECTED_STEP_SIZE=0.1
QRL_KKT_DUAL_HUBER_DELTA=1.0
QRL_KKT_DUAL_FEATURE_SCALE=5.0
QRL_KKT_DUAL_RAW_MIN=-10.0
QRL_KKT_DUAL_MAX=100000
QRL_KKT_DUAL_HIDDEN_SIZES="128 128"
```

## 2k-step smoke test

```bash
PHASE=train_qrl_kkt_functional \
DEVICE=cpu \
OUTPUT_ROOT=./results/diagnostic_u_shadow_corridors_topology_v2 \
TOTAL_STEPS=2000 \
LOG_INTERVAL=10 \
SAVE_INTERVAL=500 \
EVAL_INTERVAL=500 \
VIS_INTERVAL=500 \
KKT_TRAIN_DIR=./results/diagnostic_u_shadow_corridors_topology_v2/qrl_training_kkt_functional_smoke_v3 \
bash minimal_qrl/run_comm_inspection_diagnostic.sh
```

Use a fresh `KKT_TRAIN_DIR`.  Before a full run, verify that the logged dual is
awakened, `lagrange_mult_max` remains finite and nonzero, and increasing
violation is followed by a dual response rather than lower-tail saturation.

## Logged diagnostics

The local-constraint log contains the linear Lagrangian term, augmented
penalty, residual min/mean/max, violation and near-active fractions, dual
min/mean/max, raw dual min/mean/max, the lower-saturation fraction, the dual
wake-up state, projected-target min/mean/max, and projected increase/decrease
fractions.  It also records the last inner dual update and an absolute
complementarity diagnostic.  Training fails fast if violations are present
while the maximum multiplier is numerically dead.

This implementation establishes the neural functional-dual training path.  It
does not by itself establish KKT convergence.  The next validation stage is to
run fixed exact per-edge dual, learned tabular per-edge dual, and functional
dual comparisons on the same finite U-trap graph used by the static KKT
autopsy, measuring feasibility, complementarity, stationarity residual, and
successor ranking.

The first fixed-dual check is available without porting the full diagnostic
graph builder into this branch:

```bash
python -m minimal_qrl.industry_exp.kkt_exact_dual_validation \
  --arrays /path/to/static_vector_field_arrays.npz
```

It checks the exact edge reaction and reports the LP-sum multiplier scale as
well as the equivalent multiplier-density scale used by an expectation loss.
