# KKT-aligned functional-dual QRL v1

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

The dual network independently minimizes

```text
-E[lambda_psi(e) stopgrad(h_e)],
```

which is gradient ascent on the Lagrangian dual objective.  `lambda_psi(e)` is
an edge-conditioned positive field with a high numerical safety cap.  Its inputs are detached raw
source observation, detached raw destination observation, and `log1p(c_e)`.
It does not share a gradient path with the critic.

Every critic update is preceded by `dual_steps` dual updates.  The default v1
ablation uses three dual steps and `rho=1`.

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
QRL_KKT_DUAL_STEPS=5
QRL_KKT_DUAL_LR=0.001
QRL_KKT_DUAL_MAX=100000
QRL_KKT_DUAL_HIDDEN_SIZES="128 128"
```

## Logged diagnostics

The local-constraint log contains the linear Lagrangian term, augmented
penalty, residual min/mean/max, violation and near-active fractions, and dual
min/mean/max.  It also records the last inner dual update.

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
