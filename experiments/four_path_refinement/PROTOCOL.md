# Frozen four-path checkpoint feasibility refinement

All arms restore `main/encoder_phr_c0/latest.pt` at step 20000, including its
single-group AdamW moments, trainable IQE head, exact canonical nodes and learned
339-entry multiplier. The original artifacts remain unchanged. Same exact push,
one-step costs, task/B1 support and rho=100; no graph labels enter training.

Predeclared diagnostic budget: 5000 primal updates per arm, also report 2000.
Full evaluations every 25 updates; actual parameter deltas recorded every step.
100-step pilots use the SAME 5000-step learning-rate schedule, then restart.

* A: original constant primal lr=2e-5, eta=1, pre-primal residual dual update.
* B: only primal lr changes: 2e-6 exponentially to 2e-7 at update 5000;
  Adam moments retained; dual update identical to A.
* C: same lr schedule as B, 20 primal updates with fixed lambda, then eta=20
  projection using fresh full residuals. Nominal dual step per primal stays 1,
  actual dynamics differ. This is a combined solver strategy, not a single-factor
  contrast. Rho stays 100; no penalty escalation or reset of moments.

Joint pass unchanged: all 335 physical residuals <=1e-5, anchors/terminal <=1e-6,
same-support graph push gap <=1%, task NRMSE <=2%. No guarantee a 5k budget passes.
Report all terminal-entry edges, path cumulative positive residuals, triangle
slack, complementarity and actual parameter-update magnitudes, not just loss.
Graph distances are independent evaluation references, not continuous-environment
optimal distances. These are deterministic single-start solver diagnostics.

If B and C fail, attempt local nonlinear-verified feasibility restoration with
fixed global push floor `J_start - 0.1` (not a repeatedly decreasing floor).
The floor comes solely from the chosen current model. Use all constraints and
accept restorative progress from infeasibility; failed local optimization does
not prove infeasibility or a necessary value/feasibility tradeoff.

The conditional repair starts at B's final 5k checkpoint (chosen before results),
with 40 local iterations, all 339 edge rows and the push guard in every QP.
Reuse `feasible_direction.solve_direction`; a candidate must pass full nonlinear
forward checks. Optimize the entire final projector layer (1050624 weight/bias
parameters). Assemble its exact Jacobian Gram from endpoint derivatives and
hidden-feature dot products; factor it to reuse the existing small projection
QP, then pull its transient dual coefficients back to actual network directions.
This avoids materializing a 339 x 1050624 Jacobian, while preserving the original
shared network and IQE. Earlier network layers and the IQE reduction stay fixed
during this local diagnostic, so it is not full-network SQP. Failure is also
restricted by this subspace and local linearization. A component-shared subspace
was considered and mapping-tested during preparation, then replaced before any
repair run with the larger full-final-layer subspace and its audited Gram.
The nominal interior margin is 5e-5 on physical edges (anchors remain zero),
parameter trust radius .1, 2000 QP sweeps, 13 backtracking candidates per attempt.
Hard projection may fall back to elastic projection (penalty100), while the
actual nonlinear push floor is never relaxed. Stop after three stalled iterations.
The saved Adam state is marked stale after QP repair and must not be treated as
an uninterrupted Adam continuation.

Exploratory follow-up, specified after the minimum-norm repair stalled: repeat
from the SAME B-5k checkpoint with the SAME fixed push floor, but project a
normalized original push-ascent direction of norm .1 instead of zero. The
Gram-factor direction is exactly mapped to that real-parameter ascent vector.
This changes the local QP objective, not its constraints or the joint gate.
Keep the first unsuccessful repair as `repair`, the follow-up as `repair_push`.

## Reference-free repair stopping (subsequent correction)

Repair now selects and saves its final model using only full physical feasibility
(h<=1e-5), zero anchors (abs(h)<=1e-6), its fixed model-derived push floor, the
iteration budget and the three-stall limit. Graph push gap / task NRMSE / joint
pass are evaluated only after that selection. `evaluation.json` records the
post-stop evaluation; intermediate repair curve rows contain no graph metrics.
A low-value feasible model may therefore stop and subsequently FAIL the joint
evaluation. That failure must be retained, not used to restart or extend training.
Existing archived runs retain their original stopping semantics and are not
silently relabeled as reference-free experiments.
