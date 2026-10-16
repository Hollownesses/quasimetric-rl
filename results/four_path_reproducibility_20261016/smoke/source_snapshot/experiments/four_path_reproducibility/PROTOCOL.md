# Five fresh initializations, frozen before formal training

Seeds: **101, 202, 303, 404, 505**, all reported, no seed selection/replacement.
Create a fresh production QuasimetricCritic with torch.manual_seed(seed), enable
the existing terminal quotient, retain its encoder/projector/IQE. Do not load a
historical critic or reset only a subset of network parameters. Initial Adam
state is empty; all339 multipliers start at zero. Record distinct parameter
hashes for the complete model and separately encoder/projector.

Frozen data: the same four-path bank, exact42D canonical node identity (336),
339 replay edges (335 physical+4 anchors), costs, task goals and fixed B1 pairs
constructed with data seed42. Model seed NEVER changes pair sampling. All five
data fingerprints must match the established dataset. No new transitions,
auxiliary losses, graph-distance labels or reference multipliers.

Frozen three-stage flow, regardless of evaluation quality:

1. PHR: 20000 full-support primal updates, AdamW lr2e-5, wd0, rho100,
   independent339-entry lambda; each step lambda <- relu(lambda+pre-primal h).
2. Lower step: 5000 more updates, same Adam moments and lambda; all primal
   learning rates (including IQE reduction) exponentially2e-6 ->2e-7; otherwise
   identical PHR. No optimizer/multiplier reset at the stage boundary.
3. R0 QP: full final projector layer, same exact Gram and existing projection
   solver, desired direction zero, delta=.15 ABSOLUTE relative to stage2's MODEL
   push. Floor fixed throughout repair. No tuning per seed or use of references.
   At most40 attempts, 2000 QP sweeps, trust radius.1, margin5e-5, 13 backtracks,
   elastic fallback100; stop after3 stalled attempts or model-only feasibility
   plus value guard. Every actual nonlinear candidate checks all339 edges.

Repair selection uses physical h<=1e-5 and anchor abs(h)<=1e-6. Report strict
sign feasibility (max h<=0) separately; never relabel tolerance-feasible as strict.
Joint end evaluation is unchanged: full-edge feasibility, terminal/anchor zero,
same-support graph push gap<=1%, task NRMSE<=2%. Reference distances and metrics
are computed only AFTER the complete per-seed flow has selected its final model;
then also evaluate the fixed stage1/stage2 snapshots. No early stopping by loss,
reference scores, seed rankings, or manual best-checkpoint selection.

CPU torch threads2 for each run. Every step records actual parameter-update L2,
max pre-primal violation and push; full model-only traces every250 steps.
Save initialization and exactly named phr/decay checkpoints with full Adam,
lambda, RNG, stage and seed; repair checkpoint flags stale Adam after QP.
Wall-clock, primal/dual/QP counts and failures are reported per seed.

Before formal runs: meaningful initialization, stage-continuity, frozen-support
and reference-isolation tests plus a SHORT pipeline smoke on separate seed9001
(20+20 updates, 2 QP attempts). Smoke never chooses hyperparameters or seeds;
its intermediate learning-rate schedule uses the SAME 5000-step denominator.
All five formal runs restart fresh; the smoke is excluded from success rate.
Numerical implementation errors may be fixed and transparently rerun; optimizer
failure, poor values or stalled repair remain failures, not grounds to retune.
