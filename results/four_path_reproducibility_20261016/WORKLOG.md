# Five-seed fixed-pipeline replication — 2026-10-16

- First implemented and committed model-only QP stopping as 7bdabd0.
- 18 initial tests passed. Re-ran the user delta=.15 checkpoint with the new
  stop logic: same final values, stopped at iteration10 based on feasibility
  and model-derived value guard. Graph reference evaluated after model save.
- Then implemented fresh production network initialization and three-stage
  pipeline; 21 related tests passed, including forbidden checkpoint loads,
  seed uniqueness, Adam continuity and reference evaluation timing.
- Predeclared seeds101,202,303,404,505 before formal training.
- Completed separate seed9001 smoke:20 PHR+20 decay+2 QP attempts; retained
  failed joint evaluation, no tuning performed.
- Started all five formal runs:20000 PHR+5000 decay, rho100, delta=.15,
  R0 QP at most40 attempts and three-stall stop. All initializations distinct;
  all data/pair/cost fingerprints identical. Main source snapshot frozen.
- All five seeds finished20k PHR+5k decay. Original final joint pass1/5
  (404); 101/505 failed push gap, 202/303 stopped at QP step0 with Gram PSD
  assertions. Preserve the original exits/logs/checkpoints.
- 2026-10-17: independently diagnosed cancellation in the expanded Gram;
  relative min eigenvalues approximately-1.5e-10. Added explicit Jacobian-block
  PSD Gram fallback without changing tolerance or mathematical QP. Independent
  parameter-space quadratic-form error below2.2e-13;23 tests pass.
- Reran QP for ALL five unchanged decay checkpoints under common corrected
  numerics in qp_numerical_fix; no retraining, tuning or seed replacement.
  All5 tolerance-feasible,3 strictly nonpositive,1/5 jointly passes (404).
  All4 final failures are push gap>1%; all task NRMSE values pass.
- 404/505 reruns have identical model hashes. 101 did not use fallback but
  repeated QP stops after2 instead of3 rounds; push differs by-3.05e-5 and
  parameter L2 difference1.83e-4, same failure classification. Disclosed CPU
  numerical sensitivity rather than claiming bitwise reproducibility.
- Audited all initialization/stage checkpoints,25k-step sequences, frozen
  data, learning rates, Adam counts, push guards and all-edge final arrays.
  The first report audit stopped on an overly strong bitwise-repeat assertion;
  partial generated tables are retained as main/comparison_audit_incomplete,
  then the completed report records actual repeat differences.
- Final report: main/comparison/report.md. Original success1/5 and corrected
  success1/5 both reported. Stage success counts0/5,0/5,1/5. No reference-driven
  stopping, best-checkpoint selection or per-seed retuning.
- Light results/source snapshots are versioned with the second commit; large
  checkpoints and125000 per-update records remain local with reproduction CLI.
