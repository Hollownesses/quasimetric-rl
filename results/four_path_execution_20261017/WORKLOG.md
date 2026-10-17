# QP before/after × MPPI execution — 2026-10-17

- Learner, costs, data and all ten checkpoints unchanged. Ten adapter exports
  validated on1009 recorded edge/task/B1 pairs each: exact output agreement,
  original/canonical observation equivalence and distance_scale=1.0.
- Frozen MPPI defaults: H20,N256,sigma.8,T1,terminal weight1;300 real steps.
  Five seeds101/202/303/404/505 at decay/QP plus none/geometry baselines.
- Frozen32 poses:12 recorded path start/west-exit/east-wall,12 position+heading
  perturbations,8 independent U-interior held-out poses. Two MPPI seeds per pose;
  baseline runs are shared paired controls, not duplicated independent models.
-17 initial tests and a3-step development smoke completed before formal launch.
  Rectangle collision vectorization tested against the scalar implementation;
  sampled controls match pre-change MPPI over multiple nominal-sequence shifts.
- All768 formal episodes completed without process errors,4 workers with one
  torch thread each. Wall clock435.98s. No tuning or performance-based reruns.
- Reused16 fixed development candidate banks (4096 sequence rollouts), scored by
  all12 arms. Four east-facing banks contain zero safe/useful sampled sequences.
  Where useful candidates exist they can receive near-zero QRL weight; QP barely
  changes ranking or first action (maximum absolute control change.002169 rad/s).
- Execution: none/geometry each28/64; decay211/320 and QP211/320. One paired
  improvement and one regression, both303;210 jointly successful pairs have
  mean cost change+0.01778. Collision80→74, timeout14→20, out-of-bounds15→15.
  Holdout QRL success21/80 at both stages; baselines0/16.
- Audit detected exit-region visits missed when a movement segment crosses the
  west corner between sampled endpoints. Fixed only the measurement, preserved
  raw episodes and original frozen source, recorded222 changed stage fields in
  74 episodes. No success label, cost, action or trajectory changed. Added a
  corner regression:18 final tests pass. Analysis source separately snapshotted.
- Independently replayed all768 real action sequences (56524 real steps) in the
  unchanged environment: every state/reward and terminal outcome matches exactly.
- Report, all per-episode trajectories/diagnostics, candidate scores, source and
  checkpoint hashes retained. Draft report trees kept locally but not committed.
  The report generator's initial duplicate-key error was fixed before final
  audit; it did not affect any experiment output.
