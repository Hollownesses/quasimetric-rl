# Frozen paired MPPI execution evaluation — 2026-10-17

No learner updates. Use all initialization seeds101,202,303,404,505: original
25k decay.pt and numerical-correction QP latest.pt from the completed replication.
All checkpoint SHA256 recorded before evaluation. Do not select models by gap.

Reuse MPPIConfig defaults: horizon20, candidates256, independent per-time Gaussian
noise sigma0.8, temperature1, terminal weight1, invalid penalty1e6, geometric
terminal samples128. No tuning after outcomes. Existing environment/costs,
300-step limit, single u_trap_target task, distance_scale=1.0 (not dt).
Controls compared: no terminal, existing geometry terminal, each seed decay,
each seed QP (12 arms). Two planner seeds314159/271828 per state. Baselines are
run once per state/MPPI seed and reused for matched comparisons, not counted as
five independent baseline models. Same initial pose/task and random stream;
closed-loop states and shifted nominal controls diverge, so subsequent candidate
trajectories are NOT identical. Do not train or score against graph references.

State bank (immutable before formal execution): each recorded path's start,
first west exit, first east-of-wall state (12 support poses); one reproducible
uniform perturbation per pose (dx,dy in[-.15,.15], dtheta in[-.35,.35], reject
invalid/terminal) (12 poses); eight new U-interior poses in[3.1,2.2,4.9,5.1],
headings uniform[-pi,pi], generator2026101702, excluding positions within.25 of
ANY training trajectory state, old bank excluded diagnostic starts, and old
candidate starts (8 holdouts). No performance-based filtering. Holdouts are
finite geometric pose holdouts, not unseen maps/tasks. Perturbation seed2026101701.
Held-out outcomes are inspected only after code, tests, smoke and settings freeze.

Stages: first west-exit visit; ordered traversal outside upper/lower wall envelope
then east of closed wall; legal task termination. For support middles, historical
training-prefix stage completion is documented separately, never counted as a
new achievement. Report stages conditional on starting stage eligibility; also
report cumulative completion including known prefix. Perturbations inherit no
prefix evidence: stages already geometrically passed are marked not applicable,
not executed success. New U-interior holdouts start eligible for all stages.

Report all arms/seeds and all3 strata: task success, collision, out-of-bounds,
timeout, successful-only total cost/steps, stage rates/denominators, plan times,
sequence rollout counts and actual simulated transitions. Paired post-pre success
changes on identical pose/RNG keys; cost differences only on jointly successful
pairs. Never rank all-episode truncated costs. Do not pool baseline duplicates.
Save trajectories and per-decision diagnostics. No chosen-best rollouts.

Separate candidate diagnostic uses existing DEVELOP probes at x4.8/4.0,y3.65,
headings west/north/east/south, with both fixed planner seeds and zero nominal.
Reuse exactly one generated candidate bank+model rollout for every scoring arm.
Progress means full-horizon safe, >=.25 decrease in x toward west exit AND final
heading has cos(theta)<0; separately count west-exit visits and success. This is
only local progress coverage, not proof of a full escape sequence. For each arm
report useful count/weight mass, safe candidate rank correlation, top10 overlap,
weighted first action and actual one-step safety; preserve per-candidate arrays.
No injecting oracle or hand-designed candidates after detecting poor coverage.

Validation: IndexedIQE→production critic→existing value adapter all339 recorded
edge outputs +335 task/B1 outputs; raw observations/terminal quotient consistency;
scale1.0; checkpoint parameters unchanged; rollout vs real environment including
invalid transitions; stage/success-only accounting; common candidate scoring vs
controller action. Smoke uses separate development pose and planner seed9001,
fixed settings, at most3 real steps. Smoke does not select hyperparameters.

Any execution-only acceleration must preserve existing MPPI mathematics and
be verified against scalar environment/previous implementation. CPU workers use
one torch thread each. Concurrent wall times include scheduling; plan latency is
reported as measured under that load, not a dedicated hardware benchmark.

## Post-execution measurement correction

All768 episodes completed under the original frozen source with no controller
change or rerun. Independent audit found that exit-region membership at sampled
endpoints misses safe segments that cross the west corner between samples.
Count an exit when any recorded movement segment intersects the same west-exit
rectangle, consistent with the environment's swept collision semantics. Preserve
the original episode JSONL, states/actions and source_snapshot. The post-run
comparison records old/new stage fields and analysis source hashes separately;
success/failure and costs are unaffected. Add a regression for the corner case.
