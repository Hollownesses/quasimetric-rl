# Oracle / QP-QRL / No Terminal paired MPPI (2026-10-17)

Freeze before execution; no learner updates or outcome-dependent tuning.

- Copy previous `results/four_path_execution_20261017/main/states.json` byte for byte: 12 support, 12 perturbed, 8 previously frozen holdout poses (including heading); 314159 / 271828 planner RNGs. The holdout cohort has already been evaluated in the previous experiment; it is not a new unseen test set.
- Seven arms: none, oracle, qp_101/202/303/404/505. QP weights from `qp_numerical_fix/seed_*/latest.pt`, strict module transfer and record-pair equality validation, distance_scale=1.0. Every model is retained.
- Only MPPI H and N change relative to the previous experiment: H=10, N=128. Noise .8, temperature 1, terminal weight 1, invalid penalty 1e6, terminal_samples 128, max real steps 300; unchanged environment, terminal legality, bank, costs and pairings. RNG seed offset remains +7001, Gaussian candidates around shifted nominal, candidate zero nominal, weighted-mean control. Equal initial RNG does not imply identical closed-loop candidate trajectories after states/nominals diverge.
- Oracle ports `hybrid_astar_value.py` verbatim from 34d1c011f637b3352ac1dc1134650f6f28c20b17; MPPI provider hook follows that commit while retaining current tested rollout/collision kernels. Position resolution .25, heading bins 24, constant-control primitives 5 steps with scales [-1,-.5,0,.5,1]. Reverse Dijkstra using real environment costs and legal terminal states; nearest-bin value lookup, unreachable value 1e6, exact legal-terminal queries zero. No dt rescaling, no forward Hybrid A* replanning. Max-expansions/timeout/heuristic fields are inherited configuration metadata and do not bound reverse Dijkstra.
- This is a privileged environment-model baseline on a snapped state lattice, NOT a continuous-environment optimum or a four-path recorded-graph oracle. Snapping can underestimate achievable continuous costs; no optimality claim for its MPPI controller. No oracle values are used for QRL training or selection.
- Fresh local cache: first formal Oracle episode builds table; preserve preprocessing time, cache diagnostics and table hash. Report online decision latency separately from cold table construction/initialization. Oracle table primitive-build workload is additional to MPPI rollout counts.
- All 448 episodes retained: success, collision, out-of-bounds, timeout; success-only costs/steps; paired jointly successful cost differences; eligible exit/wrap rates and terminal success; planning time and actual model transitions. No comparison using mean truncated failure costs. Replay every saved executed action through the unchanged environment.
- Development scoring: same 16 probe/RNG banks as previous experiment but H10/N128, zero nominal. Every arm scores the identical candidate batch. Safe westward progress is only a local heuristic, not proof of an escape sequence. Preserve weights, ranks and weighted first actions.
- Smoke/tests use temporary cache, never formal outcome-driven configuration changes. Prior experiments and the historical external result directory are read-only.

Reproduce from repository root (`PY=../quasimetric-rl/.venv/bin/python`, use a new output directory):

```
$PY -m pytest -q experiments/four_path_oracle_execution/test_oracle.py experiments/four_path_execution/test_execution.py minimal_qrl/test_comm_inspection_baselines.py
$PY -m experiments.four_path_oracle_execution.run prepare --root RESULTS
$PY -m experiments.four_path_oracle_execution.run suite --root RESULTS --workers 4
$PY -m experiments.four_path_oracle_execution.run candidates --root RESULTS
$PY -m experiments.four_path_execution.audit_replay --root RESULTS
$PY -m experiments.four_path_oracle_execution.report --root RESULTS
```
