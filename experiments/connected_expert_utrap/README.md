# Connected expert U-trap graph

The old `experiments/expert_utrap_network/` builder and its frozen dataset remain
unchanged. This separate builder expands that backbone with executed local control
branches, exact inner/outer-route connectors, and new U starts in eight headings.
It never reads graph distances or Oracle values to construct/select trajectories.

```sh
export MPLCONFIGDIR=/tmp/qrl-connected-mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=../quasimetric-rl/.venv/bin/python
$PY -m experiments.connected_expert_utrap.build --output results/NEW_CONNECTED/main
$PY -m experiments.connected_expert_utrap.audit --root results/NEW_CONNECTED/main
```

Output directories must not exist. Defaults: all 64 original trajectories,
source stride 16, target stride 4, a 3×3 grid of new U positions with eight
heading bins, and six geometric target proposals per connector search. The
builder records its configuration, all failed attempts, exact trajectories,
one-step replay, source snapshots and hashes. Failed construction proposals are
not passed off as successful replay. Search exhaustion is not unreachability.

## Train QRL on the new graph

```sh
$PY -m experiments.full_replay_four_path.train \
  --connected-expert-data results/connected_expert_utrap_20261021/main \
  --output results/connected_expert_qrl_20261021/main/train \
  --seed 606 --phr-steps 20000 --decay-steps 5000 \
  --batch 256 --threads 4 --audit-interval 1000 --task-mode all
```

The existing options remain available:

| Option | Dataset |
|---|---|
| No data option | Original 200k replay + four paths |
| `--expert-data PATH` | Original designed expert graph |
| `--connected-expert-data PATH` | New connected expert graph |
| `--curriculum-data PATH` | Existing curriculum replay |

Options are mutually exclusive. Resume by repeating the full command and adding
`--resume PATH/latest.pt`. Checkpoints preserve dataset identity, fingerprints,
task masks, model, Adam, per-record multipliers, sampler, visit counters and RNG;
cross-dataset/configuration restoration is rejected. Four-thread floating-point
execution retains the existing numerical nondeterminism; single-thread restart
tests compare exact state equality.

The learner remains the existing production encoder/projector/IQE, .6 task + .4
B1, per-record PHR (rho=100, eta=1 on visits), and original learning-rate schedule.
This is one source with weight 1, not a .5 mixture with old four-path data. All
accepted trajectories have legal success suffixes, so all/witnessed masks agree.
Constraints are not silently deduplicated: uniform physical-record sampling is
retained; the audit separately reports duplicate records and unique graph edges.
No MPC changes, new optimizer, graph regression targets or formal training are
part of construction. Graph distances live only in `audit/` after construction.

The audit replays every complete trajectory, checks exact new joins and original
trajectory preservation, validates costs/anchors/B1 boundaries, compares old-node
distances and old branch jumps, and writes `audit/report.md` and `network.png`.
This is expert-assisted research data, not an independently held-out evaluation
or a complete reachable-state representation of the continuous environment.

```sh
$PY -m pytest -q experiments/connected_expert_utrap/test_connected.py \
  experiments/expert_utrap_network/test_training_adapter.py \
  experiments/full_replay_four_path/test_full_replay.py \
  experiments/witnessed_task_push/test_witnessed.py
```
