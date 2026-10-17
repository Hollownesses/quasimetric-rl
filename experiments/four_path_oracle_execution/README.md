# Frozen Oracle / QP-QRL / No Terminal execution comparison

See [the frozen protocol](PROTOCOL.md) for commands and definitions, and [the completed report](../../results/four_path_oracle_execution_20261017/main/comparison/report.md) for all 448 episodes.

H10/N128 results: Oracle 62/64; QP seeds 101/202/303/404/505: 50/48/42/42/43 of 64; no terminal 27/64. No QRL retraining, model selection or new controller tuning.

The source snapshot in the result manifest freezes execution code before the run. `report.py` and `audit_oracle.py` are post-run analysis only. The latter optionally compares the newly built lattice values to the user's historical cache and reconstructs first-decision candidates for Oracle failures:

```sh
../quasimetric-rl/.venv/bin/python -m experiments.four_path_oracle_execution.audit_oracle \
  --root RESULTS --historical-cache HISTORICAL_ORACLE_CACHE.npz
```

Run this optional audit before report generation to include its findings. It does not change episodes or controller configuration. Every output-producing command requires a fresh output path/directory; preserve old runs. The historical cache is not needed for new execution or ordinary reports. Frozen QRL checkpoint paths and hashes are recorded in the manifest; checkpoint binaries follow the repository's existing local-artifact convention and are not copied into this experiment.
