# LPCA Stepwise Execution Plan

This is the execution track for the ambitious program, without date gating.

## Stepwise Checklist

- [x] Add a frozen full-suite config and seed registry.
- [x] Add a single-command orchestrator to run E1/E2/E3/E4/M2 and write manifests.
- [x] Add fail-fast behavior to analysis scripts (no demo fallback).
- [x] Add artifact index generation mapped to concrete files.
- [x] Add model-matrix execution mode for 3 backbones with paired seeds.
- [x] Add preregistered-stats-only aggregate report generation.
- [ ] Add M2v2 architecture path (VQ bottleneck + anti-shuffle + MI + curriculum).
- [x] Add gate-report generator with CI-backed pass/fail policy.
- [x] Add reproducibility package builder (lockfile, runbook, main-table reproducer).
- [x] Add paper-pack generator that emits tables/figures directly from manifests.

## Run Order

1. `prepare`: validate config + seed registry, write run metadata.
2. `run`: execute experiment commands for selected models and experiments.
3. `aggregate`: update artifact index and build preregistered-stats-only suite report.
4. `gate`: generate CI-backed M2 gate report (`Normal > P0`, `Shuffle < P0`).
5. `paper`: emit manuscript-facing tables and figure-ready JSON from run manifests.
6. `package`: assemble reproducibility bundle (runbook, lockfile, reproducer script).
7. `render`: build run summary markdown.
8. `publish`: write manifest JSON/markdown and final pointers.

## Command

```bash
python scripts/run_full_suite.py \
  --config configs/full_suite_frozen.yaml \
  --stages prepare,run,aggregate,gate,paper,package,render,publish
```

Use `--execute` to run real commands. Without it, the runner is dry-run by default.
