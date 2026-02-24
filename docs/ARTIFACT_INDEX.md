# LPCA Artifact Index

Generated (UTC): 2026-02-24 06:51:30Z

This file maps headline numbers to concrete artifact files for traceability.

## E1 Tightened Baselines

| Experiment | Metric | Value | 95% CI | N | Artifact |
| --- | --- | --- | --- | --- | --- |
| E1 Tightened | P0 success_rate | 0.200 | [0.112, 0.330] | 50 | results/e1_tightened/llm_constraint_satisfaction_20260114_175702/summary.json |
| E1 Tightened | P1 success_rate | 0.680 | [0.542, 0.792] | 50 | results/e1_tightened/llm_constraint_satisfaction_20260114_175702/summary.json |

## E2-min Budgeted Text Sweep

| Experiment | Config | Success | 95% CI | N | Artifact |
| --- | --- | --- | --- | --- | --- |
| E2-min | P2_16B | 0.000 | [0.000, 0.114] | 30 | results/e2_sweep_20260114_191902/results.json |
| E2-min | P2_64B | 0.333 | [0.192, 0.512] | 30 | results/e2_sweep_20260114_191902/results.json |
| E2-min | P2_256B | 0.667 | [0.488, 0.808] | 30 | results/e2_sweep_20260114_191902/results.json |
| E2-min | P5_16B | 0.567 | [0.392, 0.726] | 30 | results/e2_sweep_20260114_191902/results.json |
| E2-min | P5_64B | 0.600 | [0.423, 0.754] | 30 | results/e2_sweep_20260114_191902/results.json |
| E2-min | P5_256B | 0.667 | [0.488, 0.808] | 30 | results/e2_sweep_20260114_191902/results.json |

## E3/E4 Latent Baselines

| Experiment | Metric | Value | 95% CI | N | Artifact |
| --- | --- | --- | --- | --- | --- |
| E3 CIPHER | E0 success_rate | 0.133 | [0.037, 0.379] | 15 | results/cipher_e0_20260114_175809/summary.json |
| E4 Activation | A0 success_rate | 0.200 | [0.070, 0.452] | 15 | results/activation_grafting_20260114_175810/summary.json |

## M2 Gate Attempts

| Experiment | Status | Normal | Null | Random | Shuffle | Artifact |
| --- | --- | --- | --- | --- | --- | --- |
| M2-SCALE Gate 1 | FAILED | 22.0% (11/50) | 24.0% (12/50) | 28.0% (14/50) | 32.0% (16/50) | docs/experiments/gate1-attempt-02.md |

## Notes

- Some headline docs reference larger sample sizes than currently indexed result files for E3/E4; this index reflects files present in this repository.
- Re-run this script after new experiments:
  - `python scripts/analysis/generate_artifact_index.py`
