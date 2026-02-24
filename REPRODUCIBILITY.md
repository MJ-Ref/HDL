# LPCA Reproducibility Guide

**Version:** 2.0  
**Scope:** Canonical full-suite pipeline (E1/E2/E3/E4/M2) and publication artifact generation.

This file documents how to reproduce LPCA results using the current frozen benchmark workflow.

## 1. Reproducibility Principles

LPCA reproducibility is artifact-first:

- Every run is represented by a manifest (`manifest.json`) with exact commands, statuses, and discovered artifacts.
- Analysis and paper tables are generated from manifests and artifact files, not manual copy/paste.
- Final reproducibility package is emitted automatically from the same run directory.

## 2. Frozen Benchmark Specification

Canonical config:

- `configs/full_suite_frozen.yaml`
- `configs/seeds/full_suite_seed_registry.json`

Current frozen matrix:

- Models: `qwen_3b`, `qwen_7b`, `mistral_7b`
- Episode targets: `E1=100`, `E2=100`, `E3=100`, `E4=100`, `M2_eval=100`
- Matrix guardrails: at least `3` models and `100` seeds/episodes for matrix runs

## 3. Environment Setup

```bash
# Clone
git clone https://github.com/MJ-Ref/HDL.git
cd HDL

# Python env
python -m venv .venv
source .venv/bin/activate

# Install
pip install -r requirements.txt
pip install -e .

# Sanity check
pytest -q
```

Notes:

- `setup.py` requires Python `>=3.11`.
- M2 experiments run via Modal and require Modal auth/setup (`docs/MODAL_SETUP.md`).

## 4. Canonical Run Commands

### 4.1 Dry-run (recommended first)

```bash
python scripts/run_full_suite.py \
  --config configs/full_suite_frozen.yaml
```

### 4.2 Full execution

```bash
python scripts/run_full_suite.py \
  --config configs/full_suite_frozen.yaml \
  --run-id run_lpca_full_suite \
  --stages prepare,run,aggregate,gate,paper,render,package,publish \
  --execute \
  --stop-on-error
```

### 4.3 Focused M2 matrix execution

```bash
python scripts/run_full_suite.py \
  --run-id run_m2_only \
  --models qwen_3b,qwen_7b,mistral_7b \
  --experiments M2 \
  --stages prepare,run,aggregate,gate,paper,render,package,publish \
  --execute \
  --stop-on-error
```

## 5. Run Directory Contract

Each run writes to `outputs/full_suite/<run_id>/`.

Expected top-level outputs:

- `manifest.json`
- `summary.md`
- `resolved_config.json`
- `resolved_seed_registry.json`
- `suite_report.json`, `suite_report.md`
- `m2_gate_report.json`, `m2_gate_report.md`
- `paper_pack/`
- `repro_package/`

Artifacts inside `paper_pack/`:

- `main_table.csv`
- `main_table.md`
- `m2_gate_table.md`
- `figure_data.json`
- `paper_pack_manifest.json`

Artifacts inside `repro_package/`:

- `RUNBOOK.md`
- `environment_lock.txt`
- `reproduce_main_table.sh`
- `repro_package_manifest.json`

## 6. Integrity Checks

Run these checks after execution:

```bash
# 1) No failed commands in manifest
jq '[.commands[] | select(.status=="failed")] | length' outputs/full_suite/<run_id>/manifest.json

# 2) Publish stage completed and failure flag is false
jq '{stages_completed,has_failures}' outputs/full_suite/<run_id>/manifest.json

# 3) Package warnings should be zero for clean publication runs
jq '.warnings | length' outputs/full_suite/<run_id>/repro_package/repro_package_manifest.json
```

Expected:

- failed-command count is `0`
- `has_failures` is `false`
- package warning count is `0`

## 7. Reproduce Main Table from an Existing Run

From the generated package:

```bash
bash outputs/full_suite/<run_id>/repro_package/reproduce_main_table.sh <run_id>
```

This regenerates:

- `suite_report.json` / `suite_report.md`
- `paper_pack/main_table.csv` and related paper-pack outputs

## 8. Modal and External Dependency Notes

- M2 uses Modal (`modal/train_m2_codec.py`) and cloud model downloads, so wall-clock runtime and provider scheduling can vary.
- Text/latent metrics can show minor stochastic variation across hardware/software updates; enforce fixed seeds and frozen config for comparisons.
- For claim-level traceability, use `docs/ARTIFACT_INDEX.md` plus run-specific manifests.

## 9. Figure-Pack Extension (Optional)

LPCA includes a PaperBanana adapter for publication figure generation:

```bash
# Build 3 figure briefs from a run
python scripts/analysis/generate_paperbanana_figure_pack.py \
  --run-dir outputs/full_suite/<run_id>

# Optionally execute PaperBanana generation
python scripts/analysis/generate_paperbanana_figure_pack.py \
  --run-dir outputs/full_suite/<run_id> \
  --paperbanana-root /Users/mj/Documents/PaperBanana/PaperBanana-main \
  --execute \
  --exp-mode dev_full \
  --retrieval-setting none \
  --candidates 3
```

## 10. What This Guide Replaces

This version supersedes earlier pre-registered templates that referenced deprecated sweep scripts and hypothetical release plans. The source of truth is now the artifact-backed canonical pipeline in `scripts/run_full_suite.py`.
