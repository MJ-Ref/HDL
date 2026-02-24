# LPCA: Latent-Path Communication for AI Agents

A research framework for evaluating machine-native communication in multi-agent LLM systems.

[![Tests](https://img.shields.io/badge/tests-147%20passing-brightgreen)](tests/)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](setup.py)

## Overview

LPCA studies whether multi-agent systems can outperform text-only coordination by exchanging machine-native signals under fixed budgets (bits, latency, compute).

Primary question:

- At matched budgets, can latent-path communication (expected embeddings, activations, learned codecs) beat strong text baselines on coordination-limited tasks?

## Current Status

Publication track status (as of the current codebase):

- Canonical benchmark orchestrator is implemented (`scripts/run_full_suite.py`).
- Frozen benchmark config is in place (`configs/full_suite_frozen.yaml`) with matrix requirements (`>=3` models, `>=100` seeds/episodes).
- M2v2 codec path is integrated in `modal/train_m2_codec.py` (discrete bottleneck + MI/curriculum loss components).
- Submission artifacts are auto-generated: suite report, gate report, paper pack, reproducibility package.
- Latest archived M2 gate attempts in `docs/ARTIFACT_INDEX.md` remain below publication gate targets; reruns are required for positive claims.

## Installation

```bash
# Clone
git clone https://github.com/MJ-Ref/HDL.git
cd HDL

# Environment
python -m venv .venv
source .venv/bin/activate

# Install deps + package
pip install -r requirements.txt
pip install -e .
```

## Quick Start

```bash
# Verify environment
pytest -q

# Dry-run the canonical suite (prints planned commands and writes run scaffolding)
python scripts/run_full_suite.py
```

## Canonical Benchmark Pipeline

Primary entrypoint:

```bash
python scripts/run_full_suite.py \
  --config configs/full_suite_frozen.yaml \
  --run-id run_lpca_full_suite \
  --stages prepare,run,aggregate,gate,paper,render,package,publish \
  --execute \
  --stop-on-error
```

Notes:

- Without `--execute`, the orchestrator runs in dry-run mode.
- Long runs can be resumed with `--run-id <id> --resume` (completed commands are skipped).
- Command logs are streamed live to `outputs/full_suite/<run_id>/logs/.../command.log`.
- M2 uses `modal/train_m2_codec.py` and is intended for cloud GPUs.
- Full matrix requirements are enforced unless `--skip-matrix-requirements` is set.

### Useful scoped runs

```bash
# Cloud E1-E4 matrix on Modal (avoids local GPU/CPU load)
python scripts/run_full_suite.py \
  --config configs/full_suite_modal_e1e4.yaml \
  --run-id run_e1e4_modal_matrix \
  --models qwen_3b,qwen_7b,mistral_7b \
  --experiments E1,E2,E3,E4 \
  --stages prepare,run,aggregate,render,publish \
  --execute \
  --stop-on-error

# Only M2 across the frozen model set
python scripts/run_full_suite.py \
  --run-id run_m2_only \
  --models qwen_3b,qwen_7b,mistral_7b \
  --experiments M2 \
  --stages prepare,run,aggregate,gate,paper,render,package,publish \
  --execute \
  --stop-on-error

# Fast publication-stage smoke (no expensive run stage)
python scripts/run_full_suite.py \
  --run-id run_publication_smoke \
  --models qwen_3b \
  --experiments E1 \
  --skip-matrix-requirements \
  --stages prepare,aggregate,gate,paper,render,package,publish \
  --execute
```

## Output Artifacts

Each run writes to `outputs/full_suite/<run_id>/`.

Core files:

- `manifest.json`: canonical command + status ledger.
- `suite_report.json` / `suite_report.md`: preregistered stats report.
- `m2_gate_report.json` / `m2_gate_report.md`: CI-backed M2 gate report.
- `paper_pack/`: manuscript-facing tables + figure data.
- `repro_package/`: runbook, lockfile, and `reproduce_main_table.sh`.

Global index:

- `docs/ARTIFACT_INDEX.md` maps key claims to concrete files.

## Paper Figure Generation (PaperBanana Adapter)

LPCA includes a bridge script to generate manuscript figure briefs (and optional PaperBanana renders):

```bash
# Generate figure briefs from a run
python scripts/analysis/generate_paperbanana_figure_pack.py \
  --run-dir outputs/full_suite/run_lpca_full_suite

# Optional: execute PaperBanana generation (requires PaperBanana setup + API keys)
python scripts/analysis/generate_paperbanana_figure_pack.py \
  --run-dir outputs/full_suite/run_lpca_full_suite \
  --paperbanana-root /Users/mj/Documents/PaperBanana/PaperBanana-main \
  --execute \
  --exp-mode dev_full \
  --retrieval-setting none \
  --candidates 3
```

## Documentation

- [REPRODUCIBILITY.md](REPRODUCIBILITY.md): exact reproducibility workflow and checks.
- [PLAN.md](PLAN.md): research roadmap and gate logic.
- [EXPERIMENTS.md](EXPERIMENTS.md): protocol details.
- [docs/STEPWISE_EXECUTION_PLAN.md](docs/STEPWISE_EXECUTION_PLAN.md): execution checklist.
- [docs/ARTIFACT_INDEX.md](docs/ARTIFACT_INDEX.md): claim-to-artifact mapping.
- [docs/MODAL_SETUP.md](docs/MODAL_SETUP.md): Modal setup and M2 execution notes.
- [PROJECT_STATUS.md](PROJECT_STATUS.md): broader project status notes.

## Citation

```bibtex
@software{lpca2026,
  title={LPCA: Latent-Path Communication for AI Agents},
  author={LPCA Research Team},
  year={2026},
  url={https://github.com/MJ-Ref/HDL}
}
```

## License

Apache 2.0
