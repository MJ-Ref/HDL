# LPCA: Latent-Path Communication for AI Agents

A research framework for evaluating machine-native communication between multi-agent AI systems.

## Overview

This project tests whether multi-agent AI systems can achieve higher capability and reliability by shifting inter-agent state transfer away from human-optimized text tokens and toward **machine-native representations** (expected embeddings, activations/hidden states, learned continuous latent packets, discretized latent tokens).

**Primary Research Question:** At fixed resource budgets (bits communicated, compute overhead, latency), can latent-path communication mechanisms outperform optimized text baselines on coordination-limited multi-agent tasks?

## Key Features

- **Multiple Communication Protocols:** Text baselines (P0-P5), CIPHER expected embeddings (E0), activation grafting (A0), learned codecs (L1-L2)
- **Split-Information Tasks:** Synthetic and code tasks where coordination is genuinely necessary
- **Budget Accounting:** First-class tracking of bits, compute, and latency
- **Safety Instrumentation:** Compliance gap testing, monitor disagreement, covert channel probes
- **Pre-registered Metrics:** All metrics defined before experiments run

## Installation

```bash
# Clone repository
git clone https://github.com/MJ-Ref/HDL.git
cd HDL

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install package
pip install -e .

# Or install with all dependencies
pip install -e ".[full]"
```

## Quick Start

```bash
# Run baseline experiment
python scripts/run_experiment.py --config configs/base.yaml

# Run with specific protocol
python scripts/run_experiment.py --config configs/base.yaml --protocol P1

# Dry run (print config)
python scripts/run_experiment.py --config configs/base.yaml --dry_run
```

## Project Structure

```
lpca/
├── core/           # Configuration, logging, metrics, budget
├── envs/           # Task environments (synthetic, code)
├── channels/       # Communication protocols (text, CIPHER, activation)
├── agents/         # Agent implementations with activation hooks
├── training/       # Codec training utilities
└── safety/         # Safety evaluation modules
```

## Documentation

| Document | Description |
|----------|-------------|
| [PLAN.md](PLAN.md) | Master research plan with milestones |
| [EXPERIMENTS.md](EXPERIMENTS.md) | Detailed experimental protocols |
| [METRICS.md](METRICS.md) | Pre-registered metrics specifications |
| [BASELINES.md](BASELINES.md) | Baseline implementation details |
| [SAFETY_PROTOCOL.md](SAFETY_PROTOCOL.md) | Safety evaluation procedures |
| [PROJECT_STATUS.md](PROJECT_STATUS.md) | Current implementation status |

## Research Design

### Hypotheses (Falsifiable)

1. **H1 (Capability):** Activation communication achieves ≥15% higher success rate than best text baseline at matched compute
2. **H2 (Compression):** Discrete codec retains ≥80% of continuous codec capability at 10× lower bits
3. **H3 (Budget Crossover):** There exists a budget threshold where latent strictly dominates text
4. **H4 (Safety):** Risk indicators change measurably as latent fraction increases

### Protocols Evaluated

| Protocol | Type | Description |
|----------|------|-------------|
| P0 | Text | No communication (lower bound) |
| P1 | Text | Full text (upper reference) |
| P2 | Text | Budgeted text |
| P3 | Text | Text + summarization |
| P4 | Text | Text + retrieval memory |
| P5 | Structured | JSON workspace |
| E0 | Latent | CIPHER expected embeddings |
| A0 | Latent | Activation grafting |
| L1 | Latent | Continuous codec |
| L2 | Latent | Discrete codec (VQ) |

## Hardware Requirements

**Minimum:**
- 32GB RAM + 16GB VRAM (or 64GB unified memory)
- Python 3.11+

**Recommended:**
- Apple M3 Max with 98GB unified memory, OR
- NVIDIA A100 80GB

## Citation

If you use this code in your research, please cite:

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

## Acknowledgments

This research builds on insights from:
- CIPHER (ICLR 2024)
- Communicating Activations Between Language Model Agents (2025)
- InterLAt (2025)
- Anthropic alignment research (Bloom, subliminal learning, alignment faking)
