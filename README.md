# LPCA: Latent-Path Communication for AI Agents

A research framework for evaluating machine-native communication between multi-agent AI systems.

[![Tests](https://img.shields.io/badge/tests-126%20passing-brightgreen)](tests/)
[![Milestone](https://img.shields.io/badge/E1-validated-success)](PROJECT_STATUS.md)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](setup.py)

## Overview

This project tests whether multi-agent AI systems can achieve higher capability and reliability by shifting inter-agent state transfer away from human-optimized text tokens and toward **machine-native representations** (expected embeddings, activations/hidden states, learned continuous latent packets, discretized latent tokens).

**Primary Research Question:** At fixed resource budgets (bits communicated, compute overhead, latency), can latent-path communication mechanisms outperform optimized text baselines on coordination-limited multi-agent tasks?

## Current Status

**Milestone 1: Latent Baselines** - E1 Baseline Validated, E4 In Progress

| Component | Status |
|-----------|--------|
| Text Baselines (P0-P5) | Complete |
| CIPHER Channel (E0) | Complete |
| Activation Grafting (A0) | Complete |
| LLM Integration | Complete (Qwen-2.5-3B) |
| Test Suite | 126 tests passing |
| E1 Baseline | **Validated** |

**E1 Baseline Results (Qwen-2.5-3B, n=20):**
```
Protocol   Success    95% CI           Partial    Turns
---------------------------------------------------------
P0         0.0%       [0.0%, 16.1%]    0.000      12.0
P1         30.0%      [14.5%, 51.9%]   0.338      9.5
```
**Key Finding:** Communication significantly improves success (P1 >> P0, p < 0.05)

## Key Features

- **Multiple Communication Protocols:** Text baselines (P0-P5), CIPHER expected embeddings (E0), activation grafting (A0), learned codecs (L1-L2)
- **Split-Information Tasks:** Synthetic tasks where coordination is genuinely necessary
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

# Install minimal dependencies (no torch required for mock experiments)
pip install -e .

# Install with torch for real LLM experiments
pip install -e ".[full]"
```

## Quick Start

```bash
# Run demo experiment with mock agents (no torch needed)
python scripts/demo_experiment.py --mock

# Run with specific task type
python scripts/demo_experiment.py --mock --task arithmetic

# Run unit tests
python -m pytest tests/ -v
```

### With Real LLM (requires torch)

```bash
# Install PyTorch and transformers
pip install torch transformers accelerate

# Run E1 baseline experiment
python scripts/run_llm_experiment.py --protocols P0,P1 --n_episodes 20

# Run activation grafting experiment
python scripts/run_activation_experiment.py --layer 18 --n_episodes 10

# Run with Llama model
python scripts/demo_experiment.py --model meta-llama/Llama-3.2-1B-Instruct
```

## Project Structure

```
lpca/
├── core/           # Configuration, logging, metrics, budget
│   ├── config.py   # Experiment configuration
│   ├── logging.py  # Episode logging (JSONL + Parquet)
│   ├── metrics.py  # Pre-registered metrics
│   └── budget.py   # Budget accounting
│
├── envs/           # Task environments
│   ├── base.py     # Abstract interface
│   └── split_synthetic.py  # S1-S3 tasks
│
├── channels/       # Communication protocols
│   ├── base.py     # Channel interface
│   ├── text.py     # P0-P5 text baselines
│   ├── cipher.py   # E0 CIPHER channel
│   └── activation.py  # A0 activation grafting
│
├── agents/         # Agent implementations
│   ├── base.py     # Agent interface
│   ├── model_wrapper.py  # Activation hooks
│   └── llm_agent.py  # LLM-based agent
│
├── training/       # Codec training (planned)
└── safety/         # Safety evaluation (planned)
```

## Documentation

| Document | Description |
|----------|-------------|
| [PLAN.md](PLAN.md) | Master research plan with milestones |
| [EXPERIMENTS.md](EXPERIMENTS.md) | Detailed experimental protocols |
| [docs/experiments/](docs/experiments/) | Versioned experiment attempts with decision logs |
| [docs/ARTIFACT_INDEX.md](docs/ARTIFACT_INDEX.md) | Artifact-level mapping from headline numbers to files |
| [METRICS.md](METRICS.md) | Pre-registered metrics specifications |
| [BASELINES.md](BASELINES.md) | Baseline implementation details |
| [SAFETY_PROTOCOL.md](SAFETY_PROTOCOL.md) | Safety evaluation procedures |
| [PROJECT_STATUS.md](PROJECT_STATUS.md) | Current implementation status |
| [SESSION_HANDOFF.md](SESSION_HANDOFF.md) | Session continuity notes |

## Claude Code Integration

This repo includes workflow tools for [Claude Code](https://claude.ai/code):

- **`/verify-docs`** - Custom skill to check documentation consistency across PLAN.md, PROJECT_STATUS.md, and SESSION_HANDOFF.md
- **Post-edit hook** - Automatically runs `ruff check --fix` on Python files after edits

Configuration lives in `.claude/`.

## Research Design

### Hypotheses (Falsifiable)

1. **H1 (Capability):** Activation communication achieves ≥15% higher success rate than best text baseline at matched compute
2. **H2 (Compression):** Discrete codec retains ≥80% of continuous codec capability at 10× lower bits
3. **H3 (Budget Crossover):** There exists a budget threshold where latent strictly dominates text
4. **H4 (Safety):** Risk indicators change measurably as latent fraction increases

### Protocols Evaluated

| Protocol | Type | Description | Status |
|----------|------|-------------|--------|
| P0 | Text | No communication (lower bound) | Complete |
| P1 | Text | Full text (upper reference) | Complete |
| P2 | Text | Budgeted text | Complete |
| P3 | Text | Text + summarization | Complete |
| P4 | Text | Text + retrieval memory | Complete |
| P5 | Structured | JSON workspace | Complete |
| E0 | Latent | CIPHER expected embeddings | Complete |
| A0 | Latent | Activation grafting | Complete |
| L1 | Latent | Continuous codec | Planned |
| L2 | Latent | Discrete codec (VQ) | Planned |

## Task Families

### S1: Constraint Satisfaction
- Agent A receives half the constraints
- Agent B receives the other half
- Solution requires satisfying all constraints

### S2: Arithmetic with Missing Operands
- Agent A sees some variables
- Agent B sees others
- Must compute function requiring all values

### S3: Program Synthesis (Toy)
- Agent A sees input-output examples
- Agent B sees additional test cases
- Must produce correct implementation

## Hardware Requirements

**Minimum:**
- 16GB RAM
- Python 3.10+

**For Real LLM Experiments:**
- 32GB RAM + GPU with 16GB VRAM, OR
- Apple M-series with 64GB+ unified memory

**Recommended:**
- Apple M3 Max with 98GB unified memory, OR
- NVIDIA A100 80GB

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_channels.py -v

# Run with coverage
python -m pytest tests/ --cov=lpca --cov-report=html
```

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
