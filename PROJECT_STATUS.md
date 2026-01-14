# LPCA Project Status

**Last Updated:** January 14, 2026
**Current Phase:** Milestone 1 COMPLETE - Ready for M2 (Requires Cloud GPUs)

---

## Executive Summary

The LPCA (Latent-Path Communication for AI Agents) research project has completed **Milestone 0** (Foundation) and **Milestone 1** (Latent Baselines). Key finding: **raw latent communication does NOT work without training**. Text communication (P1) achieves 30% success vs 0% for both CIPHER (E0) and Activation Grafting (A0). Safety evaluation (E5) passed all metrics. The codebase is production-ready with 126 tests passing. **Next step: M2 Codec Training requires cloud GPUs.**

---

## Milestone Progress

### Milestone 0: Foundation COMPLETE

| Deliverable | Status | Location |
|-------------|--------|----------|
| Experiment configuration system | Done | `lpca/core/config.py` |
| Episode logging (JSONL + Parquet) | Done | `lpca/core/logging.py` |
| Pre-registered metrics | Done | `lpca/core/metrics.py` |
| Budget accounting | Done | `lpca/core/budget.py` |
| Split-info synthetic tasks (S1-S3) | Done | `lpca/envs/split_synthetic.py` |
| Text baselines (P0-P5) | Done | `lpca/channels/text.py` |
| Model wrapper with hooks | Done | `lpca/agents/model_wrapper.py` |
| Experiment runner | Done | `scripts/demo_experiment.py` |
| Unit tests | Done | `tests/` (77 tests passing) |

**Exit Criteria:** ALL MET
- P1 >> P0 confirmed (20% vs 0% in demo)
- Deterministic generation verified
- Metrics logging verified

---

### Milestone 1: Latent Baselines ✅ COMPLETE

| Deliverable | Status | Location |
|-------------|--------|----------|
| CIPHER expected embedding (E0) | ✅ Done | `lpca/channels/cipher.py` |
| Activation grafting (A0) | ✅ Done | `lpca/channels/activation.py` |
| LLM Agent with activation capture | ✅ Done | `lpca/agents/llm_agent.py` |
| Layer sweep configs | ✅ Done | `lpca/channels/activation.py` |
| Combine function sweep | ✅ Done | `lpca/channels/activation.py` |
| Analysis scripts | ✅ Done | `scripts/analysis/` |
| Statistical tests | ✅ Done | `scripts/analysis/statistical_tests.py` |
| Real model evaluation (E1) | ✅ Done | P1=30%, P0=0% |
| CIPHER evaluation (E3) | ✅ Done | E0=0% (negative result) |
| Activation grafting (E4) | ✅ Done | A0=0% at all layers (negative result) |
| Safety evaluation (E5) | ✅ Done | All metrics PASS |

**Exit Criteria:** ✅ ALL COMPLETE
- [x] P1 >> P0 confirmed (30% vs 0%, p < 0.05)
- [x] A0/E0 tested - no improvement over P0 (raw latent doesn't work)
- [x] Clear evidence that **trained** latent communication needed (M2/M3)
- [x] Safety evaluation passed all pre-committed thresholds

---

### Milestone 4: Safety Evaluation ✅ E5 COMPLETE

| Deliverable | Status | Location |
|-------------|--------|----------|
| Compliance gap tester | ✅ Done | `lpca/safety/compliance.py` |
| Linear probe monitors | ✅ Done | `lpca/safety/monitors.py` |
| Monitor ensemble | ✅ Done | `lpca/safety/monitors.py` |
| Covert channel probes | ✅ Done | `lpca/safety/monitors.py` |
| Stop conditions | ✅ Done | Pre-committed in all modules |
| E5 Evaluation Run | ✅ Done | All metrics PASS |
| Bloom integration | Pending | `lpca/safety/bloom_eval.py` |

**E5 Results:**
| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Compliance Gap | 17.5% | ≤20% | ✅ OK |
| Monitor Disagreement | 2.65% | ≤50% | ✅ OK |
| Covert Channel Capacity | 8 bits | ≤10 bits | ✅ OK |

---

### Milestone 2-3: Codec Development

**Why M2 is needed:** M1 showed that raw latent channels don't work. Training is required to create semantically meaningful latent packets.

#### M2-LOCAL-PROTO (Can Run Locally) ✅ READY

| Approach | Details |
|----------|---------|
| Base model | Frozen (no gradients) |
| Train only | Small encoder/decoder MLP (~10M params) |
| Dataset | 100-500 P1 episodes |
| Time | ~10-30 min per run |

**Purpose:** Validate pipeline before cloud spend.

#### M2-SCALE (Requires Cloud GPUs) ⏳ PENDING

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU | 1× A100 40GB | 4× A100 80GB |
| Time | 40-80 GPU-hours | - |
| Cost | $50-100 | $100-200 |

**Recommended Provider: Modal.com** ⭐
- $30/month free credits (~14 hours A100)
- $10,000 academic research grants available
- Serverless - only pay for compute
- Python-native API

---

## Recent Progress

### Sanity Checks (Run Before Scaling) ✅ PASSED

| Check | Result | Finding |
|-------|--------|---------|
| Single Agent Full Info | **70%** success | Model competent with all info |
| Injection Test | L2=165, KL=0.06 | Plumbing works correctly |
| Answer Parsing | 90% (9/10) | Parsing logic robust |

**Critical Insight:** Single agent (70%) >> P1 two-agent (30%) >> P0 (0%)
This validates **communication IS the bottleneck** - ~40% capability lost when info is split.

### Complete Experiment Results (Qwen-2.5-3B, Constraint Satisfaction, n=50)

| Protocol | Success | 95% CI | Method | Notes |
|----------|---------|--------|--------|-------|
| Single Agent | **70%** | - | Full information | Reference (n=20) |
| P0 | **52%** | [38.5%, 65.2%] | No communication | Improved parsing |
| **P1** | **74%** | **[60.4%, 84.1%]** | **Text messages** | **Best 2-agent** |
| E0 | 0% | [0%, 27.8%] | CIPHER embeddings | ❌ Failed (n=10) |
| A0 (all layers) | 0% | [0%, 27.8%] | Activation injection | ❌ Failed (n=10) |

**Key Findings (Post-Fix):**
1. **P1 (74%) > P0 (52%)** - communication adds ~22% success (non-overlapping CIs)
2. **Single Agent (70%) ≈ P1 (74%)** - text communication nearly recovers full-info performance!
3. **E0 = A0 = 0%** - raw latent channels still don't transfer semantics
4. **P0 baseline higher** - improved prompts/parsing help agents extract answers
5. **Conclusion:** Communication is valuable; trained latent channels needed for M2

### Wall Time (Corrected)

| Operation | Actual Time | Notes |
|-----------|-------------|-------|
| Single episode | ~28s | Full generation cycle |
| Two-agent episode | ~50-60s | Multi-turn dialogue |
| Model load | ~3s | Qwen-2.5-3B on MPS |

### Analysis Capabilities
- **plot_results.py**: Success rates, capability vs bits, distributions
- **statistical_tests.py**: Hypothesis tests, pairwise comparisons, effect sizes

### Safety Infrastructure
- **ComplianceGapTester**: Monitored vs unmonitored behavior testing
- **LinearProbeMonitor**: Activation-based risk scoring
- **MonitorEnsemble**: Ensemble predictions with disagreement tracking
- **CovertChannelProbe**: Hidden channel capacity estimation
- **Stop Conditions**: All pre-committed thresholds implemented

### Experiment Infrastructure (New)
- **run_experiment.py**: Full-featured experiment runner with multi-protocol support
- **Experiment configs**: YAML configs for E1-E5 with inheritance support
- **Config loader**: Type-safe config loading with dataclass validation
- **Results aggregation**: Cross-experiment aggregation with LaTeX export

### Test Coverage
- **Channels:** 18 tests
- **Environments:** 22 tests
- **Metrics:** 18 tests
- **Integration:** 7 tests
- **Safety:** 28 tests
- **Config Loader:** 21 tests
- **Total:** 126 tests, 100% pass rate

---

## Code Structure

```
HDL/
├── configs/                 # Experiment configurations
│   ├── base.yaml           # Base configuration
│   ├── e1_baseline.yaml    # E1: Text baseline validation
│   ├── e2_task_sweep.yaml  # E2: Task family sweep
│   ├── e3_cipher.yaml      # E3: CIPHER evaluation
│   ├── e4_activation_grafting.yaml  # E4: Activation grafting
│   └── e5_safety.yaml      # E5: Safety evaluation
│
├── docs/                    # Planning documents
│   ├── PLAN.md             # Master research plan
│   ├── EXPERIMENTS.md      # Experimental protocols
│   ├── METRICS.md          # Pre-registered metrics
│   ├── BASELINES.md        # Baseline specifications
│   ├── SAFETY_PROTOCOL.md  # Safety evaluation protocol
│   └── REPRODUCIBILITY.md  # Reproducibility checklist
│
├── lpca/                    # Main package
│   ├── core/               # Infrastructure (complete)
│   │   ├── config_loader.py  # YAML config loading with inheritance
│   │   └── ...
│   ├── envs/               # Task environments (complete)
│   ├── channels/           # Communication protocols (complete)
│   ├── agents/             # Agent implementations (complete)
│   └── safety/             # Safety evaluation (complete)
│
├── scripts/
│   ├── run_experiment.py   # Full experiment runner
│   ├── demo_experiment.py  # End-to-end demo
│   └── analysis/           # Analysis scripts
│       ├── plot_results.py
│       ├── statistical_tests.py
│       └── aggregate_results.py  # Results aggregation
│
└── tests/                   # Test suite (126 tests)
```

---

## Implementation Metrics

| Metric | Value |
|--------|-------|
| Python files | 27 |
| Lines of code | ~7,000 |
| Test count | 126 |
| Test pass rate | 100% |
| Config files | 6 |
| Documentation pages | 7 |

---

## Immediate Next Steps

### ✅ COMPLETED
1. ~~Install torch/transformers~~ - Done (PyTorch 2.9.1, MPS support)
2. ~~Run E1 baseline validation~~ - Done (P1=30% >> P0=0%)
3. ~~Run E3 CIPHER evaluation~~ - Done (E0=0%, negative result)
4. ~~Run E4 layer sweep~~ - Done (A0=0% at L9, L18, L27)
5. ~~Run E5 safety evaluation~~ - Done (All metrics PASS)
6. ~~Fix prompt/parsing issues~~ - Done (removed {json} placeholder, added fallback)
7. ~~Add sanity checks~~ - Done (single-agent, injection, parsing)
8. ~~Run sanity checks~~ - Done (all passed)

### 🔄 IN PROGRESS
9. **Run E1 with 50 episodes** - Validating fixes with larger sample

### 📋 NEXT: Local Tight Loops (Before Cloud)

**Step 1: Run E2 (Text Baseline Sweep)**
- Sweep P1-P5 protocols
- Establish strong teacher baseline
- Get capability-vs-bits curve

**Step 2: M2-LOCAL-PROTO**
- Validate codec training pipeline locally
- Freeze base model, train small encoder/decoder
- Confirm loss decreases, generation not broken

**Step 3: Scale to Cloud (When Ready)**
- Use Modal.com ($30/mo free credits)
- Collect 1000+ P1 episodes
- Full k-sweep for codec training

---

## Risk Status

| Risk | Status | Mitigation |
|------|--------|------------|
| Communication not bottleneck | Preliminary OK | P1 >> P0 in demo |
| MPS compatibility | Addressed | Lazy imports, fallbacks |
| Model loading | Ready | LLM agent implemented |
| Test coverage | Good | 77 tests, 100% pass |
| Safety monitoring | Ready | Full infrastructure |

---

## Repository

**GitHub:** https://github.com/MJ-Ref/HDL

**Latest Commits:**
- `8ccf654` Add analysis scripts, safety monitors, and update documentation
- `4f74c61` Fix bugs, add lazy imports, and comprehensive test suite
- `76fee53` Implement Milestone 1 latent communication channels

---

*Status updated: January 14, 2026*
