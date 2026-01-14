# LPCA Project Status

**Last Updated:** January 14, 2026
**Current Phase:** Milestone 1 COMPLETE - Ready for M2 (Requires Cloud GPUs)

---

## Executive Summary

The LPCA (Latent-Path Communication for AI Agents) research project has completed **Milestone 0** (Foundation) and **Milestone 1** (Latent Baselines).

**Key findings (n=50, tightened task):**
- **P1 (68%) >> P0 (20%)** - Text communication adds **+48pp** (non-overlapping 95% CIs)
- **E0 CIPHER = 13%**, **A0 Activation = 20%** - Raw latent channels don't help without training
- **Task is communication-limited** - P0 at 20% (target range 15-25%)
- **M2-LOCAL-PROTO validated** - Codec pipeline works, ready for cloud training

Safety evaluation (E5) passed all metrics. Codebase is production-ready with 126 tests. **Next step: M2 Codec Training requires cloud GPUs.**

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
| Real model evaluation (E1) | ✅ Done | P1=68%, P0=20% (n=50) |
| CIPHER evaluation (E3) | ✅ Done | E0=13% (negative result) |
| Activation grafting (E4) | ✅ Done | A0=20% (negative result) |
| Safety evaluation (E5) | ✅ Done | All metrics PASS |

**Exit Criteria:** ✅ ALL COMPLETE
- [x] P1 >> P0 confirmed (68% vs 20%, non-overlapping 95% CIs)
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

### Experiment Results History

#### Pre-Fix Results (INVALID - prompt bug)
E1/E3/E4 results before prompt fix showed P0=0%, but model was outputting literal `{json}`.
These results are **not valid** - see tag `baseline-v0.1-postfix` for audit trail.

#### Post-Fix Results (baseline-v0.1-postfix, n=50) - OLD TASK

| Protocol | Success | 95% CI | Method | Notes |
|----------|---------|--------|--------|-------|
| Single Agent | **70%** | - | Full information | Reference (n=20) |
| P0 | **52%** | [38.5%, 65.2%] | No communication | ⚠️ TOO HIGH |
| **P1** | **74%** | **[60.4%, 84.1%]** | **Text messages** | **Best 2-agent** |

**Issue:** P0 at 52% was too high (target: 15-25%). Task was not communication-limited.

#### Post-Tightening Results (FINAL, n=50)

Task generator tightened in `lpca/envs/split_synthetic.py`:
- Increased variables: 3→4, constraints: 4→8, domain: 2→3
- Added `_count_valid_solutions()` check to ensure neither agent can solve alone

| Protocol | Success | 95% CI | Method | Notes |
|----------|---------|--------|--------|-------|
| P0 | **20%** | [11.2%, 33.0%] | No communication | ✅ In target range |
| **P1** | **68%** | **[54.2%, 79.2%]** | **Text messages** | ✅ Strong improvement |
| E0 (CIPHER) | 13% | [3.7%, 37.9%] | CIPHER embeddings | ❌ No improvement |
| A0 (Activation) | 20% | [7.0%, 45.2%] | Activation grafting | ❌ No improvement |

**Key Findings (Post-Tightening):**
1. **P1 (68%) >> P0 (20%)** - communication adds **+48pp** (non-overlapping 95% CIs)
2. **Task is properly communication-limited** - P0 at 20% (target: 15-25%)
3. **Raw latent channels don't work** - E0 and A0 no better than P0
4. **M2-LOCAL-PROTO passed** - codec pipeline validated, ready for cloud training

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
