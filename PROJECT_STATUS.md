# LPCA Project Status

**Last Updated:** January 15, 2026
**Current Phase:** M2-SCALE Gate 1 INVALID - Plumbing Proof FAIL, Training Objective Wrong

---

## Executive Summary

The LPCA (Latent-Path Communication for AI Agents) research project has completed **Milestone 0** (Foundation), **Milestone 1** (Latent Baselines), and **E2-min** (Budgeted Text Baselines).

**M2-SCALE Gate 1 results are INVALID** due to critical bugs identified during review.

**Key findings (n=50, tightened task):**
- **P1 (68%) >> P0 (20%)** - Text communication adds **+48pp** (non-overlapping 95% CIs)
- **E0 CIPHER = 13%**, **A0 Activation = 20%** - Raw latent channels don't help without training
- **P5 (structured) dominates P2 (raw text)** at all budgets
- **P5_16B = 56.7% at ~43 bits** - Target for codec at k=4

**M2-SCALE Gate 1 Results (January 15, 2026) - ⚠️ INVALID:**
| Config | Final Loss | Success Rate | Gate 1 Status |
|--------|-----------|--------------|---------------|
| k=4 | 0.097 | 34.0% | ⚠️ **INVALID** |
| k=8 | 0.111 | 38.0% | ⚠️ **INVALID** |

**Why Invalid:**
1. Latent vectors computed but never injected (Agent B always sees text)
2. Placeholder fallback can return ~30% success even if env fails to load
3. Missing shuffle ablation (required by spec)
4. Point-estimate gating with n=50 is too noisy

**Codec Targets (from E2-min):**
| Target | Success | Bits | Codec k |
|--------|---------|------|---------|
| P5_16B | 56.7% | ~43 | k=4 (~32 bits) |
| P5_64B | 60.0% | ~214 | k=16 (~128 bits) |
| P5_256B | 66.7% | ~2132 | k=64 (~512 bits) |

Safety evaluation (E5) passed all metrics. **Next step: Fix critical bugs, rerun Gate 1 with validity checks.**

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

#### M2-LOCAL-PROTO (Can Run Locally) ✅ COMPLETE

| Approach | Details |
|----------|---------|
| Base model | Frozen (no gradients) |
| Train only | Small encoder/decoder MLP (~10M params) |
| Dataset | 100-500 P1 episodes |
| Time | ~10-30 min per run |

**Purpose:** Validate pipeline before cloud spend. ✅ PASSED

#### M2-SCALE Gate 1 ⚠️ INVALID

| Config | Final Loss | Normal | Null Msg | Random Latent | Gate 1 |
|--------|-----------|--------|----------|---------------|--------|
| k=4 | 0.097 | 34.0% | 30.0% | 38.0% | ⚠️ INVALID |
| k=8 | 0.111 | 38.0% | 36.0% | 42.0% | ⚠️ INVALID |

**Training completed but evaluation invalid.** Logs: `logs/m2_gate1/`

**Plumbing Proof Results (January 15, 2026) - ❌ FAIL:**

| Condition | Success Rate | Expected |
|-----------|-------------|----------|
| P1 Baseline (text, no injection) | **70.0%** | ~68% ✓ |
| L1 Injection (no text, injection on) | **0.0%** | Should match P1 |
| P0 Reference | ~20% | ~20% |

**Root Cause: Wrong Training Objective**

The codec training uses MSE reconstruction loss, which trains geometric similarity
instead of semantic usefulness. The codec is an autoencoder, not a communication channel.

| Issue | Severity | Status |
|-------|----------|--------|
| Placeholder fallback | P0 | ✅ Fixed |
| Text leakage | P1 | ✅ Fixed |
| Missing shuffle | P1 | ✅ Fixed |
| KV cache generation | P1 | ✅ Fixed |
| Point-estimate gating | P2 | ✅ Fixed |
| **Training objective** | **P0** | ❌ **BLOCKER** - Need distillation-based training |

**Required Fix:** Replace MSE reconstruction loss with KL distillation:
```python
# Train codec so receiver with prefix matches receiver with text
teacher_logits = model(text_prompt)  # Agent B sees message
student_logits = model(prefix_prompt)  # Agent B sees injected prefix
loss = F.kl_div(student_logits, teacher_logits)
```

#### M2-SCALE Gate 2 ⏳ BLOCKED

| Gate | Criterion | Status |
|------|-----------|--------|
| Gate 2 | L1@k=16 ≥ 50% of P1 (≥34%) | Blocked on Gate 1 rerun |

**Next:** Fix training objective (KL distillation) → Retrain codec → Plumbing proof → Gate 1 rerun

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

#### E2-min Results (Budgeted Text Baselines, n=30)

Tested P2 (raw text budget) and P5 (structured) at 16B, 64B, 256B budgets.

| Config | Success | 95% CI | Avg Bits | Notes |
|--------|---------|--------|----------|-------|
| **P5_256B** | **66.7%** | [48.8%, 80.8%] | 2132 | Matches P1 |
| P2_256B | 66.7% | [48.8%, 80.8%] | 4817 | |
| **P5_64B** | **60.0%** | [42.3%, 75.4%] | 214 | Codec target |
| **P5_16B** | **56.7%** | [39.2%, 72.6%] | 43 | Codec target |
| P2_64B | 33.3% | [19.2%, 51.2%] | 2938 | |
| P2_16B | 0.0% | [0.0%, 11.4%] | 1536 | Too constrained |

**Key Insights:**
1. **P5 (structured) dominates P2 (raw text)** at all budgets
2. **P5_16B achieves 56.7% with only ~43 bits** - codec target for k=4
3. **P5_256B matches P1** - structure is highly efficient
4. **Rate-distortion curve established** for codec evaluation

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
2. ~~Run E1 baseline validation~~ - Done (P1=68% >> P0=20%)
3. ~~Run E3 CIPHER evaluation~~ - Done (E0=13%, negative result)
4. ~~Run E4 layer sweep~~ - Done (A0=20%, negative result)
5. ~~Run E5 safety evaluation~~ - Done (All metrics PASS)
6. ~~Fix prompt/parsing issues~~ - Done
7. ~~Run E2-min (budgeted text)~~ - Done (P5_16B=56.7% at 43 bits)
8. ~~M2-LOCAL-PROTO~~ - Done (pipeline validated)
9. ⚠️ M2-SCALE Gate 1 (k=4, k=8) - **INVALID** (critical bugs, rerun required)

### ✅ FIXES IMPLEMENTED
10. ~~Fix critical bugs before Gate 1 rerun~~ - **All eval fixes implemented:**
    - ✅ P0: Fail-fast on ImportError (no placeholder fallback)
    - ✅ P1: Agent B prompt excludes `message_A` for latent conditions
    - ✅ P1: Shuffle ablation implemented (rotate by 17 in seed_list index)
    - ✅ P1: KV cache generation (past_key_values maintains injected context)
    - ✅ P2: Wilson CI-aware gating (`CI_low >= threshold`)
    - ✅ P2: Paired comparisons (message_cache per seed)

### ❌ PLUMBING PROOF FAILED
11. ~~Run plumbing proof to verify injection works~~ - **FAIL: L1=0% vs P1=70%**
    - Root cause: Training objective is MSE reconstruction, not semantic communication
    - Codec learns geometric similarity, not usefulness as soft prefix

### 🔄 IN PROGRESS
12. **Fix training objective (distillation-based)**
    - Replace MSE loss with KL distillation from text-teacher to prefix-student
    - Train codec so Agent B with prefix behaves like Agent B with text

### 📋 Gate 1 Rerun Protocol

**Phase 1: Plumbing Proof (10 episodes each)**
1. Text present + injection off → expect ~P1 (68%)
2. Text absent + injection on → measure performance
3. Decision: If (2) ≈ (1) AND no-leak passes → injection works
   If (2) collapses to ~P0 → injection not working → fix before proceeding

**Phase 2: Clean Gate 1 Run (n=100)**
1. Verify all validity checks pass
2. Run k=4, k=8 with all ablations (null, random, shuffle)
3. Gate passes only if `CI_low >= 0.30` AND ablations pass

**Phase 3: Gate 2 (only after Phase 2)**
1. Run k=16, n=100
2. Gate passes if `CI_low >= 0.34`

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
