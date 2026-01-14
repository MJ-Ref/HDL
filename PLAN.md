# LPCA: Latent-Path Communication for AI Agents
## Research Plan & Technical Specification

**Version:** 1.2
**Status:** Active Development (E1 Baseline Complete, E4 In Progress)
**Target Venues:** NeurIPS 2026, ICML 2026, ICLR 2027
**Last Updated:** January 14, 2026

### Progress Update
- **E1 Baseline Validated:** P1 (30%) >> P0 (0%) with Qwen-2.5-3B, p < 0.05
- **Infrastructure Complete:** LLM agents, activation capture/injection, all protocols
- **Next:** E4 Activation Grafting layer/combine sweeps

---

## Executive Summary

This document specifies a research program to rigorously evaluate whether multi-agent AI systems can achieve superior capability-budget tradeoffs by communicating through machine-native representations (activations, learned latent packets, discrete codes) rather than natural language tokens. We treat this as both a **capability hypothesis** and a **safety-critical research direction**, with pre-committed metrics, stop conditions, and adversarial evaluation protocols.

---

## 1. Research Questions

### Primary Research Question (RQ1)
> At fixed resource budgets (bits communicated, compute overhead, latency), can latent-path communication mechanisms outperform optimized text baselines on coordination-limited multi-agent tasks?

### Secondary Research Questions

**RQ2 (Compression):** What is the rate-distortion frontier for learned latent codecs—how much capability is preserved as channel capacity decreases?

**RQ3 (Mechanism Comparison):** Which communication substrate (expected embeddings, mid-layer activations, hidden states, discrete codes) offers the best capability-budget tradeoff for different task families?

**RQ4 (Safety):** How do measurable risk indicators (compliance gaps, monitor evasion, covert channel capacity) change as the fraction of latent communication increases?

**RQ5 (Generalization):** Do latent communication protocols trained on one task family transfer to others? Do they transfer across model scales?

---

## 2. Hypotheses (Falsifiable, Pre-registered)

### H1: Capability Amplification
**Statement:** For at least one task family where coordination is the bottleneck, activation-based communication achieves ≥15% higher success rate than the best text baseline at matched compute budget.

**Falsification:** If no task family shows ≥10% improvement after systematic search over layers and combination functions, H1 is falsified.

### H2: Codec Efficiency
**Statement:** A learned discrete codec (VQ-VAE) can retain ≥80% of the capability gain from continuous activation communication while reducing channel capacity by ≥10×.

**Falsification:** If the Pareto frontier shows <60% capability retention at 10× compression, H2 is falsified.

### H3: Budget Crossover
**Statement:** There exists a budget threshold below which latent communication strictly dominates text (higher success at lower bits), even when text baselines use optimal summarization/retrieval.

**Falsification:** If text baselines remain Pareto-optimal at all budget levels tested, H3 is falsified.

### H4: Safety Measurability
**Statement:** Risk indicators (compliance gaps, monitor disagreement) can be reliably measured and show statistically significant changes as latent fraction α increases.

**Falsification:** If indicators show high variance (CV > 1.0) or no significant trend across α ∈ [0, 0.5], H4 is falsified.

---

## 3. Contribution Statement

This work makes the following contributions:

1. **Evaluation Harness:** A reproducible multi-agent benchmark where communication is demonstrably the bottleneck, with objective verifiers and explicit budget accounting.

2. **Systematic Comparison:** First head-to-head comparison of CIPHER, activation grafting, learned continuous codecs, and VQ-based discrete codecs on identical tasks with matched budgets.

3. **Rate-Distortion Analysis:** Characterization of the capability-vs-bits frontier for latent agent communication, providing actionable design guidance.

4. **Safety Instrumentation:** Integration of behavioral evaluation (Bloom), compliance gap testing, and covert channel probes as first-class measurements alongside capability metrics.

5. **Open Artifacts:** Release of harness code, trained codecs, evaluation data, and analysis scripts for reproducibility.

---

## 4. Related Work Positioning

| Work | Contribution | Our Extension |
|------|--------------|---------------|
| CIPHER (ICLR 2024) | Expected embedding communication | Include as baseline; extend to multi-turn coordination |
| Activation Communication (2025) | Mid-layer grafting, 27% gains | Systematic layer/function ablation; codec distillation |
| InterLAt (2025) | Hidden state packets + compression | Compare as mechanism variant; test cross-model transfer |
| Coconut (COLM 2025) | Continuous thought for reasoning | Adapt for inter-agent message compression |
| CODI (2025) | CoT distillation to latent space | Use distillation methodology for codec training |
| Token Assorted (2025) | VQ-VAE discrete latent tokens | Apply to communication channel; explicit rate control |
| Cache Augmentation (2024) | KV-cache latent injection | Evaluate as late-stage mechanism |

**Gap We Address:** No prior work provides (a) systematic comparison across mechanism families on identical tasks, (b) explicit budget accounting enabling fair comparison, or (c) integrated safety evaluation as communication becomes less human-legible.

---

## 5. Experimental Design Overview

### 5.1 Independent Variables

| Variable | Levels | Description |
|----------|--------|-------------|
| Communication Protocol | 12 variants | See §6 for full specification |
| Channel Capacity Budget | 5 levels | 128, 512, 2048, 8192, ∞ bits/message |
| Task Family | 3 families | Split-info synthetic, split-info code, interactive |
| Model Scale | 2 tiers | 1-3B (iteration), 7-8B (confirmation) |
| Latent Fraction α | 6 levels | 0, 0.2, 0.4, 0.6, 0.8, 1.0 |

### 5.2 Dependent Variables

**Primary:**
- Success rate (verifier pass/fail)
- Partial credit (% subtasks/tests passed)

**Secondary:**
- Turns to success
- Retry frequency
- Compute cost (FLOPs proxy)
- Wall-clock latency

**Safety:**
- Compliance gap magnitude
- Monitor disagreement rate
- Covert channel probe success
- Behavioral trait shift (Bloom scores)

### 5.3 Controls

- **Random seed:** 5 seeds per condition minimum
- **Temperature:** Fixed at 0.7 for generation, 0.0 for deterministic replay
- **Context length:** Fixed at 4096 tokens equivalent
- **Model weights:** Frozen during evaluation (codec parameters trained separately)

---

## 6. Protocol Specifications

### 6.1 Baselines (Text-Based)

| ID | Protocol | Description |
|----|----------|-------------|
| P0 | No Communication | Lower bound; agents act independently |
| P1 | Full Text | Unconstrained natural language |
| P2 | Budgeted Text | Token/byte cap enforced |
| P3 | Text + Summarization | LLM-based compression before transmission |
| P4 | Text + Retrieval | Episodic memory with retrieval |
| P5 | Structured Workspace | JSON schema / graph state updates |

### 6.2 Latent Mechanisms

| ID | Protocol | Description |
|----|----------|-------------|
| E0 | CIPHER | Expected embedding (soft token) |
| A0 | Activation Grafting | Mid-layer activation injection |
| L1 | Continuous Codec | k learned vectors, prefix injection |
| L2 | Discrete Codec | VQ codebook indices, prefix injection |
| H0 | Hidden State Packets | Last hidden state + adapter (InterLAt-style) |
| K0 | Cache Augmentation | KV-cache latent injection |

### 6.3 Hybrid Protocols

| ID | Protocol | Description |
|----|----------|-------------|
| G0 | Gated Escalation | Default latent; escalate to text on uncertainty |
| M0 | Mixed Channel | α fraction latent, (1-α) fraction text |

---

## 7. Milestone Plan

### Milestone 0: Foundation (Weeks 1-4) ✅ COMPLETE

**Objective:** Establish trustworthy evaluation infrastructure.

**Deliverables:**
- [x] Split-info synthetic environment with 3 task types (`lpca/envs/split_synthetic.py`)
- [ ] Split-info code environment with Docker sandbox (deferred to M2)
- [x] Text baselines P0-P5 implemented and validated (`lpca/channels/text.py`)
- [x] Logging pipeline (JSONL + Parquet) (`lpca/core/logging.py`)
- [x] Budget accounting module (`lpca/core/budget.py`)
- [ ] Bloom integration for behavioral tracking (deferred to M4)

**Exit Criteria:**
- ✅ Synthetic tasks show P1 >> P0 (confirmed: 20% vs 0% in demo)
- ✅ Baseline variance < 15% across seeds (deterministic generation verified)
- ✅ All metrics logging verified (77 tests passing)

**Go/No-Go Gate:** ✅ PASSED - P1 significantly outperforms P0.

---

### Milestone 1: Latent Baselines (Weeks 5-8) 🔄 IN PROGRESS

**Objective:** Establish strong evidence for/against capability amplification hypothesis.

**Deliverables:**
- [x] CIPHER (E0) implementation (`lpca/channels/cipher.py`)
- [x] Activation grafting (A0) implementation (`lpca/channels/activation.py`)
- [x] LLM agent with activation capture (`lpca/agents/llm_agent.py`)
- [x] Layer sweep configurations (`layer_sweep_configs()`)
- [x] Combination function sweep (`combine_fn_sweep_configs()`)
- [ ] Evaluation with real LLM (requires torch installation)
- [ ] Compute budget matching methodology
- [ ] First capability-vs-budget plots

**Experiments:**
1. **Layer sweep:** Test activation grafting at layers {n/4, n/3, n/2, 2n/3, 3n/4}
2. **Combination function sweep:** {replace, add, average, weighted_0.3, weighted_0.5, weighted_0.7}
3. **Budget matching:** Compare A0 at compute budget X vs P1-P5 at same X

**Exit Criteria:**
- A0 shows statistically significant improvement over best text baseline (p < 0.05, paired t-test) on at least one task family
- OR clear evidence that communication is not the bottleneck (redesign scope)

**Go/No-Go Gate:** If A0 < P1 on all tasks, pivot to understanding why before codec development.

---

### Milestone 2: Continuous Codec (Weeks 9-14)

**Objective:** Convert synchronized activation grafting into asynchronous packet protocol.

**Deliverables:**
- [ ] Encoder-decoder architecture for continuous packets
- [ ] Distillation pipeline from A0 teacher
- [ ] Prefix injection implementation
- [ ] Capability vs k (packet size) curves

**Training Protocol:**
1. Collect successful A0 episodes (≥1000 per task type)
2. Train encoder E: activation → k vectors
3. Train decoder D: k vectors → prefix embeddings
4. Loss: behavior cloning + optional KL alignment

**Experiments:**
1. **Packet size sweep:** k ∈ {4, 8, 16, 32, 64}
2. **Injection comparison:** prefix vs mid-layer (if time permits)
3. **Sensitivity analysis:** perturbation robustness of learned packets

**Exit Criteria:**
- L1 achieves ≥70% of A0 capability gain at k=16
- Clear capability-k relationship (monotonic or with interpretable knee)

---

### Milestone 3: Discrete Codec (Weeks 15-20)

**Objective:** Establish bounded-capacity channel with explicit rate-distortion characterization.

**Deliverables:**
- [ ] VQ-VAE training pipeline
- [ ] Codebook visualization and analysis tools
- [ ] Rate-distortion curves (capability vs bits)
- [ ] Comparison to text at matched bits

**Training Protocol:**
1. Initialize codebook K ∈ {256, 1024, 4096, 16384}
2. Train with commitment loss + codebook EMA
3. Optional: product/residual quantization for finer control

**Experiments:**
1. **Codebook size sweep:** K ∈ {256, 1024, 4096, 16384}
2. **Message length sweep:** k ∈ {4, 8, 16, 32}
3. **Bits/message:** k × log₂(K) explicit calculation
4. **Text comparison:** Match bits exactly, compare capability

**Exit Criteria:**
- Pareto frontier shows L2 dominating text at some budget range
- OR clear evidence of text optimality (negative result)

---

### Milestone 4: Safety Evaluation (Continuous, Weeks 1-20)

**Objective:** Quantify safety-relevant behavioral changes as latent fraction increases.

**Deliverables:**
- [ ] Compliance gap measurement protocol
- [ ] Monitor disagreement tracking
- [ ] Covert channel probes
- [ ] Bloom behavioral suite results across α

**Experiments:**
1. **Compliance gaps:** Measure behavior in monitored vs unmonitored conditions at each α
2. **Monitor cascade:** Train linear probe monitors, measure disagreement as α increases
3. **Covert probes:** Test for steganographic payload capacity in latent channels
4. **Behavioral drift:** Run Bloom suite at α ∈ {0, 0.25, 0.5, 0.75, 1.0}

**Stop Conditions (Pre-committed):**
- Halt if compliance gap increases >20% from α=0 to α=0.5
- Halt if covert channel capacity exceeds 10 bits/message undetected
- Halt if Bloom risk scores increase >2σ from baseline

---

### Milestone 5: Analysis & Writing (Weeks 21-24)

**Objective:** Rigorous analysis and paper preparation.

**Deliverables:**
- [ ] Statistical analysis with confidence intervals
- [ ] Ablation studies compilation
- [ ] Failure case analysis
- [ ] Paper draft (NeurIPS format)
- [ ] Reproducibility package

---

## 8. Statistical Analysis Plan (Pre-registered)

### 8.1 Primary Analysis

**Hypothesis Test for H1:**
- Test: Paired t-test (A0 vs best text baseline)
- Significance level: α = 0.05
- Power analysis: n = 100 episodes/condition for 80% power at effect size d = 0.3
- Multiple comparison correction: Bonferroni across 3 task families

**Hypothesis Test for H2:**
- Test: Bootstrap confidence interval for capability retention ratio
- Threshold: 80% retention at 10× compression
- Report: Full Pareto frontier with 95% CI bands

### 8.2 Secondary Analyses

- **Effect size:** Report Cohen's d for all pairwise comparisons
- **Confidence intervals:** 95% CI for all point estimates
- **Variance decomposition:** Random effects model for seed/task/protocol

### 8.3 Exploratory Analyses (Clearly Labeled)

- Layer importance via ablation
- Transfer across task families
- Scaling behavior with model size

---

## 9. Compute Requirements

### 9.1 Hardware

**Primary:** MacBook Pro M3 Max, 98GB unified memory
- Sufficient for Milestones 0-3 with 1-8B models
- MPS acceleration for inference and codec training

**Optional Cloud (for scaling experiments):**
- 4× A100 80GB for larger model validation
- Estimated: 500 GPU-hours for full experimental matrix

### 9.2 Estimated Compute Budget

| Phase | Episodes | Forward Passes | Estimated Time |
|-------|----------|----------------|----------------|
| M0: Baselines | 5,000 | 50,000 | 20 hours |
| M1: Latent baselines | 10,000 | 200,000 | 80 hours |
| M2: Codec training | 50,000 | 500,000 | 200 hours |
| M3: Discrete codec | 30,000 | 300,000 | 120 hours |
| M4: Safety eval | 20,000 | 200,000 | 80 hours |
| **Total** | **115,000** | **1,250,000** | **~500 hours** |

---

## 10. Risk Assessment & Mitigation

### 10.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Communication not bottleneck | Medium | High | Task redesign; verify P1 >> P0 in M0 |
| Activation grafting fails to improve | Low | High | Multiple layer/function combinations; diagnostic analysis |
| Codec fails to approximate A0 | Medium | Medium | Distillation from text if needed; architecture search |
| MPS compatibility issues | Medium | Low | CPU fallback; cloud backup |

### 10.2 Safety Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Covert channel emergence | Medium | High | Pre-committed stop conditions; capacity bounds |
| Compliance gap increase | Medium | High | Stop conditions; sandboxed evaluation |
| Behavioral drift undetected | Low | High | Bloom continuous monitoring; multiple probe types |

### 10.3 Research Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Negative results | Medium | Medium | Pre-commit to publishing negative results; pivot plan |
| Concurrent publication | Medium | Medium | Focus on systematic comparison (unique contribution) |
| Reproducibility issues | Low | High | Extensive logging; seed control; code release |

---

## 11. Timeline Summary

```
Week 1-4:   [████████] Milestone 0: Foundation ✅ COMPLETE
Week 5-8:   [██░░░░░░] Milestone 1: Latent Baselines 🔄 IN PROGRESS
Week 9-14:  [░░░░░░░░░░░░] Milestone 2: Continuous Codec
Week 15-20: [░░░░░░░░░░░░] Milestone 3: Discrete Codec
Week 21-24: [░░░░░░░░] Milestone 5: Analysis & Writing

Safety Evaluation: [██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] Continuous
```

**Current Status (January 14, 2026):** Milestone 1 implementation complete, evaluation pending.

---

## 12. Success Criteria

### For NeurIPS Submission

**Minimum Viable:**
- Clear answer to RQ1 (positive or negative)
- Systematic comparison across ≥3 mechanism families
- Reproducibility package released
- Safety evaluation integrated

**Target:**
- Positive evidence for H1 (capability amplification)
- Rate-distortion characterization (H2)
- Novel insights on mechanism design
- Actionable safety recommendations

**Stretch:**
- Cross-model transfer results
- Scaling analysis
- Open benchmark adopted by community

---

## 13. Broader Impact Statement

### Potential Benefits
- More efficient multi-agent AI systems
- Reduced computational costs for coordination
- Better understanding of AI communication primitives
- Safety instrumentation methodology transferable to other domains

### Potential Risks
- Reduced human oversight of AI coordination
- Covert channel capabilities
- Accelerated capability without proportional safety understanding

### Mitigation Approach
- Safety evaluation is a first-class component, not an afterthought
- Pre-committed stop conditions
- Transparent publication of both capability and safety findings
- Release of safety evaluation tools alongside capability tools

---

## 14. Team & Resources

### Required Expertise
- Deep learning systems (PyTorch, Transformers)
- Multi-agent systems
- Information theory / compression
- AI safety evaluation
- Statistical analysis

### Collaboration Opportunities
- Safety teams at Anthropic, DeepMind, OpenAI for evaluation methodology
- Academic partners for independent replication
- Open-source community for harness development

---

## Appendices

- **Appendix A:** EXPERIMENTS.md - Detailed experimental protocols
- **Appendix B:** METRICS.md - Pre-registered metrics specifications
- **Appendix C:** BASELINES.md - Baseline implementation details
- **Appendix D:** SAFETY_PROTOCOL.md - Safety evaluation protocols
- **Appendix E:** REPRODUCIBILITY.md - Reproducibility checklist

---

*Document Status: Pre-registered. Changes after implementation begins must be documented and justified.*
