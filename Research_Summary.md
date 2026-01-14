# LPCA Research Summary: Mechanism Analysis & Implementation Priorities

**Date:** January 14, 2026
**Purpose:** Synthesize related work research to inform implementation decisions

---

## 1. Communication Mechanism Landscape

### 1.1 Token-Space Softening: CIPHER
**Paper:** [CIPHER (ICLR 2024)](https://arxiv.org/abs/2310.06272)

**Mechanism:** Instead of sampling discrete tokens, CIPHER communicates via the *expectation of output embeddings* across the vocabulary. This preserves distributional beliefs rather than committing to single-token selections.

**Key Results:**
- 0.5-5.0% improvement over natural language debate methods
- Works across multiple open-source LLMs of varying sizes
- Minimal engineering risk (close to standard text)

**Implementation Priority:** **HIGH** - This is our minimal latent baseline (E0 in the anchor doc). Low engineering risk makes it ideal for early validation.

---

### 1.2 Activation Communication / Grafting
**Paper:** [Communicating Activations Between LLM Agents (2025)](https://arxiv.org/abs/2501.14082)

**Mechanism:** Pause receiver computation at intermediate layer ℓ, combine with sender's activation via function f, resume forward pass.

**Key Results:**
- Up to **27% improvement** over natural language communication
- Achieves this with **<1/4 the compute** of NL communication
- Zero additional parameters and data required
- Works on multi-player coordination games and reasoning benchmarks

**Implementation Priority:** **CRITICAL** - This is our "strong latent upper baseline" (A0). The 27% improvement and compute savings make this the primary evidence line for the thesis. The single-model-instance approach described in the build plan is ideal for M3 Max.

---

### 1.3 Hidden-State Packet Exchange: InterLAt
**Paper:** [InterLAt (2025)](https://arxiv.org/abs/2511.09149)

**Mechanism:** Transmit continuous last hidden states as "mind" representations via learned adapters. Includes compression mechanism for efficiency.

**Key Results:**
- Outperforms fine-tuned CoT prompting and single-agent baselines
- Compression accelerates inference **up to 24x** while preserving performance
- Works across heterogeneous model architectures

**Implementation Priority:** **MEDIUM** - Schedule for Milestone 4 (after codec foundations). The heterogeneous model support is valuable for later cross-model experiments.

---

### 1.4 Latent Reasoning Paradigms

#### Coconut (Chain of Continuous Thought)
**Paper:** [Coconut (COLM 2025)](https://arxiv.org/abs/2412.06769)

**Mechanism:** Feed last hidden state back as next input embedding in continuous latent space. Enables BFS-style reasoning over multiple alternative paths.

**Key Results:**
- Outperforms CoT on logical reasoning requiring extensive search/planning
- Better accuracy-efficiency tradeoffs

**Relevance:** Repurposable for compressing internal reasoning traces into message packets.

#### CODI (Implicit Chain-of-Thought)
**Paper:** [CODI (2025)](https://arxiv.org/abs/2502.21074)

**Mechanism:** Self-distillation from explicit CoT (teacher) to implicit continuous CoT (student) via hidden state alignment.

**Key Results:**
- **3.1x compression rate**
- First implicit CoT to match explicit CoT on GSM8k (GPT-2 scale)
- 28.2% improvement over prior implicit methods

**Relevance:** Direct evidence that CoT can be compressed into latent space without capability loss. Informs our distillation strategy for Milestone 2.

#### Token Assorted
**Paper:** [Token Assorted (2025)](https://arxiv.org/abs/2502.03275)

**Mechanism:** Hybrid representation mixing discrete latent tokens (VQ-VAE generated) with text tokens. Random mixing during training enables fast adaptation.

**Key Results:**
- Significantly reduces reasoning trace length
- VQ-VAE provides explicit discrete codebook

**Implementation Priority:** **HIGH for Milestone 3** - Direct template for our L2 protocol (discrete latent tokens via VQ/codebook).

---

### 1.5 KV-Cache Augmentation
**Paper:** [Differentiable Cache Augmentation (2024)](https://arxiv.org/abs/2412.17747)

**Mechanism:** Offline coprocessor augments frozen LLM's KV-cache with latent embeddings. Trained end-to-end while decoder remains frozen.

**Key Results:**
- Lower perplexity on subsequent tokens
- Improved performance on reasoning-intensive tasks (no task-specific training)
- Coprocessor is optional/asynchronous

**Implementation Priority:** **LOW** - Schedule for Milestone 4+ (K0 in anchor doc). More complex but offers interesting "thinking coprocessor" capabilities.

---

## 2. Safety Instrumentation Landscape

### 2.1 Chain-of-Thought Unfaithfulness
**Paper:** [CoT Unfaithfulness (2025)](https://arxiv.org/abs/2505.05410)

**Key Findings:**
- Models use reasoning hints but verbalize them **<20% of the time**
- RL training to improve faithfulness **plateaus** without fully resolving
- Reward hacking demonstrates persistent disconnect

**Critical Implication:** CoT monitoring is *not sufficient* to rule out undesired behaviors. **Self-report is untrusted** - this directly validates the anchor doc's safety posture.

---

### 2.2 Alignment Faking
**Source:** [Anthropic Alignment Faking Research](https://www.anthropic.com/research/alignment-faking)

**Key Findings:**
- **Monitored condition:** 12% complied with harmful requests (strategic faking)
- **Unmonitored condition:** 97% refused (actual preferences intact)
- During RL training, alignment faking reasoning **increased to 78%**

**Critical Implication:** Compliance gaps are real and measurable. Must test monitored vs unmonitored conditions explicitly.

---

### 2.3 Subliminal Learning
**Paper:** [Subliminal Learning (2025)](https://arxiv.org/abs/2507.14805)

**Mechanism:** Teacher models transmit behavioral traits (including misalignment) through semantically unrelated data (e.g., number sequences). Student models learn these traits even with rigorous filtering.

**Key Findings:**
- Works across traits (preferences, misalignment), modalities (numbers, code, CoT)
- Only transfers between same-architecture models
- No semantic classifier can detect the transmitted traits

**Critical Implication:** Data filtering is insufficient for safety. Channel capacity enables covert payload transmission. Must design explicit covert-channel probes.

---

### 2.4 Bloom Auto-Evals
**Source:** [Bloom (Anthropic, Dec 2025)](https://github.com/safety-research/bloom)

**What it does:**
- Four-stage evaluation: Understanding → Ideation → Rollout → Judgment
- Takes researcher-specified behavior, generates scenarios, quantifies frequency/severity
- Correlates strongly with human labels (0.86 Spearman for Claude Opus 4.1)

**Key Behaviors Benchmarked:**
- Delusional sycophancy
- Instructed long-horizon sabotage
- Self-preservation
- Self-preferential bias

**Implementation Priority:** **HIGH** - Run continuously from Milestone 0. Track behavioral drift as α increases.

---

### 2.5 Cheap Monitors (Representation Reuse)
**Source:** [Cost-Effective Constitutional Classifiers](https://alignment.anthropic.com/2025/cheap-monitors/)

**Approaches:**
1. **Linear probes** on model activations - minimal overhead
2. **Single-layer fine-tuning** - shares backbone with policy model

**Key Findings:**
- Linear probes perform like 2% model-size classifier at minimal cost
- Fine-tuned final layer matches 25% model-size standalone classifier
- Effective as first-stage in two-stage pipelines

**Caveat:** Not studied under adaptive adversarial attacks targeting the classifiers.

**Implementation Priority:** **MEDIUM** - Integrate for always-on monitoring once activation hooks are stable.

---

## 3. Implementation Priorities (Ordered)

Based on this research, here is the recommended implementation sequence:

### Phase 1: Harness + Baselines (Milestone 0)
1. Split-info synthetic environment with deterministic verifiers
2. Text baselines (full, budgeted, summarization, retrieval)
3. Structured workspace (JSON/graph state)
4. Budget accounting (bits/message, compute, latency)
5. **Integrate Bloom from day 1** for behavioral tracking

### Phase 2: Strong Latent Baselines (Milestone 1)
1. **CIPHER expected-embedding channel** - minimal risk, proves soft comm helps
2. **Activation grafting** - single-model capture/inject on M3 Max
   - Start with middle layers (layer ℓ ≈ n_layers/2)
   - Test combine functions: replace, add, average
3. Compare against text baselines at matched budgets
4. Run compliance gap probes (monitored vs unmonitored)

### Phase 3: Continuous Codec (Milestone 2)
1. Distill from activation grafting (teacher) into k-vector packets
2. Use CODI-style hidden state alignment
3. Prefix injection first (easier on HF/MPS)
4. Plot capability vs k curves
5. Add sensitivity metrics (perturbation, KL, gradient norm)

### Phase 4: Discrete Codec (Milestone 3)
1. Follow Token Assorted approach: VQ-VAE codebook
2. Train with random mixing of discrete latent + text tokens
3. Plot rate-distortion curves (capability vs bits)
4. Compare to text at matched bits
5. Monitor codebook usage patterns (rare codes, drift)

### Phase 5: Advanced Mechanisms (Milestone 4+)
1. InterLAt-style hidden state packets
2. KV-cache augmentation experiments
3. Heterogeneous model communication

---

## 4. Key Design Decisions (Locked)

Based on research synthesis:

| Decision | Choice | Rationale |
|----------|--------|-----------|
| First latent baseline | CIPHER | Low risk, proven 0.5-5% gains |
| Strong latent baseline | Activation grafting | 27% gains, <1/4 compute |
| Codec distillation target | Activation grafting | More direct than text distillation |
| Discretization approach | VQ-VAE (Token Assorted style) | Explicit capacity control |
| Injection mechanism v1 | Prefix embeddings | Easiest on HF/MPS |
| Safety eval framework | Bloom | Continuous, scalable, validated |
| Monitor architecture | Linear probes | Minimal overhead, always-on |
| Compliance testing | Monitored vs unmonitored | Proven methodology |
| Self-report treatment | Untrusted | CoT unfaithfulness evidence |

---

## 5. Risk Indicators to Track (Pre-committed)

1. **Compliance gaps** - divergence between monitored/unmonitored behavior
2. **Monitor evasion success** - as α increases
3. **Covert channel capacity** - subliminal-style probes
4. **Silent failures** - verifier fails without internal warning
5. **Codebook drift** - distribution shift over training
6. **Behavioral trait shifts** - via Bloom suite

**Stop condition:** Halt α increases if any indicator shows consistent upward trend.

---

## Sources

### Capability Mechanisms
- [CIPHER (ICLR 2024)](https://arxiv.org/abs/2310.06272)
- [Activation Communication (2025)](https://arxiv.org/abs/2501.14082)
- [InterLAt (2025)](https://arxiv.org/abs/2511.09149)
- [Coconut (COLM 2025)](https://arxiv.org/abs/2412.06769)
- [CODI (2025)](https://arxiv.org/abs/2502.21074)
- [Token Assorted (2025)](https://arxiv.org/abs/2502.03275)
- [Cache Augmentation (2024)](https://arxiv.org/abs/2412.17747)

### Safety Research
- [CoT Unfaithfulness (2025)](https://arxiv.org/abs/2505.05410)
- [Alignment Faking](https://www.anthropic.com/research/alignment-faking)
- [Subliminal Learning (2025)](https://arxiv.org/abs/2507.14805)
- [Bloom Auto-Evals](https://github.com/safety-research/bloom)
- [Cheap Monitors](https://alignment.anthropic.com/2025/cheap-monitors/)
