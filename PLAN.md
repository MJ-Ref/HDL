# LPCA: Latent-Path Communication for AI Agents
## Research Plan & Technical Specification

**Version:** 2.0
**Status:** Semantic signal detected - Shuffle craters 10 points below Normal (16% vs 26%)
**Target Venues:** NeurIPS 2026, ICML 2026, ICLR 2027
**Last Updated:** January 20, 2026

### Progress Update
- **M1 COMPLETE:** Raw latent communication (E0, A0) does NOT work without training
- **Key Finding (n=50, tightened task):** P1 (68%) >> P0 (20%), E0=13%, A0=20% (no improvement)
- **Task Validated:** P0 at 20% confirms task is communication-limited (target: 15-25%)
- **M2-LOCAL-PROTO:** PASSED - codec pipeline validated
- **E5 Safety:** All metrics passed (compliance gap 17.5%, covert channel 8 bits)
- **E2-min COMPLETE:** P5 (structured) dominates P2 at all budgets; P5_16B=56.7% at 43 bits
- **Prefix Collapse:** ✓ **FIXED** via identity decoder + prefix calibration
- **Semantic Signal:** ✓ **DETECTED** - Normal (26%) > Shuffle (16%) by 10 points!
- **M2-SCALE Gate 1:** ⚠️ **FAIL** - Normal=26% below threshold (30%), and Normal=Null
- **Next:** Need prefix to HELP (Normal > Null), not just "not hurt"

### Latest Findings (January 20, 2026)

**Prefix Calibration Results (scale_gate=0.5, 10 epochs):**
| Condition | Success Rate | Analysis |
|-----------|-------------|----------|
| Normal    | 26.0%       | Baseline with correct prefix |
| Null      | 26.0%       | No prefix = same as Normal |
| Random    | 28.0%       | Random prefix = slightly better |
| Shuffle   | **16.0%**   | Wrong prefix HURTS (-10 points!) |

**Key Insight:** The codec now encodes semantic information:
- Normal > Shuffle by 10 percentage points (semantic signal!)
- BUT: Normal = Null (correct prefix doesn't HELP, just doesn't hurt)
- The model learned "wrong info hurts" but not "right info helps"

**What's Working:**
1. Prefix collapse fixed - diversity maintained (cosim ~0.72)
2. Semantic encoding working - shuffle craters below P0
3. Prefix calibration (LayerNorm + scale_gate) prevents OOD embeddings

**What Needs Fixing:**
1. Normal should be > Null (correct prefix should actively help)
2. The contrastive loss forces correct > shuffle, but doesn't push correct > null
3. Need additional loss term: `L_help = max(0, NLL_null - NLL_correct + margin)`

### Previous Findings (January 16, 2026)

**Multi-token KL Distillation Results:**
| Condition | Single-Prefix | k=4 Prefix |
|-----------|--------------|------------|
| Normal    | 20%          | **28%**    |
| Null      | 28%          | 20%        |
| Random    | 20%          | 26%        |
| Shuffle   | 20%          | **28%**    |

**Key Insight:** The k-prefix architecture improved Normal from 20% → 28%, but **Shuffle didn't crater**.
This means the codec learned a general "priming" effect that helps regardless of message content,
NOT semantic encoding of the message. The prefix provides value, but not because it encodes
the actual information from Agent A's observation.

**Root Cause Analysis:**
1. KL distillation matches logit distributions, not semantic content
2. Multi-token KL over "ANSWER: {"x1": 0, ...}" is **dominated by fixed-format tokens** (braces, keys, punctuation)
3. The codec wins loss by improving formatting/generic solving bias, not by learning which digits to output
4. The student model learns a "warm start" priming effect independent of message content
5. Shuffle should crater (<20%) if semantic content mattered, but it matches Normal (28%)

**Why Normal ≈ Shuffle and Random > Null:**
- "Having some injected prefix" helps → priming effect
- But the prefix doesn't carry message-specific information
- The codec learned a useful content-agnostic prefix, not semantic encoding

**Solution: Value-Weighted CE + Anti-Shuffle Contrastive**

Instead of KL over all tokens, use:
```
L = CE_pos + α·max(0, m - (CE_neg - CE_pos))
```

Where:
- **CE_pos**: Cross-entropy on teacher answer tokens with **correct** message prefix
- **CE_neg**: Cross-entropy on teacher answer tokens with **shuffled** message prefix
- **Value-weighting**: Upweight the 4 value tokens (actual x1-x4 values) vs punctuation/keys
- **Contrastive margin**: Enforce CE_neg >> CE_pos (shuffled prefix makes correct answer harder)

This directly optimizes for "shuffle should crater" - the exact signal we need.

**Quick Diagnostics Before A100 Time:**
1. **Prefix collapse check**: Log mean cosine-similarity of prefix_embeddings across different messages. If ~1, learning a constant "magic prompt"
2. **Deterministic semantic check**: Score teacher answer sequence logprob under (a) correct prefix vs (b) shuffled prefix vs (c) null. Want clear gap: NLL(correct) << NLL(shuffle)

**When to Increase k:**
After the above makes shuffle drop hard and null/random land near P0, THEN sweep k=8,16. Increasing k before the objective is semantic just gives "priming" more bandwidth.

### Prefix Collapse Problem (January 19, 2026)

**Implemented Value-Weighted CE + Contrastive + Diversity Loss:**
Training now uses:
- Value-weighted CE: 10x weight on actual value tokens (digits) vs formatting
- Contrastive margin loss: Penalize if shuffle prefix doesn't make answer harder
- Diversity loss: Penalize high cosine similarity between prefixes within batch

**Diagnostic Results:**
| Metric | During Training | During Inference (eval mode) |
|--------|----------------|------------------------------|
| Mean cosine sim | 0.80-0.86 | 0.9999-1.0000 |
| Shuffle gap | Variable | 0.0020 (nearly zero) |

**Root Cause of Collapse:**
1. Dropout in encoder provides diversity during training (cosim ~0.80)
2. When dropout is OFF (inference), outputs collapse to constant "magic prompt"
3. The encoder architecture learns to output ~same vector for all inputs
4. Diversity loss gradient flows back, but CE loss strongly favors "universal good prefix"
5. Strong diversity weight (10x) hurts task performance (null prefix becomes BETTER than learned prefix)

**Why This Happens:**
The encoder is a simple MLP: `pooled_embedding → k latent vectors`. Without explicit constraints, the optimal solution is to ignore the input and output a fixed "magic prompt" that helps with all answers. Diversity loss fights this, but:
- Weak diversity: prefixes collapse
- Strong diversity: prefixes diverse but useless for task

**Potential Fixes to Explore:**
1. **Reconstruction loss**: Force encoder to encode message info (e.g., predict message from latent)
2. **VQ-VAE bottleneck**: Discrete codebook forces distinct codes
3. **Skip connections**: Let input bypass bottleneck to force differentiation
4. **Contrastive encoder pre-training**: Train encoder on message discrimination first
5. **Information-theoretic regularization**: Maximize mutual information I(message; latent)

### Prefix Collapse FIX (January 19, 2026)

**Solution Implemented - Three Key Changes:**

1. **Last-token pooling** instead of mean pooling
   - Mean pooling produced very similar embeddings (cosim=0.98) for different messages
   - Last-token pooling for decoder-only models improves input diversity (cosim=0.91, range 0.71-0.98)

2. **Info preservation loss**
   - Forces encoder to preserve pairwise input similarities in latent space
   - `L_info = MSE(latent_pairwise_cosim, input_pairwise_cosim)`
   - Prevents encoder from collapsing all inputs to same output

3. **Identity decoder** (latent = prefix)
   - Removed decoder MLP which was learning to collapse diversity
   - Encoder now directly produces prefix embeddings
   - No transformation layer to learn "ignore input variation"

**Results:**
| Checkpoint | Latent Cosim | Prefix Cosim | Shuffle Gap | Status |
|------------|--------------|--------------|-------------|--------|
| Before fixes | 0.9999 | 1.0000 | 0.0000 | Collapsed |
| After fixes | 0.9834 | 0.9834 | **0.4655** | Semantic signal! |

**Key Achievement:** Positive shuffle gap (0.4655) proves the codec encodes semantic information.
Correct prefix helps more than shuffled prefix - the exact signal needed for Gate 1.

**Remaining Issue:** Prefix NLL is high (3.13 vs 0.36 null). The prefixes aren't well-calibrated
as soft prompts yet. Need longer training to balance CE loss with info preservation.

**Next Steps:**
1. Train longer (10+ epochs) to calibrate prefix embeddings
2. Tune loss weights (reduce INFO_PRESERVE_WEIGHT after initial epochs)
3. Run full Gate 1 evaluation on Modal A100

### Critical Issues Identified (January 15, 2026)

**Issue 1: Placeholder fallback can fake results**
- `_run_codec_episode()` returns `random.random() < 0.3` (~30%) if env import fails
- This alone can explain the "30-42%" numbers even if nothing is working
- **Fix:** Fail hard on import error, never return placeholder results

**Issue 2: Latent vectors computed but never used**
- `prefix_embedding = codec.decode(latent)` is computed but never injected
- Agent B always receives `message_A` text regardless of ablation
- "Normal" runs test text communication, NOT the codec
- **Fix:** Either inject embeddings OR remove text from prompt in latent conditions

**Issue 3: Shuffle ablation missing**
- Required by spec ("should crater <P0") but not implemented
- Most diagnostic test for semantic dependence
- **Fix:** Implement shuffle before Gate 1 rerun

**Issue 4: Point-estimate gating is noisy**
- n=50 gives wide CIs; "pass/fail" on point estimate is unreliable
- **Fix:** Use CI-aware gating (lower bound of 95% CI must meet threshold)

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

### Milestone 1: Latent Baselines (Weeks 5-8) ✅ COMPLETE (NEGATIVE RESULT)

**Objective:** Establish strong evidence for/against capability amplification hypothesis.

**Deliverables:**
- [x] CIPHER (E0) implementation (`lpca/channels/cipher.py`)
- [x] Activation grafting (A0) implementation (`lpca/channels/activation.py`)
- [x] LLM agent with activation capture (`lpca/agents/llm_agent.py`)
- [x] Layer sweep configurations (`layer_sweep_configs()`)
- [x] Combination function sweep (`combine_fn_sweep_configs()`)
- [x] Evaluation with real LLM (Qwen-2.5-3B on MPS)
- [x] E3 CIPHER evaluation (0% success)
- [x] E4 Activation grafting layer sweep (0% at all layers)
- [x] E5 Safety evaluation (PASS)

**Results (Qwen-2.5-3B, Constraint Satisfaction, n=50, tightened task):**
| Protocol | Success | 95% CI | Interpretation |
|----------|---------|--------|----------------|
| P0 (no comm) | **20%** | [11.2%, 33.0%] | Baseline - in target range (was 52% before tightening) |
| **P1 (text)** | **68%** | **[54.2%, 79.2%]** | **Communication helps (+48pp)** |
| E0 (CIPHER) | 13% | [3.7%, 37.9%] | Expected embeddings don't help |
| A0 L18 | 20% | [7.0%, 45.2%] | Activation injection no better than P0 |

**Key Finding:** Raw latent representations do NOT transfer task-relevant information.
- Task now properly communication-limited (P0=20%, target was 15-25%)
- E0/A0 no better than P0 - receiver can't interpret without training
- P1 >> P0 with non-overlapping 95% CIs proves communication is the bottleneck

**Exit Criteria:**
- ✅ P1 >> P0 confirmed (68% vs 20%, non-overlapping CIs)
- ❌ A0/E0 do NOT show improvement over P0 baseline
- ✅ Clear evidence: raw latent communication requires TRAINING to be effective

**Go/No-Go Decision:**
- PROCEED to E2-min (budgeted text baselines) then M2-SCALE
- Rationale: Need text baselines at budgets to set targets for codec
- M2 will train encoder/decoder to make latent communication meaningful

---

### Milestone 2: Continuous Codec (Weeks 9-14)

**Objective:** Train encoder-decoder to make latent communication meaningful.

**Why M2 is Necessary:**
M1 showed that raw latent channels (E0, A0) don't work - the receiver can't interpret
untrained activations. M2 trains a codec to create semantically meaningful latent packets.

---

#### M2-LOCAL-PROTO: Local Prototyping (Can run on M3 Max)

**Purpose:** Validate pipeline, debug architecture, iterate quickly before cloud spend.

**Approach:**
- **Freeze base model** - no gradients through transformer
- Train only small encoder/decoder MLPs (~10M params)
- Use small dataset (100-500 successful P1 episodes)
- Single k value (k=16) for initial validation

**Local Feasibility:**
- Memory: ~6GB model (frozen) + ~100MB codec = fits comfortably
- Speed: ~10-30 min for one training run
- Can iterate 5-10x per day locally

**Exit Criteria:**
- [ ] Pipeline works end-to-end (encode → inject → decode)
- [ ] Loss decreases during training
- [ ] Qualitative: injected representations don't break generation

---

#### M2-SCALE: Full Training (Requires Cloud GPUs) ⚠️ Gate 1 INVALID

**Purpose:** Train high-quality codec with full sweeps once pipeline is validated.

**Gate 1 Results (January 15, 2026) - ⚠️ INVALID:**
| Config | Final Loss | Normal | Null Msg | Random Latent | Gate 1 |
|--------|-----------|--------|----------|---------------|--------|
| k=4 | 0.097 | 34.0% | 30.0% | 38.0% | ⚠️ INVALID |
| k=8 | 0.111 | 38.0% | 36.0% | 42.0% | ⚠️ INVALID |

**Why Invalid:** Results may reflect text-based communication (message_A always present)
or placeholder returns (~30%), NOT actual latent codec performance.

---

##### Validity Checklist (Required Per Run)

| Check | Description | Pass Criterion |
|-------|-------------|----------------|
| **Env Import** | Real env loaded, no placeholder | No "LPCA not available" in logs |
| **No Text Leak** | Agent B prompt has no `message_A` | Leak test assertion passes |
| **Injection Active** | `prefix_embedding` actually used | Plumbing proof shows effect |
| **Shuffle Craters** | Shuffle ablation < P0 | CI_high(shuffle) < CI_low(P0) |
| **Null ≈ P0** | Null ablation CI overlaps P0 CI | CI overlap test |
| **Random ≈ P0** | Random ablation CI overlaps P0 CI | CI overlap test |

**Invalidation Policy:** If ANY checklist item fails, results are labeled INVALID
and excluded from gate comparisons. No exceptions.

---

##### Critical Fixes Before Gate 1 Rerun

**Fix 1: Remove placeholder fallback (PRIORITY 0)**
```python
# OLD (dangerous):
except ImportError:
    print("Warning: LPCA not available, using placeholder")
    return random.random() < 0.3  # Can fake ~30% results!

# NEW (fail fast):
except ImportError as e:
    raise RuntimeError(f"LPCA env required for evaluation: {e}")
```

**Fix 2: Eliminate text leakage (PRIORITY 1)**
- Create `build_agent_b_prompt(obs_B, latent_info, protocol)` function
- For L1/latent: `latent_info` is description of injected embeddings, NOT `message_A`
- Add leak test: assert `message_A` not in `prompt_B` for latent conditions
- Log prompt hash + save 1-2 examples per condition

**Fix 3: Implement shuffle ablation (PRIORITY 1)**
```python
# Shuffle: use message from different episode (breaks semantic link)
# Rotate within the evaluated seed list (e.g., seeds 2000-2099)
if ablation == 'shuffle':
    seed_list = list(range(test_start, test_start + n_episodes))
    idx = seed_list.index(seed)
    shuffled_idx = (idx + 17) % len(seed_list)  # Rotate by 17 in index space
    shuffled_seed = seed_list[shuffled_idx]
    message_A = cached_messages[shuffled_seed]
```
Expected: CI_high(shuffle) < CI_low(P0) (proves semantic dependence)

**Fix 4: CI-aware gating (PRIORITY 2)**
```python
# Gate passes only if lower CI bound meets threshold
ci_low, ci_high = wilson_ci(successes, n_episodes)
gate_pass = ci_low >= threshold  # Not just mean >= threshold
```

**Fix 5: Paired comparisons (PRIORITY 2)**
- Cache `message_A` per env seed
- Reuse across normal/null/random/shuffle conditions
- Either use greedy decoding OR seed torch RNG per episode+condition

**Fix 6: KV cache for generation (PRIORITY 1)**
```python
# OLD (context lost - generation ignores injected prefix after first token):
outputs = model(current_ids)  # Bug: loses combined_embeds context

# NEW (maintain context via past_key_values):
outputs_B = model(inputs_embeds=combined_embeds, attention_mask=combined_mask, use_cache=True)
past_key_values = outputs_B.past_key_values
for _ in range(max_tokens - 1):
    outputs = model(input_ids=current_ids, past_key_values=past_key_values, use_cache=True)
    past_key_values = outputs.past_key_values
```

---

##### Gate 1 Rerun Protocol

**Phase 1: Plumbing Proof (before GPU spend) — ❌ FAILED (January 15, 2026)**

| Condition | Success Rate | Expected |
|-----------|-------------|----------|
| P1 Baseline (text present, no injection) | **70.0%** | ~68% ✓ |
| L1 Injection (text absent, injection on) | **0.0%** | Should match P1 |

**Root Cause:** Training objective is MSE reconstruction loss, which trains geometric
similarity instead of semantic usefulness. The codec is an autoencoder, not a communication
channel. See "Training Objective Fix" section below.

**Decision:** Fix training objective before proceeding to Phase 2.

**Phase 0: Fix Training Objective (BLOCKER)** ✅ Code done, needs training run
1. ✅ Replace MSE reconstruction with KL distillation (implemented)
2. Train codec on Modal A100 (`python modal/train_m2_codec.py --k 4 --epochs 10`)
   - Note: Local `--local` requires `--data` pointing to real JSONL (not in repo)
   - Modal volume `/data/m2_train.jsonl` has the data
3. Run plumbing proof (`--plumbing-proof <checkpoint> --k 4`)
4. If plumbing proof still ~0% after 10 epochs → upgrade to multi-token distillation
5. Only proceed to Phase 2 if plumbing proof shows L1 > P0

**Phase 2: Clean Gate 1 Run**
1. Verify all validity checks pass
2. Run k=4, k=8 with n=100 episodes each (tighter CIs)
3. Run all ablations: null, random, shuffle
4. Gate passes only if `CI_low >= 0.30` AND all ablations pass

**Phase 3: Gate 2 (only after Phase 2 passes)**
1. Run k=16 with n=100 episodes
2. Gate 2 passes if `CI_low >= 0.34` (50% of P1)

---

##### What the Codec Learns (Option A: Replace Sender Text Message)

**Recommended approach for first success:**
- Training sample is per-turn:
  - inputs: (receiver_prompt_context, sender_text_message)
  - target: receiver behavior under full text message
- Train Encoder(text) → k vectors, Decoder(k vectors) → prefix embeddings
- Objective: match receiver outputs vs the text-teacher

**Rationale:** This is the cleanest bridge from P1 (which works). Alternatives:
- Option B: Encode structured state (more compressible, but requires schema discipline)
- Defer Option B until Option A demonstrates the pipeline works.

---

##### Training Objective Fix (BLOCKER - January 15, 2026)

**Problem:** Current training uses MSE reconstruction loss:
```python
# WRONG: This trains geometric similarity, not semantic communication
loss = F.mse_loss(reconstructed, pooled)
```

This doesn't work because geometric similarity ≠ semantic usefulness as a soft prefix.
The codec learns to compress/reconstruct vectors, but not to communicate meaning.

**Solution: KL Distillation from Text-Teacher**
```python
# Teacher: Agent B with full text message
prompt_teacher = f"Your observation: {obs_B}\nMESSAGE: {message_A}\nANSWER:"
teacher_logits = model(prompt_teacher).logits

# Student: Agent B with injected prefix (no text)
prompt_student = f"Your observation: {obs_B}\nANSWER:"
prefix_embedding = codec.decode(codec.encode(message_embedding))
student_logits = model(inputs_embeds=torch.cat([prefix, prompt_embeds]))

# Train codec to minimize divergence (next-token distribution matching)
loss = F.kl_div(
    F.log_softmax(student_logits[:, -1, :], dim=-1),
    F.softmax(teacher_logits[:, -1, :], dim=-1),
    reduction='batchmean'
)
```

**Key Insight:** The codec must learn to produce prefixes that cause the model to
behave as if it had seen the text message. This requires involving the receiver model
in training, not just reconstructing embeddings.

**Current Limitation: 1-Step Distillation**
The current implementation matches only the first token after "ANSWER:". This is a weak
signal. If plumbing proof remains ~0% after 10 epochs, upgrade to multi-token distillation
using the dataset's `receiver_output`/`receiver_tokens` fields.

**Upgrade Path (if 1-step fails):**
1. **Multi-token distillation:** Match teacher logits for N tokens of the answer sequence
2. **Response-level distillation:** Match full sequence likelihood
3. **REINFORCE:** End-to-end task reward (harder to optimize)

---

##### Evaluation Gates (Pre-committed)

| Gate | Criterion | Interpretation |
|------|-----------|----------------|
| **Gate 1 (Sanity)** | L1 at k∈{4,8} > P0 by ≥10pp | Learned channel carries some signal |
| **Gate 2 (Retention)** | L1 at k=16 ≥ 50% of P1 capability | Codec retains meaningful information |
| **Gate 3 (Budget)** | Every run emits Success + CI + Bits + Tokens + Time | Comparable to text baselines |

---

##### Required Ablations (Bake In From Day One)

| Ablation | Expected Result | Purpose |
|----------|-----------------|---------|
| k=0 / null message | ~P0 (20%) | Proves latent carries signal |
| Random latent (same shape) | ~P0 (20%) | Proves learned structure matters |
| Shuffle sender messages across episodes | Should crater | Proves semantics matter |

---

##### Injection Mechanism

**Use prefix embeddings first** (even if long-term target is more exotic):
- Want to debug: dataset correctness, conditioning, training stability, eval comparability
- Don't add mid-layer surgery complexity until prefix injection works

---

##### Data Composition

**Don't train only on perfect successes:**
- Majority: successful P1 turns (ground truth)
- Some: near-miss / partial-credit turns (if PC metric is meaningful)
- Some: failures (so codec learns not to confidently serialize garbage)

**Practical split:** ~70% success, ~20% partial, ~10% failure

---

##### Cloud GPU Requirements

- **Minimum:** 1× A100 40GB ($2.10/hr on Modal)
- **Recommended:** 4× A100 for parallel k sweeps
- **Estimated Cost:** $50-200 total
- **Estimated Time:** 40-80 GPU-hours

**Cloud Options:**
| Provider | GPU | $/hr | Notes |
|----------|-----|------|-------|
| **Modal.com** ⭐ | A100 40GB | $2.10 | $30/mo free, $10K research grants |
| Lambda Labs | A100 40GB | $1.10 | Reliable |
| RunPod | A100 40GB | $0.74 | Spot instances |

---

**Deliverables:**
- [ ] Encoder-decoder architecture for continuous packets
- [ ] Distillation pipeline from P1 teacher
- [ ] Prefix injection implementation
- [ ] Capability vs k (packet size) curves
- [ ] Ablation results (null, random, shuffle)

**Training Protocol:**
1. Collect P1 episodes (≥1000 per task type) - mixed success/partial/fail
2. Train encoder E: text message → k latent vectors
3. Train decoder D: k vectors → prefix embeddings for receiver
4. Loss: behavior cloning (match P1 outputs) + reconstruction

**Exit Criteria:**
- L1 achieves ≥50% of P1 capability at k=16 (Gate 2)
- Clear capability-k relationship (monotonic or with interpretable knee)
- All three ablations confirm learned structure matters

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

**Local (M0, M1, M4 Inference):** MacBook Pro M3 Max, 98GB unified memory ✅ SUFFICIENT
- Inference with Qwen-2.5-3B: ~2-3 sec/episode
- Test suites and safety evaluation: works well
- NOT sufficient for M2/M3 training (memory + speed limitations)

**Cloud Required for M2/M3 (Training):**
- **Minimum:** 1× A100 40GB ($1.10/hr) or 2× RTX 4090 24GB
- **Recommended:** 4× A100 80GB ($4.40/hr) for faster iteration
- **Estimated Total:** 40-80 GPU-hours, ~$50-200

**Cloud Provider Options:**
| Provider | GPU | $/hr | Est. Total | Notes |
|----------|-----|------|------------|-------|
| **Modal.com** ⭐ | A100 40GB | $2.10 | $0-50 | **$30/mo free, $10K research grants** |
| Lambda Labs | A100 40GB | $1.10 | $50-100 | Reliable, good availability |
| RunPod | A100 40GB | $0.74 | $35-75 | Spot instances available |
| Vast.ai | A100 | $0.50-1.00 | $25-80 | Variable pricing, community |
| Google Colab Pro+ | A100 | $50/mo | $50 | Limited hours/day |

**Recommended: Modal.com**
- $30/month free credits (covers ~14 hours A100)
- $10,000 academic research grants available
- Serverless architecture - no idle costs
- Python-native API, integrates easily with existing code

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
Week 5-8:   [████████] Milestone 1: Latent Baselines ✅ COMPLETE (Negative Result)
Week 9-14:  [████░░░░░░░░] Milestone 2: Continuous Codec ⏳ Gate 1 PASS, Gate 2 pending
Week 15-20: [░░░░░░░░░░░░] Milestone 3: Discrete Codec
Week 21-24: [░░░░░░░░] Milestone 5: Analysis & Writing

Safety Evaluation: [████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] E5 Complete
```

**Current Status (January 15, 2026):**
- **M0:** ✅ Complete - Infrastructure ready
- **M1:** ✅ Complete - Negative result: raw latent doesn't work without training
- **M4:** ✅ E5 Complete - Safety metrics all passed
- **M2-LOCAL-PROTO:** ✅ Complete - Pipeline validated locally
- **M2-SCALE Gate 1:** ✅ **PASS** - k=4: 34%, k=8: 38% (Modal A100)
- **Next:** Fix ablation bug, then Gate 2 (k=16)

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
