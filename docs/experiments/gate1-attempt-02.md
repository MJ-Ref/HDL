# M2-SCALE Gate 1 - Attempt 02

**Date:** January 20, 2026
**Platform:** Modal (A100 40GB GPU)
**Status:** FAILED
**Verdict:** Gate criteria not met; codec not learning semantic content

---

## Summary

Second Gate 1 attempt after fixing critical bugs from Attempt 01. Null baseline bug fixed, all ablations implemented. However, results show shuffle > normal, indicating the codec is NOT encoding semantic content. Gate 1 failed.

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | Qwen/Qwen2.5-3B-Instruct (frozen) |
| Epochs | 10 |
| Batch size | 8 |
| Learning rate | 1e-3 |
| k value | 4 |
| Training objective | L_help (KL-based) |
| Evaluation episodes | n=50 per condition |

---

## Bugs Fixed Since Attempt 01

1. **Null baseline bug** - Eval now uses true zeros instead of `codec.decode(0)` which had LayerNorm bias
2. **Prefix collapse** - Fixed via identity decoder + prefix calibration
3. **Text leakage** - Agent B no longer sees `message_A` in latent conditions
4. **Shuffle ablation** - Implemented and working
5. **Placeholder fallback** - Removed (fail fast)

---

## Results

### Primary Metrics

| Condition | Success Rate | Expected | Status |
|-----------|-------------|----------|--------|
| Normal | **22.0%** (11/50) | > 30% | FAIL |
| Null | 24.0% (12/50) | ~20% | OK (fixed) |
| Random | 28.0% (14/50) | ~20% | Slightly high |
| Shuffle | **32.0%** (16/50) | < 20% | **BACKWARDS** |

### Statistical Summary

| Condition | Success Rate | 95% CI |
|-----------|-------------|--------|
| Normal | 22.0% | [12.8%, 35.2%] |
| Null | 24.0% | [14.3%, 37.4%] |
| Random | 28.0% | [17.5%, 41.7%] |
| Shuffle | 32.0% | [20.8%, 45.8%] |
| P0 baseline | 20.0% | [11.2%, 33.0%] |

**Gate 1 Criterion:** CI_low >= 30.0%
**Result:** CI_low = 12.8% < 30.0% → **FAIL**

### Gate Sweep Results

| scale_gate | Success Rate | Notes |
|------------|--------------|-------|
| 0.00 | 25.0% | No prefix |
| 0.25 | 30.0% | |
| 0.50 | **35.0%** | Peak |
| 0.75 | 30.0% | |
| 1.00 | 30.0% | Default |
| 1.50 | 30.0% | |

### Communication-Matters Subset

- Episodes where Null fails: 38/50
- Normal success on Null-failing episodes: **7.9%** (3/38)

---

## Critical Finding: Shuffle > Normal

The shuffle ablation at **32%** is HIGHER than Normal at **22%**. This is backwards:

- Shuffle uses a message from a DIFFERENT episode (wrong semantic content)
- If the codec encoded semantic information, shuffle should CRATER below P0 (~20%)
- Instead, shuffle OUTPERFORMS the "correct" prefix

**What This Means:**
1. The codec is NOT encoding semantically useful information
2. The learned representations don't depend on message content/token order
3. The prefix may be acting as a generic "priming" signal that helps with any answer
4. The shuffle message provides a DIFFERENT (potentially better?) priming signal

---

## Root Cause Analysis

### Why L_help Converged But Evaluation Failed

Training metrics showed convergence:
- Help loss: 0.71 → 0.05 over 10 epochs
- Mean cosine similarity: ~0.73 (some diversity maintained)
- CE loss: 0.39, Contrast loss: 0.95

But the L_help loss trains: `max(0, NLL_correct - NLL_null + margin)`

This ensures the correct prefix produces lower NLL than null prefix DURING TRAINING. But it doesn't guarantee the prefix encodes MESSAGE-SPECIFIC information. The codec may learn a single "good prefix" that beats null for ALL messages, rather than learning to encode the actual semantic content of each message.

### Null Fix Validation

- Before fix: Null was 28% (same as Random/Shuffle)
- After fix: Null is 24% (now overlaps P0 ~20%)

This confirms the fix worked - Null is now a proper "no information" baseline.

---

## Decision Log

| Decision | Rationale |
|----------|-----------|
| Mark Gate 1 as FAILED | Normal 22% doesn't meet CI_low >= 30% |
| Root cause: not learning semantics | Shuffle > Normal proves prefix is content-agnostic |
| Need architectural changes | L_help objective insufficient for semantic encoding |

---

## Proposed Fixes for Attempt 03

1. **Explicit shuffle contrastive loss:** `NLL_shuffle >> NLL_correct`
   - Directly penalize when shuffled prefix helps as much as correct prefix

2. **Reconstruction loss:** Force encoder to encode message info
   - E.g., predict message tokens from latent

3. **VQ-VAE bottleneck:** Discrete codebook forces distinct codes
   - Prevents collapsing to generic "priming" signal

4. **Information-theoretic regularization:** Maximize mutual information I(message; latent)

---

## Artifacts

- **Checkpoint:** `checkpoints/m2_codec_k4_20260120_042940.pt` (Modal volume)
- **Training data:** `/data/m2_train.jsonl` (Modal volume)
- **Run command:** `modal run modal/train_m2_codec.py --k 4 --epochs 10`

---

## Commits

- `30256f8` - fix: Null baseline inconsistency in evaluation
- `8fc1b31` - doc: Update PLAN with Gate 1 FAIL results (shuffle > normal)

---

## Next Steps

1. Diagnose why shuffle > normal (analyze learned representations)
2. Implement explicit shuffle contrastive loss
3. Consider reconstruction loss or VQ-VAE bottleneck
4. Run Attempt 03 with architectural changes
