# M2-SCALE Gate 1 - Attempt 01

**Date:** January 15, 2026
**Platform:** Modal (A100 40GB GPU)
**Status:** INVALID
**Verdict:** Results unreliable due to critical bugs

---

## Summary

First Gate 1 attempt. Training completed successfully but evaluation was compromised by multiple critical bugs. Results marked INVALID and excluded from gate comparisons.

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | Qwen/Qwen2.5-3B-Instruct (frozen) |
| Epochs | 10 |
| Batch size | 8 |
| Learning rate | 1e-3 |
| Training samples | 1,099 |
| k values tested | 4, 8 |
| Training objective | MSE reconstruction loss |

---

## Raw Results (For Reference Only)

| Config | Final Loss | Normal | Null Msg | Random Latent | Gate 1 |
|--------|-----------|--------|----------|---------------|--------|
| k=4 | 0.097 | 34.0% | 30.0% | 38.0% | INVALID |
| k=8 | 0.111 | 38.0% | 36.0% | 42.0% | INVALID |

**Why These Numbers Are Meaningless:**
- Text leakage: "Normal" tested text communication, NOT the codec
- Placeholder fallback could fake ~30% results
- Missing shuffle ablation (the most diagnostic test)

---

## Critical Bugs Identified

### Bug 1: Placeholder Fallback (Priority 0)

```python
# DANGEROUS CODE:
except ImportError:
    print("Warning: LPCA not available, using placeholder")
    return random.random() < 0.3  # Can fake ~30% results!
```

The `_run_codec_episode()` function returned `random.random() < 0.3` (~30% success) if the LPCA environment failed to import. This alone could explain the 30-42% success rates.

### Bug 2: Text Leakage (Priority 1)

Agent B always received `message_A` in the prompt regardless of ablation:

```python
prompt_B = f"""...MESSAGE from partner: {message_A}..."""
```

This means:
- "Normal" condition tested text communication, NOT the codec
- Null/random ablations didn't actually remove information
- The `prefix_embedding` was computed but never used

### Bug 3: Missing Shuffle Ablation (Priority 1)

The spec requires shuffle ablation ("should crater <P0") but it was not implemented. Shuffle is the most diagnostic test for semantic dependence.

---

## Plumbing Proof Results

| Condition | Success Rate | Expected | Status |
|-----------|-------------|----------|--------|
| P1 Baseline (text present, no injection) | 70.0% | ~68% | OK |
| L1 Injection (text absent, injection on) | **0.0%** | Should match P1 | **FAIL** |

**Root Cause: Wrong Training Objective**

The codec training used MSE reconstruction loss:
```python
loss = F.mse_loss(reconstructed, pooled)
```

This trained geometric similarity, not semantic usefulness. The codec was an autoencoder for embeddings, not a semantic communication channel.

---

## Fixes Applied (for Attempt 02)

1. [x] Remove placeholder fallback (fail fast on import error)
2. [x] Eliminate text leakage (Agent B never sees `message_A` in latent conditions)
3. [x] Implement shuffle ablation
4. [x] Add leak test assertions
5. [x] Use CI-aware gating (not point estimates)
6. [x] Fix KV cache generation
7. [x] Run plumbing proof to verify injection works
8. [x] **Fix training objective** - Switch from MSE to KL distillation

---

## Decision Log

| Decision | Rationale |
|----------|-----------|
| Mark results INVALID | Multiple bugs made evaluation unreliable |
| Fix training objective before rerun | Plumbing proof showed MSE doesn't produce useful prefixes |
| Implement KL distillation | Teacher-student approach trains semantic communication, not reconstruction |
| Add all ablations | Shuffle is the key diagnostic for semantic encoding |

---

## Artifacts

- **Logs:** `logs/m2_gate1/k4_run.log`, `logs/m2_gate1/k8_run.log`
- **Modal URLs (archived):**
  - k=4: https://modal.com/apps/mj-ref/main/ap-yAdVsJBNPGk7wR2qPJO35H
  - k=8: https://modal.com/apps/mj-ref/main/ap-okgtaL44T8RDaHNxDMHUjE

---

## Next Steps (at time of writing)

1. Fix training objective (KL distillation from text-teacher)
2. Fix all evaluation bugs
3. Rerun plumbing proof
4. If plumbing proof passes, run clean Gate 1

See [gate1-attempt-02.md](gate1-attempt-02.md) for the follow-up attempt.
