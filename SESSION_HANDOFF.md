# Session Handoff

**Last Updated:** January 20, 2026
**Session Focus:** M2-SCALE Gate 1 Evaluation with Fixed Null Baseline

## What Was Completed This Session

1. **Fixed null baseline bug in evaluation**
   - File: `modal/train_m2_codec.py` (lines ~1479-1492)
   - Bug: Eval null used `codec.decode(0)` which has LayerNorm bias
   - Fix: Eval null now uses actual zeros to match training null

2. **Ran full evaluation on Modal A100**
   - 10 epochs training with L_help loss
   - n=50 evaluation with all ablations
   - Gate sweep (scale_gate 0.0 to 1.5)
   - Communication-matters subset analysis

3. **Updated documentation**
   - PLAN.md updated to v2.3 with Gate 1 FAIL results
   - Committed and pushed to GitHub

## Current Results (Gate 1 FAILED)

| Condition | Success Rate | Expected | Status |
|-----------|-------------|----------|--------|
| Normal    | 22.0% | > 30% | FAIL |
| Null      | 24.0% | ~20% | OK (fixed) |
| Random    | 28.0% | ~20% | Slightly high |
| Shuffle   | **32.0%** | < 20% | **BACKWARDS** |

**Critical Finding:** Shuffle > Normal indicates codec NOT encoding semantic content.

## What's Still In Progress

Nothing actively running. All tasks completed.

## Exact Next Steps

### Immediate (before next training run):
1. **Diagnose shuffle > normal** - Analyze learned representations
   - File: `modal/train_m2_codec.py`
   - Add logging to see what prefixes look like for correct vs shuffled messages
   - Check if certain episode types cause the reversal

2. **Consider architectural changes:**
   - Explicit shuffle contrastive loss: `NLL_shuffle >> NLL_correct`
   - Reconstruction loss to force message-specific encoding
   - VQ-VAE bottleneck for discrete codes

### Files to Modify:
- `modal/train_m2_codec.py` - Training objective changes
- `PLAN.md` - Update with diagnosis results

## Gotchas Discovered

1. **LayerNorm bias in decoder** - Even with identity decoder, `prefix_norm` layer has learned bias that makes `decode(0) != 0`. Must use actual zeros for null ablation.

2. **L_help loss limitation** - Training `max(0, NLL_correct - NLL_null + margin)` ensures correct beats null, but doesn't ensure DIFFERENT messages get DIFFERENT encodings. Codec can learn a single "good prefix" that beats null for ALL messages.

3. **Shuffle ablation is the key diagnostic** - If shuffle doesn't crater, the codec isn't learning semantic content. This is more informative than comparing to null.

## Relevant Commits

- `30256f8` - fix: Null baseline inconsistency in evaluation
- `8fc1b31` - doc: Update PLAN with Gate 1 FAIL results (shuffle > normal)

## Modal Resources

- Checkpoint: `checkpoints/m2_codec_k4_20260120_042940.pt` (on Modal volume)
- Training data: `/data/m2_train.jsonl` (on Modal volume)
- Run command: `modal run modal/train_m2_codec.py --k 4 --epochs 10`
