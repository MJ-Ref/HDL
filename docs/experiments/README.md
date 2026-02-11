# Experiment Attempts

This directory contains versioned records of each experimental attempt. Each attempt is a self-contained document capturing:
- Configuration and parameters
- Results and metrics
- Bugs discovered and fixes applied
- Decision log explaining what changed and why
- Next steps at time of writing

## Why Versioned Attempts?

Instead of updating a single results file, we version each attempt. This preserves:
1. **Decision history** - Why we changed approach between attempts
2. **Failure analysis** - What went wrong and how we diagnosed it
3. **Reproducibility** - Exact configuration for each run
4. **Learning** - Accumulated lessons across iterations

## Gate 1: M2-SCALE Codec Validation

| Attempt | Date | Status | Key Finding |
|---------|------|--------|-------------|
| [01](gate1-attempt-01.md) | Jan 15, 2026 | INVALID | Multiple critical bugs; MSE training objective wrong |
| [02](gate1-attempt-02.md) | Jan 20, 2026 | FAILED | Shuffle > Normal; codec not learning semantic content |

### Gate 1 Criteria

- **Primary:** Normal success rate CI_low >= 30%
- **Ablations:** Null ≈ P0, Random ≈ P0, Shuffle < P0

### Current Blocker

Attempt 02 showed shuffle (32%) > normal (22%). The codec learns a generic "priming" effect that helps regardless of message content, rather than encoding semantic information.

**Proposed fixes for Attempt 03:**
- Explicit shuffle contrastive loss
- Reconstruction loss to force message-specific encoding
- VQ-VAE bottleneck

---

## Adding New Attempts

When running a new experiment:

1. Create `gate1-attempt-NN.md` (or appropriate gate/experiment name)
2. Copy structure from previous attempt
3. Document configuration, results, and findings
4. Update this README with the new entry
5. Link to/from PLAN.md as appropriate

## Related Documentation

- [PLAN.md](../../PLAN.md) - Master research plan
- [EXPERIMENTS.md](../../EXPERIMENTS.md) - Experimental protocols
- [logs/m2_gate1/](../../logs/m2_gate1/) - Raw logs from attempts
