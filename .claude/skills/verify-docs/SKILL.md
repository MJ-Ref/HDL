# Documentation Verification Skill

Verify that LPCA/HDL documentation files are in sync with current codebase state.

## Primary Documentation Files

1. **PLAN.md** - Master research plan
   - Current status (version, status line)
   - Progress Update section
   - Latest Findings section
   - Milestone status in timeline

2. **PROJECT_STATUS.md** - Quick status reference
   - Should match PLAN.md status
   - Current milestone
   - Blockers/next steps

3. **README.md** - Project overview
   - Installation instructions
   - Quick start guide
   - Links to other docs

4. **SESSION_HANDOFF.md** - Session continuity
   - What was completed
   - What's in progress
   - Next steps with file paths

## Secondary Documentation

5. **EXPERIMENTS.md** - Detailed protocols
6. **docs/experiments/** - Versioned experiment attempts
   - gate1-attempt-01.md (Jan 15 - INVALID)
   - gate1-attempt-02.md (Jan 20 - FAILED)
7. **logs/m2_gate1/RESULTS.md** - Gate 1 raw logs
8. **docs/MODAL_SETUP.md** - Cloud setup

## Verification Checklist

### Cross-Document Consistency
- [ ] PLAN.md version matches actual progress
- [ ] PLAN.md status matches PROJECT_STATUS.md
- [ ] Latest findings in PLAN.md reflect actual results
- [ ] SESSION_HANDOFF.md reflects current state (not stale)
- [ ] No "RERUNNING" or "IN PROGRESS" for completed items

### Staleness Checks
- [ ] PLAN.md "Last Updated" is recent for active work
- [ ] SESSION_HANDOFF.md updated after significant changes
- [ ] docs/experiments/ has entry for latest Gate attempt
- [ ] docs/experiments/README.md index is current

### Key Metrics to Verify
From latest Gate 1 results:
- Normal: 22.0%
- Null: 24.0%
- Shuffle: 32.0%
- Gate 1 status: FAILED

## Output Format

```markdown
## Documentation Verification Report

### Primary Docs
| File | Status | Last Updated | Issues |
|------|--------|--------------|--------|
| PLAN.md | OK/STALE | date | ... |
| PROJECT_STATUS.md | OK/STALE | date | ... |
| SESSION_HANDOFF.md | OK/STALE | date | ... |

### Cross-Document Consistency
- [ ] Status alignment: [OK/MISMATCH]
- [ ] Results alignment: [OK/MISMATCH]

### Action Items
1. [specific fix needed]
```
