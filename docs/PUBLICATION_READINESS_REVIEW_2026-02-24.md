# LPCA Publication Readiness Review (2026-02-24)

## Executive Verdict

The project is promising and already has a strong core signal (`P1 >> P0` on tightened tasks), but it is **not yet submission-ready** for top-tier ML venues. The main blocker is not model quality alone; it is **evidence validity and traceability** for latent-baseline results (E3/E4) and documentation consistency.

This review focuses on what will most improve acceptance probability in the shortest time.

## High-Priority Findings (Ordered by Severity)

### [P0] E3 CIPHER generation likely drops prompt/latent context after first token

- Evidence:
  - [`scripts/run_cipher_experiment.py:227`](/Users/mj/Documents/HDL/scripts/run_cipher_experiment.py:227)
  - [`scripts/run_cipher_experiment.py:241`](/Users/mj/Documents/HDL/scripts/run_cipher_experiment.py:241)
  - [`scripts/run_cipher_experiment.py:244`](/Users/mj/Documents/HDL/scripts/run_cipher_experiment.py:244)
- Why this matters:
  - The script computes logits once with `inputs_embeds=combined_embeds`, samples one token, then sets `current_ids = next_token` and autoregresses only from generated tokens.
  - This can invalidate E0 performance claims because receiver decoding is no longer conditioned on the full prompt + soft tokens beyond the first step.
- Publication impact:
  - Any conclusion like "E0 does not work" is vulnerable to desk-reject or reviewer rejection due to implementation confound.

### [P0] E4 activation grafting applies injection only on first token generation path

- Evidence:
  - [`scripts/run_activation_experiment.py:205`](/Users/mj/Documents/HDL/scripts/run_activation_experiment.py:205)
  - [`scripts/run_activation_experiment.py:224`](/Users/mj/Documents/HDL/scripts/run_activation_experiment.py:224)
- Why this matters:
  - Injection is used to get initial logits, then generation continues via plain model forwards without continued grafting/cached injected context.
  - This likely underestimates A0 performance and weakens any negative finding.
- Publication impact:
  - E4 negative result can be challenged as an artifact of decoder path, not a property of activation communication.

### [P1] Evidence traceability mismatch for E3/E4 run scale

- Evidence:
  - Claims: [`PROJECT_STATUS.md:14`](/Users/mj/Documents/HDL/PROJECT_STATUS.md:14), [`EXPERIMENTS.md:12`](/Users/mj/Documents/HDL/EXPERIMENTS.md:12)
  - Available artifacts (example): [`results/cipher_e0_20260114_175809/summary.json:8`](/Users/mj/Documents/HDL/results/cipher_e0_20260114_175809/summary.json:8), [`results/activation_grafting_20260114_175810/summary.json:8`](/Users/mj/Documents/HDL/results/activation_grafting_20260114_175810/summary.json:8)
- Why this matters:
  - Status docs center "n=50 tightened task" narrative, but visible E3/E4 summaries in `results/` are `n=15`.
  - Even if larger runs exist elsewhere, missing linked artifacts hurts reproducibility.
- Publication impact:
  - Reviewer confidence drops when core claims are not directly traceable to released run artifacts.

### [P1] Reproducibility document is materially out of sync with current repo and findings

- Evidence:
  - Outdated claim table: [`REPRODUCIBILITY.md:15`](/Users/mj/Documents/HDL/REPRODUCIBILITY.md:15)
  - Mismatched tree entries (e.g., files/modules not present): [`REPRODUCIBILITY.md:73`](/Users/mj/Documents/HDL/REPRODUCIBILITY.md:73), [`REPRODUCIBILITY.md:83`](/Users/mj/Documents/HDL/REPRODUCIBILITY.md:83), [`REPRODUCIBILITY.md:96`](/Users/mj/Documents/HDL/REPRODUCIBILITY.md:96)
- Why this matters:
  - The reproducibility appendix is a reviewer-facing trust document; currently it is not reliable.
- Publication impact:
  - High risk for negative reproducibility score and credibility concerns.

### [P1] Legacy `modal/train_codec.py` remains runnable but contains placeholder collection + autoencoder objective

- Evidence:
  - MSE objective: [`modal/train_codec.py:231`](/Users/mj/Documents/HDL/modal/train_codec.py:231)
  - Placeholder collection TODO / all-false examples: [`modal/train_codec.py:317`](/Users/mj/Documents/HDL/modal/train_codec.py:317), [`modal/train_codec.py:326`](/Users/mj/Documents/HDL/modal/train_codec.py:326)
- Why this matters:
  - Users may accidentally run an obsolete pipeline that cannot produce valid semantic communication results.
- Publication impact:
  - Reproduction failures and noisy external feedback from collaborators/reviewers.

### [P2] Statistical analysis script can silently use demo data and diverges from metric spec

- Evidence:
  - Demo fallback: [`scripts/analysis/statistical_tests.py:302`](/Users/mj/Documents/HDL/scripts/analysis/statistical_tests.py:302)
  - Uses two-proportion z-test and bootstrap CI: [`scripts/analysis/statistical_tests.py:55`](/Users/mj/Documents/HDL/scripts/analysis/statistical_tests.py:55), [`scripts/analysis/statistical_tests.py:319`](/Users/mj/Documents/HDL/scripts/analysis/statistical_tests.py:319)
  - Metric spec calls for Wilson CI on success rates: [`METRICS.md:39`](/Users/mj/Documents/HDL/METRICS.md:39)
- Why this matters:
  - Analysis should fail fast on missing data and align exactly with preregistered statistical choices.

### [P2] Dataset composition target is declared but not enforced

- Evidence:
  - Target distribution in generator docstring: [`scripts/generate_m2_dataset.py:9`](/Users/mj/Documents/HDL/scripts/generate_m2_dataset.py:9)
  - Actual generated composition: [`data/m2_train.stats.json:4`](/Users/mj/Documents/HDL/data/m2_train.stats.json:4), [`data/m2_train.stats.json:5`](/Users/mj/Documents/HDL/data/m2_train.stats.json:5), [`data/m2_train.stats.json:6`](/Users/mj/Documents/HDL/data/m2_train.stats.json:6)
- Why this matters:
  - Shifted class mixture changes training dynamics and can confound comparisons between attempts.

### [P3] Documentation status drift across primary entry points

- Evidence:
  - README still presents early E1-era status: [`README.md:17`](/Users/mj/Documents/HDL/README.md:17), [`README.md:28`](/Users/mj/Documents/HDL/README.md:28)
  - Current phase in status doc: [`PROJECT_STATUS.md:4`](/Users/mj/Documents/HDL/PROJECT_STATUS.md:4)
- Why this matters:
  - New readers (including reviewers) get conflicting project state narratives.

## What Is Already Strong

- Tightened task appears communication-limited with clear `P1 >> P0` signal:
  - [`results/e1_tightened/llm_constraint_satisfaction_20260114_175702/summary.json`](/Users/mj/Documents/HDL/results/e1_tightened/llm_constraint_satisfaction_20260114_175702/summary.json)
- M2 gate discipline (ablation-minded, CI-aware framing) is directionally solid in current planning:
  - [`PLAN.md`](/Users/mj/Documents/HDL/PLAN.md)
- Core unit/integration suite is healthy for foundational components:
  - Local run: `126 passed`

## Publication Strategy (Realistic, Near-Term)

### Strategy A (Best for Main-Track Acceptance): "Rigorous Negative + Failure-Mode Study"

Frame the paper around:
- Why naive latent channels fail despite high bandwidth.
- How evaluation bugs can produce false latent gains.
- What diagnostics (shuffle/null/random + CI-aware gating) are necessary to avoid self-deception.

This can be competitive if evidence is airtight and multi-model.

### Strategy B (If M2 improves soon): "Latent Codec With Verified Semantic Dependence"

Only pursue if:
- `Normal > P0` with CI-backed margin,
- `Shuffle < P0`,
- and results hold across at least 2 model backbones.

## 4-Week Forward Plan (Publication-Critical)

### Week 1: Validity Repair (must-do)

1. Fix E3/E4 generation context handling with cache-consistent decoding.
2. Add regression tests that would have caught the context-drop bug.
3. Mark `modal/train_codec.py` as deprecated or remove runnable entrypoints.
4. Freeze one canonical "official pipeline" path for latent evaluation.

Exit criteria:
- Re-run E3/E4 sanity checks on 10 episodes with deterministic seeds and confirm behavior changes are expected.

### Week 2: Reproduce Core Table End-to-End

1. Regenerate E1/E2/E3/E4 with a single frozen config bundle.
2. Produce one artifact manifest mapping each table row -> exact JSON files and commit hashes.
3. Remove demo-data fallback in statistical scripts; fail hard when data is missing.
4. Align CIs/tests with preregistered metrics.

Exit criteria:
- Every headline number in `PROJECT_STATUS.md` traceable to a file path + seed range.

### Week 3: External Validity and Power

1. Add at least one additional model family (e.g., Llama-class) for E1/E2/E3/E4.
2. Increase sample sizes for latent baselines to reduce CI width.
3. Run paired-seed comparisons and report robust effect sizes.

Exit criteria:
- Core conclusions stable across models or explicitly bounded as model-specific.

### Week 4: Submission Package

1. Rewrite `README.md`, `PROJECT_STATUS.md`, and `REPRODUCIBILITY.md` for consistency.
2. Prepare camera-ready plots/tables from scripted pipeline only.
3. Draft limitations section emphasizing negative or mixed latent results and safety implications.

Exit criteria:
- Reproducibility checklist can be executed by a third party without manual guesswork.

## Recommended Immediate Edits (Low Effort, High Return)

1. Update README status to current M2 gate reality and link to attempt docs.
2. Add an "Artifact Index" markdown file under `docs/` with run IDs and result file paths.
3. Add warning banner to obsolete scripts (`modal/train_codec.py`) to prevent accidental use.
4. Make analysis scripts fail when no episode data is found.

## Bottom Line

The project can still become publication-grade, but only if it first becomes **artifact-verifiable and evaluation-valid** for E3/E4. Once those validity blockers are fixed, the existing E1/E2 foundation gives a credible path to either:
- a strong negative-results/mechanistic paper, or
- a positive codec paper if M2 clears semantic dependence gates.
