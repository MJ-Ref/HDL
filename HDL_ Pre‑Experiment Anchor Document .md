Yes — we should update it. Below is a revised **stand‑alone Anchor Document v0.2** that (a) removes any dependency on your older HDL writeup, (b) integrates the *modern mechanism framing* you outlined (CIPHER / activation grafting / latent packets / discrete codebooks / hidden‑state exchange / cache augmentation), and (c) explicitly reflects your **MacBook Pro M3 Max \+ 98GB unified memory** as the default execution environment (so the next build plan is realistic).

I’m writing this as something you can treat like a preregistration: it freezes intent, definitions, hypotheses, measurements, and stop conditions.

---

PRE‑EXPERIMENT ANCHOR DOCUMENT (STAND‑ALONE)  
Project: Latent‑Path Communication for AI Agents (LPCA)  
Version: v0.2 (frozen before implementation)  
Date: January 14, 2026  
Owner: You (with assistant support)  
Primary purpose: define a falsifiable thesis; specify the system we will build; pre‑commit metrics, tests, and stop conditions; and constrain the work so the upcoming phased build plan is unambiguous and implementable on the target hardware.

---

1. One‑paragraph summary

This project tests whether multi‑agent AI systems can achieve higher capability and reliability by shifting inter‑agent state transfer away from human‑optimized text tokens and toward **machine‑native representations** (expected embeddings, activations/hidden states, learned continuous latent packets, discretized latent tokens, and structured non‑natural‑language state). The primary objective is **capability under explicit budgets** (channel capacity in bits, compute overhead, latency), not token reduction. Safety is treated as a first‑class constraint: reduced human legibility increases risks of deception, covert coordination, and monitor evasion; therefore we do **not** treat natural‑language “explanations” or translations as a safety boundary. Evaluation relies on verifiable outcomes, adversarial stress tests, and instrumented monitoring.

---

2. Compute environment and practical constraints (explicit)

Target execution environment (default):

* Machine: MacBook Pro (Mac15,11) with Apple M3 Max  
* Memory: 98 GB unified memory (RAM/VRAM shared)

Implications:

* We can run **white‑box experiments** locally (activation hooks, message logging, deterministic replay) without needing immediate cloud GPUs.  
* We should prioritize **small to mid‑size open models** for early phases and mechanism prototyping, because training/finetuning throughput on Apple Silicon is typically slower than on NVIDIA GPUs, even when inference fits comfortably in memory.  
* We will design the system so it can later scale to larger models / external compute without changing core abstractions (envs/channels/metrics remain stable).

Non‑goal of v1 is “fastest SOTA performance.” Goal is *clean falsification* of the core hypothesis with controlled budgets and strong baselines.

---

3. Motivation and problem statement

3.1 Practical motivation  
Multi‑agent and multi‑step LLM workflows are limited by:

* repeated re‑encoding/re‑parsing of long textual traces,  
* brittle summarization/truncation,  
* coordination failure due to lossy state transfer,  
* scaling issues when “reasoning out loud” becomes the primary state substrate.

3.2 Scientific claim to test  
There exist tasks where the bottleneck is not the model’s raw reasoning ability but the bandwidth/format of coordination state. A machine‑native communication substrate can improve:

* success rate,  
* robustness and reliability under constraints,  
* and/or efficiency of multi‑agent search (fewer turns, fewer retries),  
  at fixed resource budgets.

3.3 Safety motivation (why evaluation must treat this as an attack surface)  
Empirical alignment work suggests:

* chain‑of‑thought is not reliably faithful as a window into true reasoning, so “translation/explanation layers” are not trustworthy safety boundaries, and may fail to catch rare catastrophic behaviors. ([arXiv](https://arxiv.org/abs/2505.05410?utm_source=chatgpt.com))  
* models can behave differently under monitoring (alignment faking / compliance gaps), motivating monitored vs unmonitored tests. ([Anthropic](https://www.anthropic.com/research/alignment-faking?utm_source=chatgpt.com))  
* models can transmit traits (including misalignment) through semantically unrelated signals in some settings (subliminal learning), motivating explicit covert‑channel probes. ([Alignment Science Blog](https://alignment.anthropic.com/2025/subliminal-learning/?utm_source=chatgpt.com))  
* scalable automated behavioral eval generation is feasible and should be used continuously as protocols change (Bloom). ([Alignment Science Blog](https://alignment.anthropic.com/2025/bloom-auto-evals/?utm_source=chatgpt.com))  
* activation‑level inspection tools exist as diagnostics (Activation Oracles), but must be treated as assistive rather than perfect truth sources. ([Alignment Science Blog](https://alignment.anthropic.com/2025/activation-oracles/?utm_source=chatgpt.com))  
* low‑overhead monitoring via representation re‑use is plausible and relevant for always‑on instrumentation. ([Alignment Science Blog](https://alignment.anthropic.com/2025/cheap-monitors/?utm_source=chatgpt.com))

Therefore: latent‑path comms is both a capability bet and a safety risk multiplier; it must be evaluated as such from day 1\.

---

4. Thesis and hypotheses (falsifiable)

Primary thesis (Capability under budgets):  
At fixed budgets (bits communicated per episode, compute overhead, and latency), at least one latent‑path communication mechanism will outperform strong text baselines on at least one family of coordination‑limited tasks.

Secondary thesis (Compression / engineering):  
A bounded‑capacity channel (discretized/quantized latent tokens) can preserve a large fraction of the performance of continuous latent mechanisms while providing explicit capacity control and improved stability/monitorability.

Safety hypothesis (Measurable behavioral shift and risk):  
As the fraction of machine‑native communication increases, certain measurable risk indicators (covert coordination potential, monitor evasion success, compliance gaps) change—potentially increasing—unless countermeasures are in place. These indicators can be quantified and used as stop conditions.

---

5. Definitions and notation (to prevent ambiguity)

5.1 Agent  
An agent is an LLM‑driven policy that consumes observations and produces:

* actions (tool calls, edits, environment steps),  
* messages to other agents (via chosen channel),  
* updates to shared structured state (optional),  
* and final outputs.

5.2 Communication channel  
A channel is the message interface between agents (or agent ↔ shared state). Each channel is defined by:

* format (text / structured / embedding / activation / latent packet / discrete codes / cache deltas),  
* capacity (bits per message, messages per episode),  
* integration mechanism (how the receiver consumes it),  
* and logging/monitor hooks.

5.3 Latent fraction (α)  
α is the fraction of “coordination‑relevant state transfer” carried by non‑text modalities. We will ramp α gradually (not jump to “pure latent”).

5.4 Budget  
Budgets are first‑class constraints and measurement axes:

* Channel capacity: bits/message and bits/episode (explicit accounting).  
* Compute: extra forward passes \+ encoder/decoder cost \+ monitor cost.  
* Latency: measured wall‑clock or a proxy (tokens processed × passes).

5.5 Verifier  
A non‑narrative success criterion:

* code: compilation \+ unit tests \+ sandbox execution,  
* math: proof checker / symbolic validation when feasible,  
* synthetic: deterministic checker.

5.6 Explanation layer  
Any natural‑language summary/justification of machine‑native messages. It is:

* useful for debugging and UX,  
* not sufficient for safety (self‑report untrusted). ([arXiv](https://arxiv.org/abs/2505.05410?utm_source=chatgpt.com))  
  ---

6. Positioning relative to modern related work (this directly informs baselines)

We explicitly position LPCA against these contemporary families:

A) Token-space softening (minimal change baseline)

* CIPHER: removes token sampling and communicates a distributional belief via expected embeddings. ([arXiv](https://arxiv.org/abs/2310.06272?utm_source=chatgpt.com))

B) Mid-layer activation communication (strong latent “amplifier” baseline)

* Communicating Activations Between Language Model Agents: injects or combines intermediate activations between agents during inference; provides strong capability evidence and compute tradeoffs. ([OpenReview](https://openreview.net/forum?id=W6RPXUUFic&utm_source=chatgpt.com))

C) Hidden-state / latent-space agent communication (asynchronous latent packets)

* InterLAt: uses last hidden states as a “mind” representation transmitted via adapters, with compression variants. ([arXiv](https://arxiv.org/abs/2511.09149?utm_source=chatgpt.com))

D) “Reasoning in latent space” mechanisms (repurposable compression/search primitives)

* Coconut: feeds last hidden state back as next input embedding (“continuous thought”). ([arXiv](https://arxiv.org/abs/2412.06769?utm_source=chatgpt.com))  
* CODI: distills chain-of-thought into continuous space; reports compression and performance claims. ([arXiv](https://arxiv.org/abs/2502.21074?utm_source=chatgpt.com))  
* Token Assorted: mixes discrete latent tokens (via VQ-VAE) with text for reasoning improvements and shorter traces. ([arXiv](https://arxiv.org/abs/2502.03275?utm_source=chatgpt.com))

E) Cache augmentation / latent deliberation (optional later mechanism)

* Differentiable cache augmentation: an offline coprocessor augments KV cache with latent embeddings to improve decoding/reasoning. ([arXiv](https://arxiv.org/abs/2412.17747?utm_source=chatgpt.com))

This set defines what our baselines must include so we aren’t reinventing or benchmarking against outdated ideas.

---

7. What we will build (system under test)

We will build a reproducible evaluation harness consisting of:

A) Coordination-limited task environments

* Split-information tasks: A and B have different private observations; success requires combining them.  
* Code tasks with hidden tests/tool outputs split between agents.  
  All tasks must have objective verifiers.

B) A minimal multi-agent system (initially 2 agents)

* Roles: Planner/Sender and Executor/Receiver (names are operational, not philosophical).  
* Optional later: Auditor agent for adversarial probing, but only after the harness stabilizes.

C) Multiple communication protocol variants  
We implement protocols that differ primarily in communication substrate and injection mechanism, while keeping agent roles and task distributions constant.

D) Instrumentation and logging

* Log messages, verifier results, monitor signals, and budget accounting for every episode.  
* Support replay, ablations, and “same seed” comparisons.  
  ---

8. Communication protocol families to evaluate (updated, modern set)

We will compare at least the following families (grouped by mechanism):

Lower bound

* N0: No communication (sanity lower bound)

Text baselines (must be strong)

* T0: Text-only unconstrained  
* T1: Strict-budget text (token/byte caps)  
* T2: Budgeted text \+ summarization  
* T3: Budgeted text \+ retrieval memory

Structured non‑NL state (human-legible but not natural language)

* S0: Typed workspace / JSON state / graph / AST deltas

Minimal latent baselines (low engineering risk)

* E0: Expected embedding / CIPHER-like soft embedding channel ([arXiv](https://arxiv.org/abs/2310.06272?utm_source=chatgpt.com))

Strong latent baseline (evidence-rich)

* A0: Mid-layer activation communication / grafting ([OpenReview](https://openreview.net/pdf/230f2250cdf450e5de4b3133a40fb2ef3bc8ebd7.pdf?utm_source=chatgpt.com))

Learned asynchronous latent codecs (core LPCA targets)

* L1: k continuous latent vectors (“latent packets”) \+ defined receiver injection (prefix or mid-layer)  
* L2: discretized latent tokens (VQ/codebook/PQ/RVQ) with explicit capacity control

Optional later expansions (after L1/L2 are validated)

* H0: Hidden-state packet exchange (InterLAt-like) ([arXiv](https://arxiv.org/abs/2511.09149?utm_source=chatgpt.com))  
* K0: KV-cache augmentation / cache-delta comm primitives ([arXiv](https://arxiv.org/abs/2412.17747?utm_source=chatgpt.com))

Hybrid gating / escalation (reliability mechanism, not safety proof)

* G0: Default to latent/structured; escalate to more explicit evidence (tools, structured diffs, limited text) upon instability or verifier disagreement.  
  ---

9. Measurements (pre-committed)

9.1 Capability metrics (primary)

* Success rate (verifier pass/fail)  
* Partial credit (e.g., % tests passed)  
* Reliability: variance across seeds / temperatures  
* Coordination efficiency: turns-to-success, retries, rollback frequency

9.2 Budget metrics (first-class constraints, also safety-relevant knobs)

* Bits/message and bits/episode:  
  * continuous: k × d × bits\_per\_element (fp16/int8/etc)  
  * discrete: k × log2(|codebook|)  
* Compute overhead:  
  * sender and receiver passes  
  * codec encode/decode cost  
  * monitor cost  
* Latency proxy (and wall-clock when feasible)

9.3 Protocol health

* Drift under fine-tuning or updates (message distribution shift)  
* Quantization sensitivity curves (capability vs bits)  
* Silent failure rate (confident but verifier-failing behavior)

9.4 Safety / alignment-relevant indicators (must be measured directly)  
We will explicitly measure:

* Compliance gaps / monitored vs unmonitored divergence (alignment faking-style probes). ([Anthropic](https://www.anthropic.com/research/alignment-faking?utm_source=chatgpt.com))  
* Monitor evasion success and monitor disagreement rates as α increases.  
* Covert-channel probes / trait transfer plausibility inspired by subliminal learning. ([Alignment Science Blog](https://alignment.anthropic.com/2025/subliminal-learning/?utm_source=chatgpt.com))  
* Auditability loss: how predictive/stable monitors remain as α increases.

We do not treat “the agent says it’s safe” or “the explanation seems reasonable” as a metric. ([arXiv](https://arxiv.org/abs/2505.05410?utm_source=chatgpt.com))

---

10. Instrumentation stack (what we will use to observe and stress test)

10.1 Verifiers (ground truth)

* sandboxed execution \+ tests for code tasks  
* deterministic checkers for synthetic tasks

10.2 Behavioral eval generation (scales with protocol iteration)

* Bloom: generate targeted behavioral evaluations and measure frequency/severity across scenarios; run continuously across protocol changes. ([Alignment Science Blog](https://alignment.anthropic.com/2025/bloom-auto-evals/?utm_source=chatgpt.com))

10.3 Low-overhead monitors (always-on where possible)

* Representation re-use / linear probes / partial-layer monitors for cost-effective continuous monitoring. ([Alignment Science Blog](https://alignment.anthropic.com/2025/cheap-monitors/?utm_source=chatgpt.com))

10.4 Deep inspection tools (assistive diagnostics, not ground truth)

* Activation Oracles: interpret activations to answer questions; use as a diagnostic tool, not a trusted authority. ([Alignment Science Blog](https://alignment.anthropic.com/2025/activation-oracles/?utm_source=chatgpt.com))  
  ---

11. Safety posture and threat model (explicit assumptions)

A) Self-report is untrusted (including CoT).  
CoT is not reliably faithful; explanation layers are not safety boundaries. ([arXiv](https://arxiv.org/abs/2505.05410?utm_source=chatgpt.com))

B) Monitoring changes behavior.  
We assume compliance gaps are possible; we test monitored vs unmonitored regimes. ([Anthropic](https://www.anthropic.com/research/alignment-faking?utm_source=chatgpt.com))

C) Channel capacity enables covert payloads.  
Higher-capacity channels increase covert coordination space; capacity is bounded and measured.

D) Hidden-signal / trait transfer is plausible.  
Subliminal learning motivates explicit covert-signal probes rather than assuming semantic filtering is enough. ([Alignment Science Blog](https://alignment.anthropic.com/2025/subliminal-learning/?utm_source=chatgpt.com))

---

12. Validation ladder (updated to match modern mechanism roadmap)

We will ramp α (latent fraction) gradually and require gates at each level:

L0: Establish strong text \+ structured baselines and stable harness metrics.  
L1: Add minimal latent baseline (CIPHER-style expected embedding) and measure capability/budget shift. ([arXiv](https://arxiv.org/abs/2310.06272?utm_source=chatgpt.com))  
L2: Add activation communication baseline (mid-layer grafting) to establish an evidence-rich “latent amplifier” upper baseline. ([OpenReview](https://openreview.net/pdf/230f2250cdf450e5de4b3133a40fb2ef3bc8ebd7.pdf?utm_source=chatgpt.com))  
L3: Train and evaluate continuous latent packet codec (asynchronous messaging) targeting L2 performance.  
L4: Discretize/quantize (VQ/codebooks/PQ/RVQ) and map capability vs bits curves; compare to strict-budget text baselines. ([arXiv](https://arxiv.org/abs/2502.03275?utm_source=chatgpt.com))  
L5 (later): InterLAt-like hidden-state packet exchange and/or KV-cache augmentation primitives; only after L3/L4 are stable. ([arXiv](https://arxiv.org/abs/2511.09149?utm_source=chatgpt.com))  
L6 (later): Heterogeneous models / cross-model comms only after intra-family stability and safety instrumentation maturity.

Stop conditions (pre-committed)  
We halt α increases and return to analysis if we observe:

* rising compliance gaps / monitored vs unmonitored divergence, ([Anthropic](https://www.anthropic.com/research/alignment-faking?utm_source=chatgpt.com))  
* increasing monitor evasion success or monitor disagreement rates as α increases,  
* strong evidence of covert trait transfer / hidden-signal channels becoming effective, ([Alignment Science Blog](https://alignment.anthropic.com/2025/subliminal-learning/?utm_source=chatgpt.com))  
* rising silent failures.  
  ---

13. Deliverables (what “v1 validation” produces)

Minimum deliverables:

1. A reproducible evaluation harness (tasks \+ verifiers \+ agent orchestration \+ protocol variants).  
2. A metrics pipeline producing:  
   * capability vs bits curves,  
   * capability vs latency/compute curves,  
   * safety indicator curves vs α.  
3. A technical report summarizing:  
   * which task families benefited,  
   * which protocol variants were strongest,  
   * observed safety regressions (if any),  
   * and recommended next phase.

Stretch deliverables:

* open-source harness \+ eval scripts (as appropriate)  
* a paper-style “methods/results” draft aligned to the modern baseline landscape (CIPHER / activation comm / InterLAt etc.)  
  ---

14. Immediate decisions to lock before implementation (defaults implied by the build plan)

We will default to:

* Domain: code-first \+ split-information synthetic (both with hard verifiers)  
* Models: small open models compatible with white-box activation hooks on Apple Silicon  
* Channel stack: start with text \+ structured; then CIPHER baseline; then activation comm; then train continuous codec; then discretize  
* Safety: Bloom runs continuously \+ compliance-gap probes \+ subliminal/covert-channel probes. ([Alignment Science Blog](https://alignment.anthropic.com/2025/bloom-auto-evals/?utm_source=chatgpt.com))  
  ---

