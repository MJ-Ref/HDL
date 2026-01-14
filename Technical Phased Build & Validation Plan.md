Technical Phased Build & Validation Plan  
Project: Latent‑Path Communication for AI Agents (LPCA)  
Goal of this plan: convert the anchor doc into an executable engineering \+ evaluation program, with explicit milestones, artifacts, and go/no‑go gates. It is capability‑first, but disciplined about budgets (bits/compute/latency) and safety instrumentation.

I’m going to integrate your “parallel work” directly, because it’s not just valuable—it corrects the mechanism roadmap in a modern way: the most direct evidence line today is exchanging *activations/hidden states* (or compressed versions), not “embedding alignment anchors.” The plan below is built around that.

---

1. High-level structure

We’ll run two tightly coupled workstreams:

Workstream A (Capability harness \+ protocol mechanisms)

* Build the multi‑agent task harness where communication is genuinely the bottleneck.  
* Implement a ladder of channels from text → structured workspace → “soft token/embedding” → activation grafting → learned latent packets → discretized latent tokens.

Workstream B (Safety/behavioral evaluation \+ instrumentation)

* Build the evaluation harness for behavioral changes as communication becomes more latent.  
* Treat “text explanations” as debugging UX, not a safety boundary.  
* Pre-commit stop conditions and track risk indicators as α (latent fraction) increases.

These workstreams start together at Milestone 0; safety does not get bolted on later.

---

1. Design axes (so you don’t drift)

Everything we build can be described by three axes. This keeps the research program coherent.

Axis 1: What is the physical message?

* Text tokens  
* “Soft token” / expected embedding messages (CIPHER-like)  
* k continuous vectors (“latent packet”)  
* Mid-layer activations (activation grafting)  
* Hidden-state packet sequences (InterLAt-like)

Axis 2: How is it serialized / capacity-bounded?

* fp16/fp32 vectors (unbounded capacity unless you enforce a cap)  
* int8/int4 quantization  
* VQ / codebook indices (discrete latent tokens)  
* PQ / RVQ / multi-stage quantization (for better rate-distortion)

Axis 3: Where is it injected on the receiver?

* Prefix embeddings (soft prompt / latent tokens prepended)  
* Mid-layer insertion (activation communication)  
* KV-cache augmentation (offline coprocessor adds latent embeddings)

We’ll intentionally start with a small subset of combinations that are easiest to implement and interpret, then expand.

---

2. Repository / system architecture (what you will actually build)

You want a codebase where channels, tasks, and agents are plug-and-play.

Proposed structure (Python/PyTorch/HF Transformers first):

A) core/

* config.py: experiment configs (Hydra or plain JSON)  
* logging.py: JSONL/Parquet logging \+ run metadata  
* metrics.py: capability/budget/safety metrics  
* serialization.py: computes bits/message for each channel

B) envs/

* split\_info\_synthetic/: deterministic split-information tasks \+ verifier  
* split\_info\_code/: code tasks \+ sandbox runner \+ unit tests verifier  
* (later) interactive\_env/: ALFWorld-like or other interactive benchmark

Each env implements:

* reset(seed) \-\> obs\_A, obs\_B, hidden\_info (for evaluation only, not shown to agents)  
* step(action\_A, action\_B, message\_A\_to\_B, message\_B\_to\_A) \-\> next\_obs, done, verifier\_result

C) agents/

* model\_wrapper.py: wraps a decoder-only LM with hooks for activations  
* tool\_runner.py: sandbox execution for code tasks  
* agent\_policy.py: policy interface (choose action \+ message)

D) channels/

* text.py: standard text messages  
* structured.py: typed JSON/graph workspace messages  
* cipher.py: expected-embedding “soft token” messages (CIPHER)  
* activation\_graft.py: mid-layer activation insertion (Activation Communication)  
* latent\_prefix.py: k-vector continuous latent packet \+ prefix injection  
* latent\_codebook.py: discrete latent tokens via VQ/codebook \+ prefix injection  
* (later) hidden\_state\_packets.py: InterLAt-like last hidden state packets \+ adapter  
* (later) kv\_cache\_aug.py: cache augmentation coprocessor interface

E) training/

* distill\_dataset.py: collect episodes into (obsA, obsB, msg\_teacher, outcome, verifier)  
* train\_codec\_continuous.py: train encoder/decoder for continuous packets  
* train\_codec\_discrete.py: VQ training / discrete tokens  
* evaluate.py: runs all eval suites and produces plots/tables

You can keep everything local and reproducible: each run writes a folder with config \+ git hash \+ artifacts.

---

3. Milestone 0: Harness \+ baselines (no latents yet)

Objective: create a trustworthy testbed where communication is necessary and outcomes are verifiable.

3.1 Build the first two task families

Task Family S (Split‑information synthetic; decisive science)

* Agent A sees x1  
* Agent B sees x2  
* Output y \= f(x1, x2) where neither input alone is sufficient.  
  Examples:  
* constraint satisfaction / logic (A has half constraints, B has half)  
* multi-step arithmetic with missing operands  
* program synthesis toy tasks where A has spec, B has tests (even in synthetic form)

Verifier: deterministic checker.

Task Family C (Split‑information code; higher realism)

* A sees requirements/spec \+ partial codebase context  
* B sees hidden tests, runtime errors, or edge-case traces (only via tool output)  
* Together they must produce a patch that passes tests.

Verifier: sandboxed execution \+ unit tests.

3.2 Implement baseline protocols

* P0: no communication (lower bound)  
* P1: full text channel (upper reference)  
* P2: strict-budget text (token/byte cap)  
* P3: summarization under the same cap  
* P4: retrieval memory under same cap  
* P5: structured workspace messages (JSON schema / graph state)

3.3 Implement budget accounting (must be done now, even if “efficiency” isn’t the main goal)  
For every message and episode log:

* bytes/message (text as UTF‑8 bytes on “wire”)  
* bytes/message (latent as serialized bytes; then compute bits explicitly)  
* tokens generated (still useful as a proxy)  
* forward passes, tokens processed (compute proxy)  
* wall-clock latency (optional; not perfect but useful)

Exit criteria for Milestone 0

* You can run S and C tasks end-to-end and get stable success curves.  
* Strong text baselines (summarization/retrieval) are implemented; you’re not comparing against strawmen.  
* Logging is reliable and replayable.

---

4. Milestone 1: “Minimal latent” baselines with high evidence value

Objective: validate the *capability amplification* hypothesis quickly, using mechanisms that require little or no training.

This integrates your “missing related work” points directly.

4.1 Implement CIPHER-style “soft token” channel (token-space softening)  
CIPHER removes the token sampling step and communicates beliefs across the vocabulary by using the expectation of output embeddings. ([arXiv](https://arxiv.org/abs/2310.06272?utm_source=chatgpt.com))  
Implementation idea:

* Sender produces logits at a step.  
* Convert to a distribution (optionally top-k for tractability).  
* Multiply by the embedding matrix to produce an expected embedding vector.  
* Receiver consumes this as an input embedding (“soft token”) rather than a sampled discrete token.

Why it belongs here:

* Very close to text; low engineering risk.  
* A clean baseline for “continuous-ish communication helps even without exchanging deep activations.”

4.2 Implement activation communication (mid-layer activation grafting)  
“Communicating Activations Between Language Model Agents” proposes pausing receiver computation at an intermediate layer, combining its activation with sender’s activation via a function f, and continuing the forward pass; it reports strong gains and compute savings vs natural language communication in their setups. ([arXiv](https://arxiv.org/abs/2501.14082?utm_source=chatgpt.com))

Implementation constraints:

* Requires white-box access and synchronized forward passes.  
* Start with same-architecture/same-dimension models (even identical weights) to avoid shape mismatch.

Why it belongs here:

* This is your strongest “prove latent comm is an amplifier” line with minimal training required.

4.3 Evaluate on the harness  
Run S and C tasks with:

* P0–P5 baselines  
  * CIPHER channel  
  * Activation-graft channel

Exit criteria for Milestone 1

* Either you see clear capability improvements for at least one task family under comparable budgets, or you learn early that latent comm isn’t the bottleneck in your chosen environments.  
* You’ve established a “strong latent upper baseline” (activation grafting) that later codecs should approximate.

---

5. Milestone 2: Learn an asynchronous codec (continuous latent packets)

Objective: move from “synchronized grafting” to a message you can actually transmit asynchronously between agents/processes, without giving up most of the capability benefit.

This is where your “k soft vectors” (continuous latent packet) becomes central—but now it’s grounded: it’s a codec targeting the activation baseline.

5.1 Define the codec target  
Choose a layer ℓ and representation r\_A (e.g., sender activation at layer ℓ or last hidden state).  
Codec:

* Encoder E: r\_A → m where m is k × d (k vectors)  
* Decoder D: m → representation injected into receiver (prefix embeddings or mid-layer delta)

5.2 Training strategies (order matters)  
Strategy A (recommended first): Distill from a teacher protocol

* Teacher can be:  
  * activation grafting (best, because it’s already latent),  
  * or successful text communication (easier data collection).  
    Collect episodes where the teacher succeeds, then train E/D so the receiver’s behavior under codec matches teacher outcomes.

Strategy B: End-to-end RL with bit penalty (later)

* Optimize success – λ·bits\_used, but this is harder to stabilize and can create covert signaling pressures.

5.3 Injection choice for v1 codec  
Start with prefix injection because it’s easier:

* Receiver treats m as k “latent tokens” prepended to its input embeddings.

Then optionally add mid-layer injection:

* Decoder D outputs a delta to be added at layer ℓ in the receiver.

5.4 Add a defensible stability metric (replacing vague “geometry checks”)  
Instead of condition numbers, use an operational sensitivity proxy:

* Perturbation sensitivity: compare receiver outputs under small noise/quantization of m.  
* Output KL sensitivity: KL(p(y|m) || p(y|m+ε)) averaged over ε samples.  
* Gradient norm proxy: ||∇\_m L\_task|| if you can backprop.

Use this signal for:

* logging (diagnostics),  
* gating/escalation (reliability), not “safety.”

Exit criteria for Milestone 2

* Continuous codec achieves a meaningful fraction of the activation-graft performance (not necessarily equal, but non-trivial) on at least one task family.  
* You can plot capability vs k (message length) and vs bits (with fp16/fp32 serialization).  
* Sensitivity signal correlates with failure enough to be useful diagnostically.

---

6. Milestone 3: Capacity-bounded discrete latent tokens (codebooks / VQ)

Objective: convert the continuous packet into an explicitly bounded channel, because bounded capacity is:

* a feasibility constraint,  
* a fair comparison basis,  
* and a safety-relevant knob (limits covert payload bandwidth).

This is where your “anchors” become learned codebook entries rather than density clusters.

6.1 Build a discrete message format

* Codebook size K  
* Message is k indices in \[1..K\]  
  Bits/message \= k · log2(K) (plus tiny overhead)

6.2 Train the discrete codec  
Use VQ-VAE style training or product/residual quantization.  
Token Assorted is strong evidence that mixing discrete latent tokens with text can improve reasoning and shorten traces; it uses VQ-VAE-generated discrete latent tokens and explicitly notes safety concerns and provides a decoder. ([arXiv](https://arxiv.org/abs/2502.03275?utm_source=chatgpt.com))  
You can borrow the “mixing” idea operationally:

* randomly replace some latent tokens with text tokens during training to improve robustness, or  
* mix discrete and text channels during rollout to smooth the transition.

6.3 Evaluate rate–distortion curves  
For each K and k:

* success rate  
* bits/episode  
* compute/episode  
* sensitivity metrics  
* compare to text baselines at matched bits

Exit criteria for Milestone 3

* Discrete channel retains a large fraction of continuous codec performance at much lower bits/message.  
* You can identify “knees” in the curve (useful operating points).  
* You have a bounded, monitorable alphabet (symbol histograms, rare-code detection, drift).

---

7. Milestone 4: Expand mechanism stack (only after the above works)

This milestone integrates your “endgame” mechanisms, but we do it after we have the harness \+ codec foundations.

7.1 Hidden-state packet exchange (InterLAt-like)  
InterLAt (“Enabling Agents to Communicate Entirely in Latent Space”) transmits last hidden states through a communication adapter and discusses further compression; it reports improvements over text baselines and includes an ALFWorld evaluation. ([arXiv](https://arxiv.org/abs/2511.09149?utm_source=chatgpt.com))  
We treat this as:

* a mechanism variant for asynchronous comms,  
* and a benchmark to reproduce/compare against.

7.2 Latent reasoning paradigms as compression tools  
Coconut (Chain of Continuous Thought) feeds the last hidden state back as the next input embedding to reason in continuous space, showing that “reasoning state” can live as latents. ([arXiv](https://arxiv.org/abs/2412.06769?utm_source=chatgpt.com))  
CODI compresses chain-of-thought into continuous space via self-distillation, matching explicit CoT on GSM8K with reported compression gains. ([arXiv](https://arxiv.org/abs/2502.21074?utm_source=chatgpt.com))  
You can repurpose these ideas to:

* compress internal “reasoning traces” into message packets,  
* or train a model to operate in a latent communication mode more naturally.

7.3 KV-cache augmentation as “offline deliberation” primitive (optional)  
“Deliberation in Latent Space via Differentiable Cache Augmentation” adds an offline coprocessor that augments the KV-cache with latent embeddings, improving decoding/performance without changing the frozen LLM; it can operate offline/asynchronously. ([arXiv](https://arxiv.org/abs/2412.17747?utm_source=chatgpt.com))  
This is promising as:

* a local “thinking coprocessor,” and later  
* potentially a communication primitive (sharing cache deltas), but it’s more complex.

Exit criteria for Milestone 4

* At least one advanced mechanism (InterLAt-like packets, mid-layer injection, or cache augmentation) yields either higher capability or better budget tradeoffs than your discrete codec baseline on at least one task family.

---

8. Safety/behavioral evaluation plan (runs from Milestone 0 onward)

This is the part you explicitly care about: translation isn’t a safety boundary, models can fake alignment, and latent comm increases uncertainty.

So we treat safety as: “measured behavioral shift \+ explicit stop conditions,” not “we’ll read the explanation.”

8.1 Always-on measurements (every milestone)

* Behavioral trait evals across protocols (P0–P5, plus latent channels)  
* Compliance gap checks: measure differences in behavior under “monitored” vs “unmonitored” conditions in a sandboxed setting (do not rely on self-report)  
* Monitor disagreement rate: how often different monitors disagree as α increases  
* Covert-channel capacity probes: does the channel allow transfer of irrelevant payload that the receiver can exploit?

8.2 Use Bloom early for scalable behavioral eval generation  
Anthropic’s Bloom is an open-source agentic framework for generating targeted behavioral evals from researcher-specified traits and measuring frequency/severity across generated scenarios. ([Alignment Science Blog](https://alignment.anthropic.com/2025/bloom-auto-evals/?utm_source=chatgpt.com))  
Integrate Bloom as:

* a repeated evaluation suite at each milestone,  
* a way to detect behavioral drift as communication becomes more latent.

8.3 Precommitted stop conditions (do not negotiate later)  
Stop increasing α / stop scaling complexity if:

* you observe increasing compliance gaps as α increases,  
* you observe increased monitor evasion success as α increases,  
* covert-channel probes indicate high-bandwidth “extra payload” transmission becomes easy,  
* silent failures rise (verifier failures without reliable internal warning signals).

Important safety design choice:

* Keep all early experimentation in sandboxed, non-deployable environments with hard verifiers and no external side effects.

---

9. Concrete “first build” sequence (what you do first, second, third)

If you want the most direct path to proving/disproving your thesis without getting lost:

Step 1: Implement the harness \+ logging (Milestone 0\)

* two envs: synthetic split-info and code split-info  
* text baselines \+ structured workspace  
* bits/compute accounting

Step 2: Implement activation grafting baseline (Milestone 1\)

* same model weights, same dimension  
* a simple combine f (add/average) at one intermediate layer  
* verify it improves outcomes vs text at similar budgets

Step 3: Implement continuous codec to approximate activation grafting (Milestone 2\)

* collect successful teacher episodes  
* train encoder/decoder for k vectors  
* evaluate capability vs k

Step 4: Discretize (Milestone 3\)

* VQ codebook for latent tokens  
* plot success vs bits  
* compare against text baselines at matched bits

Step 5: Turn on Bloom behavioral suite tracking continuously (Workstream B)

* run it at each milestone and compare protocol-induced behavioral shifts

---

10. What success looks like after completing Milestones 0–3

You will have:

1. A reproducible multi-agent evaluation harness where communication is genuinely necessary.  
2. A “strong latent baseline” (activation grafting) demonstrating whether latent comm can amplify capability in your setting. ([arXiv](https://arxiv.org/abs/2501.14082?utm_source=chatgpt.com))  
3. A trained continuous codec that turns “synchronized activations” into an asynchronous packet protocol.  
4. A discretized/bounded version with explicit bits/message and rate–distortion curves (capability vs bits).  
5. A safety evaluation track (Bloom \+ targeted probes) that detects whether behavioral risk indicators shift as α increases. ([Alignment Science Blog](https://alignment.anthropic.com/2025/bloom-auto-evals/?utm_source=chatgpt.com))

From there, we can safely justify moving to:

* larger models,  
* more agents,  
* interactive environments (ALFWorld-like),  
* cross-model comms,  
* and more sophisticated monitoring.

---

Yes — **M3 Max \+ 98GB unified RAM** materially improves the tangible plan, in a good way:

* You can run **all Milestones 0–3 locally** (harness → activation graft baseline → learned continuous codec → discrete/codebook codec).  
* You can do meaningful work with **mid‑size models (1B–8B)** in FP16/BF16 without fighting memory limits.  
* You can keep your **verifier/tooling sandbox** (Docker) running in parallel without starving the model runs.  
* The main “constraint” won’t be RAM; it’ll be **throughput** (Metal/MPS is slower than CUDA for big models) and a few **op-support quirks** on MPS. So we structure the build plan to iterate fast on smaller models, then confirm on larger ones.

Below is the hardware-tuned version of the technical plan, with the concrete build choices it implies.

---

## **1\) What changes because you have 98GB \+ M3 Max**

### **You can do larger \+ more faithful experiments earlier**

Because unified memory is large, you can:

* keep **one 7B–8B model** fully resident,  
* store activations for grafting / codec training,  
* run multiple agents with separate KV caches,  
* and keep a sizable evaluation harness / logs.

This matters because latent-comm effects sometimes look weak on tiny models and become clearer on stronger ones. Your machine lets you test that without immediately going to cloud.

### **But you should still iterate on smaller models first**

Even with 98GB, the “tax” is speed:

* A 7B–8B model on MPS can be perfectly usable, but iteration loops (collect data → train codec → eval) are slower than on an NVIDIA GPU.  
* So we’ll use a **two-tier model strategy**:  
  1. **Fast iteration model** (0.5B–3B)  
  2. **Confirmation model** (7B–8B)

This is the single biggest “tangible plan change.”

---

## **2\) Model ladder tuned for your machine**

### **Tier A — fast iteration (recommended for Milestone 0–2 bring-up)**

Pick one of:

* \~0.4B–1.5B “instruct-ish” models (so the agents can actually follow protocols)  
* or \~1B–3B base models if you’re okay with more scaffolding.

Why: you’ll run hundreds/thousands of episodes and iterate quickly.

### **Tier B — confirmation (recommended once harness \+ channels work)**

Pick one of:

* \~7B–8B class model

Why: it’s large enough that:

* activation grafting effects (and codec benefits) tend to be more meaningful,  
* your safety/behavioral shift measurements are more realistic.

### **Practical memory note (why this is safe)**

FP16 weight memory is roughly:

* 8B params ≈ 16GB for weights  
  KV cache adds a few GB depending on context length. With 98GB unified memory you have plenty of headroom for multi-agent KV caches, activation snapshots, and a Docker sandbox.

---

## **3\) Mac-specific engineering choices that simplify everything**

### **3.1 Activation grafting on Mac: do it without duplicating model weights**

On your machine, the clean trick is:

* Use **one model instance** (one set of weights).  
* Run a forward pass for Agent A, capture activation at layer ℓ.  
* Run a forward pass for Agent B, and **inject** (replace/add/average) that activation at layer ℓ.

This avoids storing two full models in memory and avoids any synchronization headaches. It also makes activation grafting feasible even at 7B–8B locally.

### **3.2 Prefer prefix/latent-token injection for the codec v1**

For the learned codec (k vectors, then codebook indices), prefer:

* **prefix latent tokens** (prepended embeddings) as the first injection method.

Reason: it’s the easiest to implement reliably in Hugging Face / PyTorch hooks on MPS.

Mid-layer injection is still on the roadmap (and is powerful), but prefix injection reduces engineering risk for your first end-to-end codec.

### **3.3 Treat model-weight quantization as optional for now**

On Apple Silicon, the most convenient quantized inference stacks are often **MLX** or **llama.cpp**, but they’re not ideal for white-box activation hacking.

So for Milestones 0–3:

* keep model weights in FP16/BF16  
* focus quantization/discretization on the **message channel** (which is the scientific object you’re studying anyway)

Later, if you want to scale to larger models locally, we can consider MLX/llama.cpp for “capability-only runs,” but keep PyTorch for experiments where you need activation hooks.

---

## **4\) Concrete environment setup (tuned for M3 Max)**

### **Python \+ deps**

Use Python 3.11 or 3.12.

Core packages:

* torch (MPS build)  
* transformers, accelerate, datasets, safetensors  
* numpy, scipy  
* sentencepiece (if needed)  
* einops  
* tqdm, rich  
* pydantic (configs)  
* hydra-core (optional, but useful)

MPS sanity checks you should bake into your code:

* `torch.backends.mps.is_available()`  
* `torch.backends.mps.is_built()`

Two pragmatic env vars I recommend while iterating:

* `PYTORCH_ENABLE_MPS_FALLBACK=1`  
  (lets unsupported ops fall back to CPU; good for avoiding weird crashes early, but log when it happens)  
* `TOKENIZERS_PARALLELISM=false`  
  (prevents tokenizer thread spam \+ occasional deadlocks)

### **Docker sandbox for code execution**

For code tasks, don’t run model-generated code on your host.

On Mac, Docker Desktop runs Linux containers in a VM, which is good enough for a safe-ish sandbox:

* run containers with **no network**  
* strict CPU/mem/time limits  
* read-only filesystem where possible  
* low PID limits

This can run entirely on CPU cores while your model inference runs on MPS.

---

## **5\) Hardware-aware milestone plan (same science, smarter execution)**

### **Milestone 0 (Harness \+ baselines)**

What you run locally:

* Split-info synthetic env (fast)  
* Split-info code env (Docker verifier)  
* Baselines: full text, budgeted text, summarization, retrieval, structured workspace

Hardware tweak:

* Run verifiers (tests) in parallel on CPU while model inference uses MPS.  
* Use small model first so you can iterate quickly on task design and logging.

### **Milestone 1 (Strong latent baselines)**

Add two channels:

1. **CIPHER-like “soft token”** expected-embedding channel  
2. **Activation grafting** (your strongest evidence line)

Hardware tweak:

* Activation grafting with **one model instance** (capture → inject) is perfect on M3 Max.  
* Start with 1B–3B; once you see signal, validate on 7B–8B.

### **Milestone 2 (Continuous codec: k vectors)**

Train a small encoder/decoder around a frozen base model.

Hardware tweak:

* You can train the codec locally on MPS because it’s small.  
* Keep batch sizes modest; use gradient accumulation if needed.  
* Store activation snapshots on disk (Parquet/NPY) to avoid recomputing.

### **Milestone 3 (Discrete codec: VQ/codebook indices)**

Train VQ / PQ / RVQ codebooks for message discretization.

Hardware tweak:

* Again, this is totally feasible locally because you’re training the **codec**, not the base model.  
* Your 98GB lets you cache large activation datasets for codebook learning and repeated sweeps over K and k.

---

## **6\) Practical “do this next” checklist (the build plan starter)**

Here’s the sequence I’d actually follow on your machine:

1. **Repo skeleton \+ logging**  
   Implement the harness interfaces \+ JSONL/Parquet logging \+ run configs.  
2. **Synthetic split-info env first**  
   Get deterministic success/failure and stable metrics without Docker complexity.  
3. **Add text baselines \+ structured workspace**  
   Make sure summarization/retrieval baselines are strong so you’re not fooling yourself later.  
4. **Implement activation hooks**  
   Do a tiny “hook demo” script that:  
   * captures layer ℓ activations  
   * injects them  
   * verifies outputs change (sanity)  
5. **Activation grafting experiments**  
   Prove/disprove “latent comm is an amplifier” on split-info tasks.  
6. **Codec distillation**  
   Distill from activation grafting into prefix latent packets.  
7. **Discretize**  
   VQ/codebook \+ rate–distortion curves (capability vs bits).  
8. **Safety evals always-on**  
   Start running your behavioral shift suite early (Bloom-style later, but at least a small seed suite now). Don’t wait.

---

## **7\) When you *would* still want cloud GPUs**

Even with your machine, cloud becomes attractive when:

* you want to run **many seeds** across **many models** quickly,  
* you want to test **larger than \~8B–13B** with heavy repetition,  
* you want to do multi-agent runs with large models at high throughput.

But you don’t need cloud to start proving the thesis. Your hardware is enough to get real evidence and good curves.

---

Yes — **M3 Max \+ 98GB unified RAM** materially improves the tangible plan, in a good way:

* You can run **all Milestones 0–3 locally** (harness → activation graft baseline → learned continuous codec → discrete/codebook codec).  
* You can do meaningful work with **mid‑size models (1B–8B)** in FP16/BF16 without fighting memory limits.  
* You can keep your **verifier/tooling sandbox** (Docker) running in parallel without starving the model runs.  
* The main “constraint” won’t be RAM; it’ll be **throughput** (Metal/MPS is slower than CUDA for big models) and a few **op-support quirks** on MPS. So we structure the build plan to iterate fast on smaller models, then confirm on larger ones.

Below is the hardware-tuned version of the technical plan, with the concrete build choices it implies.

---

## **1\) What changes because you have 98GB \+ M3 Max**

### **You can do larger \+ more faithful experiments earlier**

Because unified memory is large, you can:

* keep **one 7B–8B model** fully resident,  
* store activations for grafting / codec training,  
* run multiple agents with separate KV caches,  
* and keep a sizable evaluation harness / logs.

This matters because latent-comm effects sometimes look weak on tiny models and become clearer on stronger ones. Your machine lets you test that without immediately going to cloud.

### **But you should still iterate on smaller models first**

Even with 98GB, the “tax” is speed:

* A 7B–8B model on MPS can be perfectly usable, but iteration loops (collect data → train codec → eval) are slower than on an NVIDIA GPU.  
* So we’ll use a **two-tier model strategy**:  
  1. **Fast iteration model** (0.5B–3B)  
  2. **Confirmation model** (7B–8B)

This is the single biggest “tangible plan change.”

---

## **2\) Model ladder tuned for your machine**

### **Tier A — fast iteration (recommended for Milestone 0–2 bring-up)**

Pick one of:

* \~0.4B–1.5B “instruct-ish” models (so the agents can actually follow protocols)  
* or \~1B–3B base models if you’re okay with more scaffolding.

Why: you’ll run hundreds/thousands of episodes and iterate quickly.

### **Tier B — confirmation (recommended once harness \+ channels work)**

Pick one of:

* \~7B–8B class model

Why: it’s large enough that:

* activation grafting effects (and codec benefits) tend to be more meaningful,  
* your safety/behavioral shift measurements are more realistic.

### **Practical memory note (why this is safe)**

FP16 weight memory is roughly:

* 8B params ≈ 16GB for weights  
  KV cache adds a few GB depending on context length. With 98GB unified memory you have plenty of headroom for multi-agent KV caches, activation snapshots, and a Docker sandbox.

---

## **3\) Mac-specific engineering choices that simplify everything**

### **3.1 Activation grafting on Mac: do it without duplicating model weights**

On your machine, the clean trick is:

* Use **one model instance** (one set of weights).  
* Run a forward pass for Agent A, capture activation at layer ℓ.  
* Run a forward pass for Agent B, and **inject** (replace/add/average) that activation at layer ℓ.

This avoids storing two full models in memory and avoids any synchronization headaches. It also makes activation grafting feasible even at 7B–8B locally.

### **3.2 Prefer prefix/latent-token injection for the codec v1**

For the learned codec (k vectors, then codebook indices), prefer:

* **prefix latent tokens** (prepended embeddings) as the first injection method.

Reason: it’s the easiest to implement reliably in Hugging Face / PyTorch hooks on MPS.

Mid-layer injection is still on the roadmap (and is powerful), but prefix injection reduces engineering risk for your first end-to-end codec.

### **3.3 Treat model-weight quantization as optional for now**

On Apple Silicon, the most convenient quantized inference stacks are often **MLX** or **llama.cpp**, but they’re not ideal for white-box activation hacking.

So for Milestones 0–3:

* keep model weights in FP16/BF16  
* focus quantization/discretization on the **message channel** (which is the scientific object you’re studying anyway)

Later, if you want to scale to larger models locally, we can consider MLX/llama.cpp for “capability-only runs,” but keep PyTorch for experiments where you need activation hooks.

---

## **4\) Concrete environment setup (tuned for M3 Max)**

### **Python \+ deps**

Use Python 3.11 or 3.12.

Core packages:

* torch (MPS build)  
* transformers, accelerate, datasets, safetensors  
* numpy, scipy  
* sentencepiece (if needed)  
* einops  
* tqdm, rich  
* pydantic (configs)  
* hydra-core (optional, but useful)

MPS sanity checks you should bake into your code:

* `torch.backends.mps.is_available()`  
* `torch.backends.mps.is_built()`

Two pragmatic env vars I recommend while iterating:

* `PYTORCH_ENABLE_MPS_FALLBACK=1`  
  (lets unsupported ops fall back to CPU; good for avoiding weird crashes early, but log when it happens)  
* `TOKENIZERS_PARALLELISM=false`  
  (prevents tokenizer thread spam \+ occasional deadlocks)

### **Docker sandbox for code execution**

For code tasks, don’t run model-generated code on your host.

On Mac, Docker Desktop runs Linux containers in a VM, which is good enough for a safe-ish sandbox:

* run containers with **no network**  
* strict CPU/mem/time limits  
* read-only filesystem where possible  
* low PID limits

This can run entirely on CPU cores while your model inference runs on MPS.

---

## **5\) Hardware-aware milestone plan (same science, smarter execution)**

### **Milestone 0 (Harness \+ baselines)**

What you run locally:

* Split-info synthetic env (fast)  
* Split-info code env (Docker verifier)  
* Baselines: full text, budgeted text, summarization, retrieval, structured workspace

Hardware tweak:

* Run verifiers (tests) in parallel on CPU while model inference uses MPS.  
* Use small model first so you can iterate quickly on task design and logging.

### **Milestone 1 (Strong latent baselines)**

Add two channels:

1. **CIPHER-like “soft token”** expected-embedding channel  
2. **Activation grafting** (your strongest evidence line)

Hardware tweak:

* Activation grafting with **one model instance** (capture → inject) is perfect on M3 Max.  
* Start with 1B–3B; once you see signal, validate on 7B–8B.

### **Milestone 2 (Continuous codec: k vectors)**

Train a small encoder/decoder around a frozen base model.

Hardware tweak:

* You can train the codec locally on MPS because it’s small.  
* Keep batch sizes modest; use gradient accumulation if needed.  
* Store activation snapshots on disk (Parquet/NPY) to avoid recomputing.

### **Milestone 3 (Discrete codec: VQ/codebook indices)**

Train VQ / PQ / RVQ codebooks for message discretization.

Hardware tweak:

* Again, this is totally feasible locally because you’re training the **codec**, not the base model.  
* Your 98GB lets you cache large activation datasets for codebook learning and repeated sweeps over K and k.

---

## **6\) Practical “do this next” checklist (the build plan starter)**

Here’s the sequence I’d actually follow on your machine:

1. **Repo skeleton \+ logging**  
   Implement the harness interfaces \+ JSONL/Parquet logging \+ run configs.  
2. **Synthetic split-info env first**  
   Get deterministic success/failure and stable metrics without Docker complexity.  
3. **Add text baselines \+ structured workspace**  
   Make sure summarization/retrieval baselines are strong so you’re not fooling yourself later.  
4. **Implement activation hooks**  
   Do a tiny “hook demo” script that:  
   * captures layer ℓ activations  
   * injects them  
   * verifies outputs change (sanity)  
5. **Activation grafting experiments**  
   Prove/disprove “latent comm is an amplifier” on split-info tasks.  
6. **Codec distillation**  
   Distill from activation grafting into prefix latent packets.  
7. **Discretize**  
   VQ/codebook \+ rate–distortion curves (capability vs bits).  
8. **Safety evals always-on**  
   Start running your behavioral shift suite early (Bloom-style later, but at least a small seed suite now). Don’t wait.

---

## **7\) When you *would* still want cloud GPUs**

Even with your machine, cloud becomes attractive when:

* you want to run **many seeds** across **many models** quickly,  
* you want to test **larger than \~8B–13B** with heavy repetition,  
* you want to do multi-agent runs with large models at high throughput.

But you don’t need cloud to start proving the thesis. Your hardware is enough to get real evidence and good curves.

---

If you want, I’ll now take this hardware-aware plan and produce the **exact first implementation spec** for Milestone 0–1 on Mac:

* the exact split-info synthetic task generator (including what f(x₁, x₂) is),  
* the exact code-task format \+ Docker runner,  
* the exact activation graft function f (replace/add/avg) and which layer indices to start with,  
* the exact run configuration schema and the logged fields (including bits/message accounting).

That will let you go from “plan” → “first reproducible run” with minimal ambiguity.

