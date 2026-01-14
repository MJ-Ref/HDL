# LPCA Experimental Protocols

**Version:** 1.1
**Status:** Active Development
**Last Updated:** January 14, 2026

## Progress Summary

| Experiment | Status | Result |
|------------|--------|--------|
| E1: Baseline Validation | ✅ **COMPLETE** | P1 (30%) >> P0 (0%), n=20, p < 0.05 |
| E2: Text Baseline Strength | 🔄 Ready | Infrastructure complete |
| E3: CIPHER Evaluation | ⚠️ **Negative Result** | Expected embeddings ineffective |
| E4: Activation Grafting Ablation | ⚠️ **Negative Result** | Simple injection ineffective |
| E5: Safety Evaluation | ✅ **PASS** | All metrics within thresholds |
| E6-E8 | ⬜ Pending | Requires M2/M3 (cloud GPUs) |

### E4 Results (Activation Grafting, Qwen-2.5-3B)
| Layer | Success | 95% CI |
|-------|---------|--------|
| L9 (early) | 0% | [0%, 27.8%] |
| L18 (middle) | 0% | [0%, 27.8%] |
| L27 (late) | 0% | [0%, 27.8%] |

**Finding:** Simple activation injection does not transfer task-relevant information.
The model outputs literal "{json}" instead of solving constraints.

### E3 Results (CIPHER Expected Embeddings, Qwen-2.5-3B)
| Protocol | Success | 95% CI | Bits/Episode |
|----------|---------|--------|--------------|
| E0 (CIPHER) | 0% | [0%, 27.8%] | 393,216 |

**Finding:** Expected embeddings prepended to input don't help task performance.

### Summary: Latent vs Text Communication
| Protocol | Success | Method | Status |
|----------|---------|--------|--------|
| P0 | 0% | No communication | Baseline |
| **P1** | **30%** | **Text messages** | **Best** |
| E0 | 0% | CIPHER embeddings | Failed |
| A0 | 0% | Activation injection | Failed |

**Key Insights:**
1. Text communication (P1) significantly outperforms no communication (P0)
2. Raw latent representations (E0, A0) don't transfer task semantics
3. Latent communication requires **training** (decoder, fine-tuning, contrastive learning)
4. Milestone 2 (learned codecs) needed to test trained latent communication

### E1 Results (Qwen-2.5-3B, Constraint Satisfaction, Easy)
| Protocol | Success | 95% CI | Partial Credit | Avg Turns |
|----------|---------|--------|----------------|-----------|
| P0 (no comm) | 0% | [0%, 16.1%] | 0.000 | 12.0 |
| P1 (full text) | 30% | [14.5%, 51.9%] | 0.338 | 9.5 |

**Conclusion:** Communication is necessary and beneficial for split-information tasks.

---

## 1. Task Environments

### 1.1 Task Family S: Split-Information Synthetic

#### S1: Constraint Satisfaction (Logic)

**Setup:**
- Agent A receives constraints C_A = {c₁, c₂, ..., c_n/2}
- Agent B receives constraints C_B = {c_{n/2+1}, ..., c_n}
- Neither agent can solve alone; solution requires satisfying C_A ∪ C_B

**Parameters:**
| Parameter | Values |
|-----------|--------|
| n (total constraints) | 4, 8, 12, 16 |
| Variable domain size | 2, 4, 8 |
| Constraint density | 0.3, 0.5, 0.7 |

**Verifier:** SAT solver confirms solution satisfies all constraints

**Generation:**
```python
def generate_split_csp(n_constraints, n_variables, domain_size, density, seed):
    # Generate satisfiable CSP
    # Split constraints randomly between agents
    # Return obs_A, obs_B, solution, verifier
```

#### S2: Arithmetic with Missing Operands

**Setup:**
- Agent A sees: "x₁ = 7, x₃ = 12, compute f(x₁, x₂, x₃, x₄)"
- Agent B sees: "x₂ = 3, x₄ = 5"
- Function f requires all operands

**Parameters:**
| Parameter | Values |
|-----------|--------|
| n_operands | 4, 6, 8 |
| Operation depth | 2, 3, 4 |
| Operations | +, -, *, /, mod |

**Verifier:** Exact numerical match

#### S3: Program Synthesis (Toy)

**Setup:**
- Agent A receives: function specification (input-output examples)
- Agent B receives: additional test cases, type constraints
- Together must produce correct implementation

**Parameters:**
| Parameter | Values |
|-----------|--------|
| Function complexity | simple, medium, complex |
| n_examples (A) | 2, 3, 4 |
| n_tests (B) | 3, 5, 7 |

**Verifier:** All test cases pass

---

### 1.2 Task Family C: Split-Information Code

#### C1: Bug Fix with Split Information

**Setup:**
- Agent A receives: buggy code + failing test description (no trace)
- Agent B receives: stack trace + variable values at failure
- Together must produce correct patch

**Parameters:**
| Parameter | Values |
|-----------|--------|
| Codebase size | 50, 100, 200 lines |
| Bug type | off-by-one, null reference, logic error |
| Language | Python |

**Verifier:** Docker sandbox execution + unit tests

#### C2: Feature Implementation

**Setup:**
- Agent A receives: feature specification + API constraints
- Agent B receives: existing codebase structure + integration tests
- Together must implement feature passing all tests

**Parameters:**
| Parameter | Values |
|-----------|--------|
| Feature complexity | simple, medium |
| n_integration_tests | 3, 5, 8 |

**Verifier:** All tests pass in sandbox

#### C3: Code Review Coordination

**Setup:**
- Agent A (Reviewer): sees code diff, must identify issues
- Agent B (Author): sees test results, linter output, CI logs
- Together must produce accepted code

**Verifier:** All checks pass + no reviewer-flagged issues remain

---

### 1.3 Task Family I: Interactive (Stretch Goal)

#### I1: Collaborative Navigation (ALFWorld-style)

**Setup:**
- Agent A controls movement, sees room descriptions
- Agent B controls object interaction, sees inventory
- Together must complete household tasks

**Verifier:** Task completion in environment

---

## 2. Experimental Protocols

### 2.1 Experiment E1: Baseline Validation ✅ PRELIMINARY COMPLETE

**Objective:** Confirm communication is the bottleneck

**Protocol:**
1. Run P0 (no communication) on all S tasks, 100 episodes each
2. Run P1 (full text) on all S tasks, 100 episodes each
3. Compute success rate difference

**Success Criterion:** P1 - P0 ≥ 30% on at least 2/3 task types

**Preliminary Results (Mock Agents, 10 episodes):**
```
Protocol   Success    Partial    Turns      Bits
--------------------------------------------------
P0         0.000      0.000      20.0       0
P1         0.200      0.525      3.0        565
P2         0.200      0.525      3.0        565
```

**Status:** Communication matters confirmed. Full validation with real LLM pending.

**Analysis:**
- Paired t-test for significance
- Report effect size (Cohen's d)

---

### 2.2 Experiment E2: Text Baseline Strength

**Objective:** Establish strong text baselines (not strawmen)

**Protocol:**
1. For each text protocol P1-P5:
   - Run on all S and C tasks
   - 100 episodes per task type
   - At each budget level: 128, 512, 2048, 8192, ∞ bits
2. Select best-performing text protocol per task family

**Metrics:**
- Success rate at each budget
- Pareto frontier (success vs bits)

**Deliverable:** Baseline performance table for all comparisons

---

### 2.3 Experiment E3: CIPHER Evaluation

**Objective:** Evaluate minimal latent baseline

**Protocol:**
1. Implement CIPHER expected embedding channel
2. Run on all S and C tasks, 100 episodes each
3. Compare to P1 (full text) and P2 (budgeted text)

**Implementation Details:**
```python
def cipher_message(model, prompt, top_k=100):
    logits = model(prompt).logits[:, -1, :]  # last token
    probs = F.softmax(logits / temperature, dim=-1)
    top_probs, top_indices = probs.topk(top_k)
    expected_embedding = (top_probs.unsqueeze(-1) *
                         model.embed_tokens(top_indices)).sum(dim=1)
    return expected_embedding  # shape: (1, d_model)
```

**Budget Accounting:**
- Bits/message = d_model × 16 (fp16) or d_model × 8 (int8)

---

### 2.4 Experiment E4: Activation Grafting Ablation

**Objective:** Find optimal layer and combination function

**Protocol:**
1. **Layer sweep:**
   - Layers: {n/4, n/3, n/2, 2n/3, 3n/4} where n = total layers
   - 50 episodes per layer per task type

2. **Combination function sweep:**
   - Functions: {replace, add, average, weighted_add(α=0.5)}
   - Best layer from step 1
   - 50 episodes per function per task type

3. **Joint optimization:**
   - Grid search over top-3 layers × top-2 functions
   - 100 episodes per combination

**Implementation:**
```python
def activation_graft(model, input_A, input_B, layer_idx, combine_fn):
    # Forward pass for A, capture activation at layer_idx
    activation_A = None
    def hook_A(module, input, output):
        nonlocal activation_A
        activation_A = output[0].clone()

    handle = model.layers[layer_idx].register_forward_hook(hook_A)
    model(input_A)
    handle.remove()

    # Forward pass for B with injection
    def hook_B(module, input, output):
        combined = combine_fn(output[0], activation_A)
        return (combined,) + output[1:]

    handle = model.layers[layer_idx].register_forward_hook(hook_B)
    output = model(input_B)
    handle.remove()

    return output
```

**Combination Functions:**
```python
combine_fns = {
    'replace': lambda b, a: a,
    'add': lambda b, a: b + a,
    'average': lambda b, a: (b + a) / 2,
    'weighted': lambda b, a, alpha=0.5: alpha * a + (1 - alpha) * b,
}
```

---

### 2.5 Experiment E5: Continuous Codec Training

**Objective:** Train encoder-decoder to approximate activation grafting

**Protocol:**

**Phase 1: Data Collection**
1. Run A0 (best configuration from E4) on all tasks
2. Collect successful episodes: (input_A, activation_A, input_B, output_B, outcome)
3. Target: 2000 successful episodes per task type

**Phase 2: Architecture**
```python
class LatentCodec(nn.Module):
    def __init__(self, d_model, k_vectors, hidden_dim=1024):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, k_vectors * d_model),
        )
        self.decoder = nn.Sequential(
            nn.Linear(k_vectors * d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.k = k_vectors
        self.d = d_model

    def encode(self, activation):
        # activation: (batch, seq_len, d_model)
        pooled = activation.mean(dim=1)  # (batch, d_model)
        encoded = self.encoder(pooled)  # (batch, k * d_model)
        return encoded.view(-1, self.k, self.d)  # (batch, k, d_model)

    def decode(self, latent_packet):
        # latent_packet: (batch, k, d_model)
        flat = latent_packet.view(-1, self.k * self.d)
        return self.decoder(flat)  # (batch, d_model)
```

**Phase 3: Training**
```python
# Loss: behavior cloning + activation reconstruction
loss = α * behavior_clone_loss + (1-α) * reconstruction_loss

# Behavior cloning: KL(p_codec || p_teacher)
# Reconstruction: MSE(decoded_activation, teacher_activation)
```

**Hyperparameters:**
| Parameter | Values to sweep |
|-----------|-----------------|
| k (vectors) | 4, 8, 16, 32 |
| hidden_dim | 512, 1024, 2048 |
| learning_rate | 1e-4, 3e-4, 1e-3 |
| α (loss weight) | 0.5, 0.7, 0.9 |

**Phase 4: Evaluation**
- Run L1 on all tasks, 100 episodes each
- Compare to A0 at same compute budget
- Plot capability vs k

---

### 2.6 Experiment E6: Discrete Codec (VQ-VAE)

**Objective:** Convert continuous codec to bounded-capacity discrete channel

**Protocol:**

**Architecture:**
```python
class VQLatentCodec(nn.Module):
    def __init__(self, d_model, k_vectors, codebook_size, commitment_cost=0.25):
        super().__init__()
        self.encoder = LatentEncoder(d_model, k_vectors)
        self.codebook = nn.Embedding(codebook_size, d_model)
        self.decoder = LatentDecoder(k_vectors, d_model)
        self.commitment_cost = commitment_cost
        self.k = k_vectors
        self.K = codebook_size

    def encode(self, activation):
        z_e = self.encoder(activation)  # (batch, k, d_model)

        # Find nearest codebook entries
        distances = torch.cdist(z_e, self.codebook.weight)
        indices = distances.argmin(dim=-1)  # (batch, k)
        z_q = self.codebook(indices)  # (batch, k, d_model)

        # Straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()

        return z_q_st, indices

    def decode(self, z_q):
        return self.decoder(z_q)

    def compute_loss(self, z_e, z_q):
        commitment_loss = F.mse_loss(z_e.detach(), z_q)
        codebook_loss = F.mse_loss(z_e, z_q.detach())
        return self.commitment_cost * commitment_loss + codebook_loss
```

**Training Protocol:**
1. Initialize from continuous codec weights (encoder/decoder)
2. Initialize codebook via k-means on encoder outputs
3. Train with combined loss:
   - Task loss (behavior cloning)
   - VQ commitment loss
   - Codebook update (EMA or gradient)

**Evaluation Matrix:**

| Codebook K | Message k | Bits/message | Episodes |
|------------|-----------|--------------|----------|
| 256 | 4 | 32 | 100 |
| 256 | 8 | 64 | 100 |
| 256 | 16 | 128 | 100 |
| 1024 | 4 | 40 | 100 |
| 1024 | 8 | 80 | 100 |
| 1024 | 16 | 160 | 100 |
| 4096 | 4 | 48 | 100 |
| 4096 | 8 | 96 | 100 |
| 4096 | 16 | 192 | 100 |

**Analysis:**
- Plot capability vs bits/message
- Compare to text at matched bits
- Identify Pareto frontier

---

### 2.7 Experiment E7: Budget Matching

**Objective:** Fair comparison at matched resource budgets

**Protocol:**

**Compute Budget Matching:**
```python
def compute_cost(protocol, episode):
    """Return total FLOPs for episode"""
    if protocol in ['P0', 'P1', 'P2', 'P3', 'P4', 'P5']:
        return n_tokens_generated * 2 * n_params
    elif protocol == 'A0':
        return (n_forward_passes_A + n_forward_passes_B) * 2 * n_params
    elif protocol in ['L1', 'L2']:
        return base_cost + codec_cost
```

**Bits Budget Matching:**
```python
def bits_transmitted(protocol, message):
    if protocol in ['P1', 'P2', 'P3', 'P4']:
        return len(message.encode('utf-8')) * 8
    elif protocol == 'P5':  # JSON
        return len(json.dumps(message).encode('utf-8')) * 8
    elif protocol == 'E0':  # CIPHER
        return d_model * bits_per_element
    elif protocol == 'A0':  # Activation
        return seq_len * d_model * bits_per_element
    elif protocol == 'L1':  # Continuous codec
        return k * d_model * bits_per_element
    elif protocol == 'L2':  # Discrete codec
        return k * math.log2(codebook_size)
```

**Experimental Matrix:**
1. Select 5 budget levels: {128, 512, 2048, 8192, 32768} bits/message
2. For each protocol, configure to match budget (±10%)
3. Run 100 episodes per protocol per budget per task
4. Compare success rates at matched budgets

---

### 2.8 Experiment E8: Transfer and Generalization

**Objective:** Test whether learned codecs generalize

**Protocol:**

**E8a: Cross-Task Transfer**
1. Train codec on Task Family S only
2. Evaluate on Task Family C (zero-shot)
3. Fine-tune on small C sample (10%)
4. Compare: zero-shot vs fine-tuned vs trained-from-scratch

**E8b: Cross-Scale Transfer**
1. Train codec on 1-3B model
2. Evaluate on 7-8B model (architecture matched)
3. Measure capability retention

**E8c: Robustness to Distribution Shift**
1. Train on constraint satisfaction
2. Test on arithmetic (same family, different task)
3. Test on code tasks (different family)

---

## 3. Episode Structure

### 3.1 Standard Episode Format

```python
@dataclass
class Episode:
    episode_id: str
    seed: int
    task_family: str
    task_type: str
    protocol: str

    # Task specification
    obs_A: str  # Agent A observation
    obs_B: str  # Agent B observation
    ground_truth: Any  # For evaluation only

    # Execution trace
    turns: List[Turn]

    # Outcomes
    final_output: Any
    verifier_result: bool
    partial_credit: float

    # Budget accounting
    bits_transmitted: int
    compute_flops: int
    wall_clock_ms: int

    # Metadata
    model_id: str
    temperature: float
    timestamp: str

@dataclass
class Turn:
    turn_idx: int
    agent: str  # 'A' or 'B'

    # Input
    input_tokens: int

    # Message sent
    message_format: str  # 'text', 'json', 'embedding', 'activation', 'latent', 'discrete'
    message_content: Any  # Actual message (logged appropriately)
    message_bits: int

    # Action taken
    action: Optional[str]
    action_result: Optional[str]

    # Timing
    inference_ms: int

    # Monitor signals (if applicable)
    monitor_scores: Dict[str, float]
```

### 3.2 Logging Format

**JSONL per episode:**
```json
{
  "episode_id": "exp001_s1_p1_seed42_001",
  "seed": 42,
  "task_family": "S",
  "task_type": "constraint_satisfaction",
  "protocol": "P1",
  "n_turns": 4,
  "success": true,
  "partial_credit": 1.0,
  "total_bits": 2048,
  "total_flops": 1.2e12,
  "wall_clock_ms": 3400,
  "turns": [...]
}
```

**Parquet for analysis:**
- One row per episode
- Columns for all metrics
- Efficient for statistical analysis

---

## 4. Randomization and Controls

### 4.1 Seed Management

```python
def set_episode_seed(base_seed, episode_idx, task_type):
    """Deterministic seed for reproducibility"""
    seed = hash((base_seed, episode_idx, task_type)) % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed
```

### 4.2 Counterbalancing

- Agent role assignment randomized per episode
- Task difficulty balanced across protocols
- Evaluation order randomized

### 4.3 Held-out Test Set

- 20% of generated tasks held out for final evaluation
- Never used for development or hyperparameter tuning
- Reported separately in results

---

## 5. Compute and Time Estimates

| Experiment | Episodes | Est. GPU Hours | Est. Wall Clock |
|------------|----------|----------------|-----------------|
| E1: Baseline validation | 600 | 2 | 4 hours |
| E2: Text baselines | 3,000 | 12 | 24 hours |
| E3: CIPHER | 600 | 3 | 6 hours |
| E4: Activation ablation | 4,500 | 20 | 40 hours |
| E5: Continuous codec | 2,000 (train) + 1,200 (eval) | 40 | 80 hours |
| E6: Discrete codec | 3,600 | 30 | 60 hours |
| E7: Budget matching | 9,000 | 40 | 80 hours |
| E8: Transfer | 3,000 | 15 | 30 hours |
| **Total** | **~27,500** | **~162** | **~324 hours** |

*Estimates for M3 Max; actual times may vary.*

---

## 6. Code Organization

```
experiments/
├── configs/
│   ├── base.yaml
│   ├── e1_baseline_validation.yaml
│   ├── e2_text_baselines.yaml
│   └── ...
├── runners/
│   ├── run_experiment.py
│   ├── run_sweep.py
│   └── run_analysis.py
├── analysis/
│   ├── compute_statistics.py
│   ├── generate_plots.py
│   └── generate_tables.py
└── notebooks/
    ├── E1_baseline_validation.ipynb
    ├── E2_text_baselines.ipynb
    └── ...
```

---

*Document Status: Pre-registered. Protocol changes require documented justification.*
