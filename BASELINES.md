# LPCA Baseline Specifications

**Version:** 1.0
**Status:** Pre-registered

---

## 1. Design Philosophy

Strong baselines are essential for credible research. We follow these principles:

1. **No strawmen:** Text baselines must use modern techniques (summarization, retrieval)
2. **Fair budgets:** Compare at matched resource constraints
3. **Optimized hyperparameters:** Tune baselines with same care as novel methods
4. **Documented choices:** Every implementation decision is justified

---

## 2. Text Baselines

### 2.1 P0: No Communication (Lower Bound)

**Purpose:** Establish floor performance when agents cannot coordinate.

**Implementation:**
```python
class NoCommProtocol:
    def send_message(self, agent_state) -> None:
        return None  # No message sent

    def receive_message(self, message) -> str:
        return ""  # No message received
```

**Expected Behavior:**
- Performance should be significantly below P1
- If P0 ≈ P1, communication is not the bottleneck (redesign tasks)

---

### 2.2 P1: Full Text (Upper Reference)

**Purpose:** Establish ceiling for text-based communication.

**Implementation:**
```python
class FullTextProtocol:
    def __init__(self, max_tokens: int = 2048):
        self.max_tokens = max_tokens

    def format_message(self, content: str) -> str:
        # Standard text message
        return content[:self.max_tokens]

    def parse_message(self, message: str) -> str:
        return message
```

**System Prompt for Agents:**
```
You are Agent {A/B} in a collaborative task. Communicate with your partner
via natural language messages. Your goal is to {task_description}.

Be clear, concise, and informative in your messages. Include all relevant
information your partner needs to succeed.
```

**Hyperparameters:**
| Parameter | Value | Justification |
|-----------|-------|---------------|
| max_tokens | 2048 | Sufficient for most coordination needs |
| temperature | 0.7 | Balance creativity and consistency |

---

### 2.3 P2: Strict-Budget Text

**Purpose:** Text communication under explicit capacity constraints.

**Implementation:**
```python
class BudgetedTextProtocol:
    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes

    def format_message(self, content: str) -> str:
        encoded = content.encode('utf-8')
        if len(encoded) > self.max_bytes:
            # Truncate at character boundary
            while len(content.encode('utf-8')) > self.max_bytes:
                content = content[:-1]
        return content

    def bits_used(self, message: str) -> int:
        return len(message.encode('utf-8')) * 8
```

**Budget Levels:**
| Level | Bytes | Bits | ~Tokens |
|-------|-------|------|---------|
| Minimal | 16 | 128 | ~4 |
| Low | 64 | 512 | ~16 |
| Medium | 256 | 2048 | ~64 |
| High | 1024 | 8192 | ~256 |
| Uncapped | ∞ | ∞ | ∞ |

**System Prompt Addition:**
```
IMPORTANT: Your message must be at most {max_bytes} bytes. Be extremely
concise while preserving critical information.
```

---

### 2.4 P3: Text + Summarization

**Purpose:** Use LLM compression to maximize information density under budget.

**Implementation:**
```python
class SummarizationProtocol:
    def __init__(self, max_bytes: int, summarizer_model: str):
        self.max_bytes = max_bytes
        self.summarizer = load_model(summarizer_model)

    def compress_message(self, content: str) -> str:
        prompt = f"""Summarize the following message in at most {self.max_bytes} bytes.
Preserve all critical information needed for task completion.

Original message:
{content}

Compressed message:"""

        compressed = self.summarizer.generate(prompt, max_tokens=self.max_bytes // 4)
        return self._truncate_to_budget(compressed)

    def _truncate_to_budget(self, text: str) -> str:
        while len(text.encode('utf-8')) > self.max_bytes:
            text = text[:-1]
        return text
```

**Summarizer Options:**
| Model | Quality | Speed | Use Case |
|-------|---------|-------|----------|
| Same as agent | Highest | Slow | Accuracy-first |
| Smaller model | Good | Fast | Budget-first |
| Distilled summarizer | Good | Fastest | Production |

**Default:** Use same model as agent (conservative choice for research).

---

### 2.5 P4: Text + Retrieval Memory

**Purpose:** Episodic memory with retrieval for efficient long-context coordination.

**Implementation:**
```python
class RetrievalProtocol:
    def __init__(self, max_bytes: int, memory_size: int = 100,
                 retrieval_k: int = 3):
        self.max_bytes = max_bytes
        self.memory = EpisodicMemory(max_size=memory_size)
        self.retriever = DenseRetriever()
        self.retrieval_k = retrieval_k

    def send_message(self, content: str, query: str) -> str:
        # Store in memory
        self.memory.add(content, embedding=self.retriever.encode(content))

        # Compress for transmission
        compressed = self._compress(content)
        return compressed

    def receive_message(self, message: str, context: str) -> str:
        # Store received message
        self.memory.add(message, embedding=self.retriever.encode(message))

        # Retrieve relevant history
        retrieved = self.memory.retrieve(context, k=self.retrieval_k)

        # Return augmented context
        return self._format_with_retrieval(message, retrieved)

    def _format_with_retrieval(self, current: str, retrieved: List[str]) -> str:
        return f"""Current message: {current}

Relevant history:
{chr(10).join(f'- {r}' for r in retrieved)}"""
```

**Memory Implementation:**
```python
class EpisodicMemory:
    def __init__(self, max_size: int):
        self.entries = []
        self.embeddings = []
        self.max_size = max_size

    def add(self, content: str, embedding: np.ndarray):
        if len(self.entries) >= self.max_size:
            self.entries.pop(0)
            self.embeddings.pop(0)
        self.entries.append(content)
        self.embeddings.append(embedding)

    def retrieve(self, query: str, k: int) -> List[str]:
        query_emb = self.retriever.encode(query)
        similarities = [cosine_similarity(query_emb, e) for e in self.embeddings]
        top_k = np.argsort(similarities)[-k:][::-1]
        return [self.entries[i] for i in top_k]
```

---

### 2.6 P5: Structured Workspace

**Purpose:** Non-NL structured state for coordination.

**Implementation:**
```python
class StructuredWorkspaceProtocol:
    def __init__(self, schema: Dict):
        self.schema = schema
        self.workspace = {}

    def update_workspace(self, updates: Dict) -> Dict:
        """Apply updates to workspace"""
        self._validate_schema(updates)
        self.workspace = self._merge(self.workspace, updates)
        return self.workspace

    def get_workspace(self) -> Dict:
        return copy.deepcopy(self.workspace)

    def workspace_to_message(self) -> str:
        """Serialize for transmission"""
        return json.dumps(self.workspace, separators=(',', ':'))

    def message_to_workspace(self, message: str) -> Dict:
        """Deserialize received workspace"""
        return json.loads(message)
```

**Schema Example (Constraint Satisfaction):**
```python
CONSTRAINT_SCHEMA = {
    "type": "object",
    "properties": {
        "known_values": {
            "type": "object",
            "additionalProperties": {"type": ["integer", "null"]}
        },
        "constraints_checked": {
            "type": "array",
            "items": {"type": "string"}
        },
        "current_hypothesis": {
            "type": "object",
            "additionalProperties": {"type": "integer"}
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1}
    }
}
```

**Schema Example (Code Task):**
```python
CODE_SCHEMA = {
    "type": "object",
    "properties": {
        "proposed_changes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "file": {"type": "string"},
                    "line": {"type": "integer"},
                    "old": {"type": "string"},
                    "new": {"type": "string"}
                }
            }
        },
        "test_results": {
            "type": "object",
            "additionalProperties": {"type": "boolean"}
        },
        "error_messages": {
            "type": "array",
            "items": {"type": "string"}
        }
    }
}
```

**System Prompt:**
```
You are Agent {A/B} communicating via a structured workspace. Update the
workspace JSON with your findings. Do not use natural language in the
workspace—use only the defined schema fields.

Schema: {schema}
Current workspace: {workspace}

Your task: {task_description}
```

---

## 3. Latent Baselines

### 3.1 E0: CIPHER Expected Embedding

**Purpose:** Minimal latent baseline with low engineering risk.

**Implementation:**
```python
class CIPHERProtocol:
    def __init__(self, model, top_k: int = 100, temperature: float = 1.0):
        self.model = model
        self.top_k = top_k
        self.temperature = temperature
        self.embed_matrix = model.get_input_embeddings().weight

    def encode_message(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Generate expected embedding message"""
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[:, -1, :]  # Last token logits

            # Apply temperature and top-k filtering
            logits = logits / self.temperature
            top_logits, top_indices = logits.topk(self.top_k, dim=-1)

            # Softmax over top-k
            probs = F.softmax(top_logits, dim=-1)

            # Compute expected embedding
            top_embeddings = self.embed_matrix[top_indices]  # (batch, k, d)
            expected_emb = (probs.unsqueeze(-1) * top_embeddings).sum(dim=1)

            return expected_emb  # (batch, d_model)

    def inject_message(self, receiver_input_ids: torch.Tensor,
                      message_embedding: torch.Tensor) -> torch.Tensor:
        """Inject expected embedding as soft token prefix"""
        input_embeds = self.model.get_input_embeddings()(receiver_input_ids)

        # Prepend message embedding
        augmented_embeds = torch.cat([
            message_embedding.unsqueeze(1),  # (batch, 1, d)
            input_embeds
        ], dim=1)

        return augmented_embeds

    def bits_per_message(self, dtype: torch.dtype = torch.float16) -> int:
        d_model = self.embed_matrix.shape[1]
        bits_per_element = 16 if dtype == torch.float16 else 32
        return d_model * bits_per_element
```

**Hyperparameters:**
| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| top_k | 100 | [10, 500] | Higher = more information, noisier |
| temperature | 1.0 | [0.5, 2.0] | Lower = sharper distribution |

---

### 3.2 A0: Activation Grafting

**Purpose:** Strong latent baseline demonstrating activation-level communication.

**Implementation:**
```python
class ActivationGraftingProtocol:
    def __init__(self, model, layer_idx: int, combine_fn: str = 'average'):
        self.model = model
        self.layer_idx = layer_idx
        self.combine_fn = self._get_combine_fn(combine_fn)
        self.captured_activation = None

    def _get_combine_fn(self, name: str):
        fns = {
            'replace': lambda b, a: a,
            'add': lambda b, a: b + a,
            'average': lambda b, a: (b + a) / 2,
            'weighted_0.3': lambda b, a: 0.7 * b + 0.3 * a,
            'weighted_0.5': lambda b, a: 0.5 * b + 0.5 * a,
            'weighted_0.7': lambda b, a: 0.3 * b + 0.7 * a,
        }
        return fns[name]

    def capture_sender_activation(self, input_ids: torch.Tensor):
        """Run sender forward pass, capture activation"""
        def hook(module, input, output):
            self.captured_activation = output[0].clone()

        handle = self.model.model.layers[self.layer_idx].register_forward_hook(hook)
        with torch.no_grad():
            self.model(input_ids)
        handle.remove()

        return self.captured_activation

    def inject_to_receiver(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run receiver with grafted activation"""
        def hook(module, input, output):
            if self.captured_activation is not None:
                # Handle sequence length mismatch
                sender_act = self.captured_activation
                receiver_act = output[0]

                # Align sequence lengths (use last sender_len positions)
                if sender_act.shape[1] != receiver_act.shape[1]:
                    min_len = min(sender_act.shape[1], receiver_act.shape[1])
                    sender_act = sender_act[:, -min_len:, :]
                    receiver_act_aligned = receiver_act[:, -min_len:, :]
                    combined = self.combine_fn(receiver_act_aligned, sender_act)
                    # Replace aligned portion
                    new_output = receiver_act.clone()
                    new_output[:, -min_len:, :] = combined
                else:
                    new_output = self.combine_fn(receiver_act, sender_act)

                return (new_output,) + output[1:]
            return output

        handle = self.model.model.layers[self.layer_idx].register_forward_hook(hook)
        with torch.no_grad():
            outputs = self.model(input_ids)
        handle.remove()

        return outputs

    def bits_per_message(self, seq_len: int,
                        dtype: torch.dtype = torch.float16) -> int:
        d_model = self.model.config.hidden_size
        bits_per_element = 16 if dtype == torch.float16 else 32
        return seq_len * d_model * bits_per_element
```

**Layer Selection Heuristics:**
| Model Size | Recommended Layer | Rationale |
|------------|-------------------|-----------|
| < 3B | n/2 | Middle layers most compositional |
| 3-8B | n/2 to 2n/3 | Later layers for complex reasoning |
| > 8B | Ablation required | Model-specific |

**Combination Function Selection:**
| Function | When to Use |
|----------|-------------|
| replace | When sender has critical info receiver lacks |
| average | General purpose, stable |
| weighted_0.7 | When sender info is more important |
| add | When info is complementary (can be unstable) |

---

## 4. Baseline Optimization Protocol

### 4.1 Hyperparameter Tuning

**Procedure:**
1. Hold out 20% of tasks for final evaluation
2. Use 10% for validation during tuning
3. Grid search over hyperparameter space
4. Select based on validation performance
5. Report final results on held-out set

**Tuning Budget:**
- Text baselines: 50 trials per protocol
- Latent baselines: 100 trials per protocol (more sensitive)

### 4.2 Tuning Ranges

**P3 (Summarization):**
| Parameter | Range |
|-----------|-------|
| summarization_prompt | 3 variants |
| temperature | [0.3, 0.5, 0.7] |

**P4 (Retrieval):**
| Parameter | Range |
|-----------|-------|
| retrieval_k | [1, 3, 5, 7] |
| memory_size | [50, 100, 200] |
| embedding_model | [same, e5-small, e5-base] |

**E0 (CIPHER):**
| Parameter | Range |
|-----------|-------|
| top_k | [10, 50, 100, 200, 500] |
| temperature | [0.5, 0.7, 1.0, 1.5] |

**A0 (Activation):**
| Parameter | Range |
|-----------|-------|
| layer_idx | [n/4, n/3, n/2, 2n/3, 3n/4] |
| combine_fn | [replace, average, weighted_0.5, weighted_0.7] |

---

## 5. Fairness Considerations

### 5.1 Compute Matching

All protocols must be compared at matched compute budgets:

```python
def compute_budget(protocol: str, episode: Episode, config: Dict) -> float:
    """Estimate compute in FLOPs"""
    n_params = config['model_params']

    if protocol in ['P0', 'P1', 'P2', 'P3', 'P4', 'P5']:
        # Standard inference
        total_tokens = sum(t.input_tokens + t.output_tokens for t in episode.turns)
        return 2 * n_params * total_tokens

    elif protocol == 'P3':
        # Add summarization cost
        base = compute_budget('P1', episode, config)
        summarizer_cost = config.get('summarizer_params', n_params) * episode.summary_tokens
        return base + 2 * summarizer_cost

    elif protocol == 'E0':
        # CIPHER: extra softmax + matmul per message
        base = compute_budget('P1', episode, config)
        vocab_size = config['vocab_size']
        d_model = config['d_model']
        n_messages = len([t for t in episode.turns if t.message_bits > 0])
        cipher_cost = n_messages * (vocab_size * d_model)  # Expected embedding
        return base + cipher_cost

    elif protocol == 'A0':
        # Activation grafting: two forward passes
        tokens_A = sum(t.input_tokens for t in episode.turns if t.agent == 'A')
        tokens_B = sum(t.input_tokens for t in episode.turns if t.agent == 'B')
        return 2 * n_params * (tokens_A + tokens_B)

    elif protocol in ['L1', 'L2']:
        # Codec protocols: base + codec
        base = compute_budget('A0', episode, config)
        codec_params = config['codec_params']
        n_messages = len([t for t in episode.turns if t.message_bits > 0])
        codec_cost = 2 * codec_params * config['k_vectors'] * config['d_model']
        return base + n_messages * codec_cost
```

### 5.2 Information Matching

For rate-distortion analysis, match on bits transmitted:

| Bits | P2 Config | L2 Config |
|------|-----------|-----------|
| 128 | 16 bytes | k=4, K=256 |
| 512 | 64 bytes | k=8, K=256 |
| 2048 | 256 bytes | k=16, K=256 |

### 5.3 Latency Matching

For latency-sensitive comparisons:
- Measure wall-clock time
- Report median and P95
- Account for serialization overhead

---

## 6. Baseline Validation Checklist

Before proceeding to novel method evaluation:

- [ ] P0 << P1 (communication is necessary)
- [ ] P1 performance stable across seeds (CV < 0.15)
- [ ] P3 ≥ P2 at same budget (summarization helps)
- [ ] P4 ≥ P2 at same budget (retrieval helps)
- [ ] P5 ≥ P2 at same budget (structure helps)
- [ ] Budget accounting verified (manual spot check)
- [ ] Logging complete (replay possible)

---

## 7. Expected Baseline Performance

**Preliminary estimates (to be validated in M0):**

| Protocol | Task S (SR) | Task C (SR) | Notes |
|----------|-------------|-------------|-------|
| P0 | 0.15-0.25 | 0.05-0.15 | Random baseline |
| P1 | 0.65-0.80 | 0.45-0.60 | Upper text reference |
| P2 (512 bits) | 0.40-0.55 | 0.25-0.40 | Significant degradation |
| P3 (512 bits) | 0.50-0.65 | 0.35-0.50 | Summarization helps |
| P4 (512 bits) | 0.55-0.70 | 0.40-0.55 | Retrieval helps more |
| P5 | 0.60-0.75 | 0.40-0.55 | Structure helps |

These estimates inform our hypothesis thresholds. If P1 is significantly higher/lower than expected, recalibrate expectations before proceeding.

---

*Document Status: Pre-registered. Baseline implementations frozen before comparison with novel methods.*
