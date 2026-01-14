# LPCA Reproducibility Documentation

**Version:** 1.0
**Status:** Pre-registered
**Aligned With:** NeurIPS 2024 Reproducibility Checklist

---

## 1. NeurIPS Reproducibility Checklist Compliance

### 1.1 Claims and Evidence

| Claim | Evidence Type | Location |
|-------|---------------|----------|
| Latent comm improves capability | Experimental | EXPERIMENTS.md §E4, E5 |
| Discrete codec preserves capability | Experimental | EXPERIMENTS.md §E6 |
| Rate-distortion frontier exists | Experimental | METRICS.md §6.1 |
| Safety indicators are measurable | Experimental | SAFETY_PROTOCOL.md §3-6 |

### 1.2 Theoretical Claims

No unproven theoretical claims are made. All claims are empirical and testable.

### 1.3 Experimental Claims

All experiments will include:
- Central tendency (mean) AND variation (std, CI)
- Statistical significance tests where appropriate
- Effect sizes for all comparisons
- Multiple random seeds (minimum 5)

---

## 2. Code and Data Availability

### 2.1 Code Release Plan

| Component | Release | License |
|-----------|---------|---------|
| Evaluation harness | Full code | Apache 2.0 |
| Task generators | Full code | Apache 2.0 |
| Protocol implementations | Full code | Apache 2.0 |
| Codec architectures | Full code | Apache 2.0 |
| Training scripts | Full code | Apache 2.0 |
| Analysis scripts | Full code | Apache 2.0 |
| Trained codec weights | Released | Apache 2.0 |
| Evaluation logs | Sample released | CC-BY 4.0 |

### 2.2 Repository Structure

```
lpca/
├── README.md                    # Quick start guide
├── LICENSE                      # Apache 2.0
├── requirements.txt             # Pinned dependencies
├── setup.py                     # Package installation
│
├── configs/                     # Experiment configurations
│   ├── base.yaml
│   ├── experiments/
│   └── sweeps/
│
├── lpca/                        # Main package
│   ├── __init__.py
│   ├── core/
│   │   ├── config.py
│   │   ├── logging.py
│   │   ├── metrics.py
│   │   └── budget.py
│   ├── envs/
│   │   ├── base.py
│   │   ├── split_synthetic.py
│   │   └── split_code.py
│   ├── agents/
│   │   ├── base.py
│   │   ├── model_wrapper.py
│   │   └── policy.py
│   ├── channels/
│   │   ├── base.py
│   │   ├── text.py
│   │   ├── cipher.py
│   │   ├── activation.py
│   │   └── codec.py
│   ├── training/
│   │   ├── distill.py
│   │   ├── vq_train.py
│   │   └── evaluate.py
│   └── safety/
│       ├── monitors.py
│       ├── compliance.py
│       ├── covert_probe.py
│       └── bloom_eval.py
│
├── scripts/
│   ├── run_experiment.py
│   ├── run_sweep.py
│   ├── analyze_results.py
│   └── generate_figures.py
│
├── tests/                       # Unit tests
│   ├── test_envs.py
│   ├── test_channels.py
│   └── test_metrics.py
│
├── notebooks/                   # Analysis notebooks
│   └── results_analysis.ipynb
│
└── docs/                        # Additional documentation
    ├── PLAN.md
    ├── EXPERIMENTS.md
    ├── METRICS.md
    ├── BASELINES.md
    ├── SAFETY_PROTOCOL.md
    └── REPRODUCIBILITY.md
```

### 2.3 Data Release

| Data Type | Size Est. | Format | Release |
|-----------|-----------|--------|---------|
| Task instances | ~100MB | JSONL | Full |
| Episode logs | ~10GB | Parquet | Sample (1%) |
| Activation snapshots | ~500GB | NPY | Not released (regenerable) |
| Trained codecs | ~1GB | PyTorch | Full |
| Evaluation results | ~100MB | Parquet | Full |

---

## 3. Environment Specification

### 3.1 Hardware Requirements

**Minimum (for reproduction):**
- GPU: Any CUDA-capable or Apple Silicon
- RAM: 32GB system + 16GB VRAM (or 64GB unified)
- Storage: 100GB free

**Recommended (for full experiments):**
- GPU: Apple M3 Max (98GB unified) or NVIDIA A100 (80GB)
- RAM: 64GB+ system
- Storage: 1TB SSD

**Used in Paper:**
```yaml
primary_hardware:
  machine: MacBook Pro (Mac15,11)
  chip: Apple M3 Max
  memory: 98GB unified
  storage: 1TB SSD
  os: macOS Sonoma 14.x

cloud_hardware:  # For scaling experiments only
  instance: AWS p4d.24xlarge
  gpu: 8x NVIDIA A100 80GB
  memory: 1152GB
  storage: 8TB NVMe
```

### 3.2 Software Requirements

**Python Version:** 3.11.x or 3.12.x

**Core Dependencies:**
```
# requirements.txt
torch>=2.2.0
transformers>=4.38.0
accelerate>=0.27.0
datasets>=2.18.0
safetensors>=0.4.0
numpy>=1.26.0
scipy>=1.12.0
pandas>=2.2.0
pyarrow>=15.0.0
einops>=0.7.0
pydantic>=2.6.0
hydra-core>=1.3.0
tqdm>=4.66.0
rich>=13.7.0
matplotlib>=3.8.0
seaborn>=0.13.0
scikit-learn>=1.4.0
pytest>=8.0.0
```

**Exact Versions (for perfect reproduction):**
```
# requirements-lock.txt
# Generated at experiment time, included in release
```

### 3.3 Environment Setup

```bash
# Clone repository
git clone https://github.com/[org]/lpca.git
cd lpca

# Create environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .

# Verify installation
python -c "import lpca; print(lpca.__version__)"
pytest tests/ -v
```

### 3.4 Environment Variables

```bash
# Required for MPS (Apple Silicon)
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false

# Optional: Weights & Biases logging
export WANDB_PROJECT=lpca
export WANDB_ENTITY=your-entity

# Optional: HuggingFace cache
export HF_HOME=/path/to/cache
```

---

## 4. Model Specifications

### 4.1 Models Used

| Model | Parameters | Source | Access |
|-------|------------|--------|--------|
| Llama-3.2-1B-Instruct | 1.24B | HuggingFace | Open weights |
| Llama-3.2-3B-Instruct | 3.21B | HuggingFace | Open weights |
| Llama-3.1-8B-Instruct | 8.03B | HuggingFace | Open weights |
| Qwen2.5-1.5B-Instruct | 1.54B | HuggingFace | Open weights |
| Qwen2.5-7B-Instruct | 7.62B | HuggingFace | Open weights |

### 4.2 Model Loading

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_CONFIGS = {
    'llama-1b': {
        'name': 'meta-llama/Llama-3.2-1B-Instruct',
        'dtype': torch.float16,
        'device_map': 'auto',
    },
    'llama-3b': {
        'name': 'meta-llama/Llama-3.2-3B-Instruct',
        'dtype': torch.float16,
        'device_map': 'auto',
    },
    'llama-8b': {
        'name': 'meta-llama/Llama-3.1-8B-Instruct',
        'dtype': torch.float16,
        'device_map': 'auto',
    },
}

def load_model(config_name: str):
    config = MODEL_CONFIGS[config_name]
    model = AutoModelForCausalLM.from_pretrained(
        config['name'],
        torch_dtype=config['dtype'],
        device_map=config['device_map'],
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(config['name'])
    return model, tokenizer
```

---

## 5. Experiment Reproduction

### 5.1 Quick Reproduction (Subset)

```bash
# Run baseline validation (E1) - ~4 hours
python scripts/run_experiment.py --config configs/experiments/e1_baseline.yaml

# Run single protocol comparison - ~1 hour
python scripts/run_experiment.py \
    --config configs/experiments/quick_test.yaml \
    --protocols P1,A0 \
    --n_episodes 50
```

### 5.2 Full Reproduction

```bash
# Full experimental suite - ~2 weeks on M3 Max
# Or ~3 days on 4x A100

# Step 1: Baselines (E1-E2)
python scripts/run_sweep.py --config configs/sweeps/baselines.yaml

# Step 2: Latent baselines (E3-E4)
python scripts/run_sweep.py --config configs/sweeps/latent_baselines.yaml

# Step 3: Codec training (E5)
python scripts/run_sweep.py --config configs/sweeps/codec_training.yaml

# Step 4: Discrete codec (E6)
python scripts/run_sweep.py --config configs/sweeps/discrete_codec.yaml

# Step 5: Safety evaluation
python scripts/run_sweep.py --config configs/sweeps/safety_eval.yaml

# Step 6: Generate figures and tables
python scripts/analyze_results.py --output_dir results/
python scripts/generate_figures.py --input_dir results/ --output_dir figures/
```

### 5.3 Expected Results

We will release expected result ranges for validation:

```yaml
# expected_results.yaml
e1_baseline_validation:
  P0_synthetic_sr: [0.15, 0.25]
  P1_synthetic_sr: [0.65, 0.80]
  gap_threshold: 0.30

e4_activation_grafting:
  best_layer: ["n/2", "2n/3"]
  best_combine: ["average", "weighted_0.7"]
  improvement_over_P1: [0.10, 0.30]

e6_discrete_codec:
  retention_at_10x: [0.60, 0.90]
  pareto_optimal_bits: [128, 512]
```

---

## 6. Hyperparameter Documentation

### 6.1 Fixed Hyperparameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Temperature | 0.7 | Standard for instruction-following |
| Top-p | 0.9 | Nucleus sampling default |
| Max tokens | 2048 | Sufficient for all tasks |
| Context window | 4096 | Model context limit |

### 6.2 Tuned Hyperparameters

All tuned hyperparameters will be documented with:
- Search space
- Selection criterion
- Final selected value
- Sensitivity analysis

```yaml
# hyperparameters.yaml
activation_grafting:
  layer_idx:
    search_space: [n/4, n/3, n/2, 2n/3, 3n/4]
    selection_criterion: max_success_rate
    selected: n/2
    sensitivity: "±10% performance within [n/3, 2n/3]"

  combine_fn:
    search_space: [replace, add, average, weighted]
    selection_criterion: max_success_rate
    selected: average
    sensitivity: "weighted_0.7 within 2%"

codec_training:
  k_vectors:
    search_space: [4, 8, 16, 32, 64]
    selection_criterion: pareto_optimal
    selected: 16
    sensitivity: "see rate-distortion curve"

  learning_rate:
    search_space: [1e-4, 3e-4, 1e-3]
    selection_criterion: validation_loss
    selected: 3e-4
```

---

## 7. Statistical Reporting

### 7.1 Reporting Standards

All results will include:

1. **Point estimates:** Mean or median as appropriate
2. **Uncertainty:** 95% confidence intervals
3. **Sample size:** Number of episodes/seeds
4. **Significance tests:** p-values where claimed
5. **Effect sizes:** Cohen's d for comparisons

### 7.2 Multiple Comparisons

When making multiple comparisons:
- Pre-specified comparisons: Bonferroni correction
- Exploratory analysis: Benjamini-Hochberg FDR
- Clearly label which is which

### 7.3 Negative Results

We commit to reporting:
- Experiments that didn't work
- Hypotheses that were falsified
- Approaches that were abandoned and why

---

## 8. Logging and Provenance

### 8.1 Experiment Logging

Every experiment logs:

```python
@dataclass
class ExperimentLog:
    # Identification
    experiment_id: str
    git_commit: str
    timestamp: str

    # Configuration
    config: Dict
    hyperparameters: Dict
    random_seeds: List[int]

    # Environment
    hardware: Dict
    software_versions: Dict

    # Results
    metrics: Dict
    artifacts: List[str]

    # Reproducibility
    command: str  # Exact command to reproduce
    duration_seconds: float
```

### 8.2 Artifact Tracking

```python
class ArtifactTracker:
    def __init__(self, experiment_id: str, output_dir: str):
        self.experiment_id = experiment_id
        self.output_dir = Path(output_dir) / experiment_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = []

    def save_artifact(self, name: str, data: Any, format: str = 'auto'):
        path = self.output_dir / name
        if format == 'json' or (format == 'auto' and isinstance(data, dict)):
            with open(path.with_suffix('.json'), 'w') as f:
                json.dump(data, f, indent=2)
        elif format == 'parquet':
            data.to_parquet(path.with_suffix('.parquet'))
        elif format == 'torch':
            torch.save(data, path.with_suffix('.pt'))

        self.manifest.append({
            'name': name,
            'path': str(path),
            'hash': compute_hash(path),
            'timestamp': datetime.now().isoformat(),
        })

    def save_manifest(self):
        manifest_path = self.output_dir / 'manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=2)
```

---

## 9. Known Limitations

### 9.1 Reproducibility Challenges

| Challenge | Mitigation |
|-----------|------------|
| Non-deterministic GPU operations | Document variance, use multiple seeds |
| Model version updates | Pin exact model commits |
| Hardware differences | Report hardware-specific results |
| Floating point precision | Use deterministic algorithms where possible |

### 9.2 What May Differ

Even with careful reproduction, expect:
- ±5% variance in success rates (sampling)
- ±10% variance in training curves (optimization)
- ±20% variance in wall-clock times (hardware)

### 9.3 What Should Not Differ

- Qualitative conclusions
- Ranking of protocols
- Direction of effects
- Stop condition triggers

---

## 10. Contact and Support

### 10.1 Bug Reports

File issues at: `https://github.com/[org]/lpca/issues`

Include:
- Exact error message
- Environment details (`pip freeze`)
- Reproduction steps
- Expected vs actual behavior

### 10.2 Questions

For questions about:
- **Reproduction:** Open GitHub issue with `[reproduction]` tag
- **Methodology:** Email corresponding author
- **Collaboration:** Email corresponding author

---

## Appendix: NeurIPS Checklist Mapping

| Checklist Item | Status | Evidence |
|----------------|--------|----------|
| Claims supported by evidence | Yes | EXPERIMENTS.md |
| Theoretical claims have proofs | N/A | No theoretical claims |
| Experimental claims include variance | Yes | METRICS.md §7 |
| Datasets described | Yes | EXPERIMENTS.md §1 |
| Code available | Yes | §2.1 |
| Training details provided | Yes | EXPERIMENTS.md §E5-E6 |
| Error bars/CIs reported | Yes | METRICS.md §7.3 |
| Compute requirements stated | Yes | §3.1, PLAN.md §9 |
| Hyperparameters documented | Yes | §6 |
| Best hyperparameters justified | Yes | §6.2 |
| Random seeds documented | Yes | EXPERIMENTS.md §4 |

---

*Document Status: Pre-registered. Will be updated with actual values and links upon experiment completion.*
