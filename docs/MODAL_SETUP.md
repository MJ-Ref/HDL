# Modal Setup for M2 Experiments

This guide covers setting up Modal.com for M2 codec training experiments.

## Why Modal?

- **$30/month free credits** - covers ~14 hours of A100 40GB
- **$10,000 academic research grants** - apply at modal.com/research
- **Serverless** - only pay for compute, no idle costs
- **Python-native** - no YAML, integrates with existing code
- **Easy GPU access** - A100 40GB at $2.10/hr

## Quick Start

### 1. Install Modal

```bash
pip install modal
```

### 2. Create Account & Authenticate

```bash
# Opens browser for account creation/login
modal setup
```

### 3. Verify Setup

```bash
modal run --help
```

## M2 Training Configuration

### GPU Options for Codec Training

| GPU | VRAM | $/hr | Use Case |
|-----|------|------|----------|
| A100 40GB | 40GB | $2.10 | Primary training |
| A100 80GB | 80GB | $2.50 | Large batch sizes |
| A10G | 24GB | $0.50 | Development/debugging |
| H100 | 80GB | $3.60 | Fastest training |

### Estimated Costs

| Task | GPU-Hours | Cost | Notes |
|------|-----------|------|-------|
| Data collection (P1 episodes) | 5-10 | $10-20 | Can run locally |
| Encoder-decoder training | 20-40 | $40-80 | Main expense |
| Evaluation sweeps | 10-20 | $20-40 | Multiple k values |
| **Total** | **35-70** | **$70-140** | Within free tier + small top-up |

## Project Structure for Modal

```
HDL/
├── modal/
│   ├── train_codec.py      # Main training script
│   ├── collect_data.py     # P1 episode collection
│   └── evaluate.py         # Evaluation sweeps
└── ...
```

## Example: Training Script

```python
import modal

app = modal.App("lpca-codec-training")

# Define container image with dependencies
image = modal.Image.debian_slim().pip_install([
    "torch",
    "transformers",
    "numpy",
    "tqdm",
])

@app.function(
    gpu="a100-40gb",
    image=image,
    timeout=3600,  # 1 hour max
)
def train_codec(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    k_vectors: int = 16,
    epochs: int = 10,
):
    """Train continuous codec on P1 episodes."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Training code here...
    pass

@app.local_entrypoint()
def main():
    train_codec.remote(k_vectors=16, epochs=10)
```

## Running Experiments

### Development (local)
```bash
# Test locally first (uses CPU)
modal run modal/train_codec.py --local
```

### Production (GPU)
```bash
# Run on A100
modal run modal/train_codec.py
```

### Sweep
```bash
# Run multiple k values in parallel
modal run modal/train_codec.py --k-values 4 8 16 32
```

## Monitoring

```bash
# View logs
modal app logs lpca-codec-training

# Check usage
modal token stats
```

## Budget Management

1. Set up billing alerts at modal.com/settings
2. Use `timeout` parameter to prevent runaway costs
3. Start with A10G ($0.50/hr) for debugging
4. Scale to A100 for final training

## Research Grant Application

If costs exceed $30/month free tier:

1. Go to modal.com/research
2. Apply for up to $10,000 in credits
3. Describe LPCA research project
4. Expected approval: 1-2 weeks

## Troubleshooting

### GPU not available
Modal pools GPUs across providers. If A100 unavailable, try:
- A100 80GB (slightly more expensive)
- H100 (faster but pricier)
- Wait 5-10 minutes for availability

### Out of memory
- Reduce batch size
- Use gradient checkpointing
- Switch to A100 80GB

### Timeout
- Increase timeout parameter
- Split training into checkpointed stages
