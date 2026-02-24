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
pip install "modal>=1.1.0"
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
│   ├── train_m2_codec.py   # Main M2-SCALE training/eval script
│   ├── upload_data.py      # Upload dataset to Modal volume
│   └── train_codec.py      # Deprecated (kept for history)
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

## Migration Guide 2025.06 Notes

LPCA now assumes Modal's 2025.06+ image behavior:

1. Modal client dependencies are no longer installed into base image layers.
2. Dependencies installed in your image recipe take precedence over Modal runtime fallbacks.
3. For external base images (`from_registry` / `from_dockerfile`), Modal no longer forces `pip/uv/wheel` upgrades.

What this means for this repo:

- Current scripts use `modal.Image.debian_slim(...).pip_install(...)`, so no action is needed for base-image dependency conflicts.
- If you switch to external images later, explicitly ensure your recipe has compatible packaging tooling (`pip` and optionally `uv`) and any required Python runtime.
- If you start using `modal.Sandbox.create`, explicitly pass a long-lived command (for example `sleep 48h`) when needed, since newer images use the image `CMD` by default.

## Running Experiments

### Development (local)
```bash
# Local prototyping (MPS/CPU/GPU)
python modal/train_m2_codec.py --local --k 4 --epochs 2
```

### Production (GPU)
```bash
# Run M2-SCALE on A100
modal run modal/train_m2_codec.py --k 4 --epochs 10
```

### Sweep
```bash
# Run multiple k values
python modal/train_m2_codec.py --sweep --local
# or on Modal (run each k explicitly)
modal run modal/train_m2_codec.py --k 4 --epochs 10
modal run modal/train_m2_codec.py --k 8 --epochs 10
modal run modal/train_m2_codec.py --k 16 --epochs 10
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
