#!/usr/bin/env python3
"""
M2-SCALE: Codec Training on Modal (Cloud GPUs)

This script trains the continuous codec to replace sender text messages
with learned latent vectors (Option A from PLAN.md).

Gates (pre-committed):
- Gate 1 (Sanity): L1 at k={4,8} must beat P0 by ≥10pp (threshold: 30%)
- Gate 2 (Retention): L1 at k=16 must retain ≥50% of P1 (threshold: 34%)

Ablations (required with every evaluation):
- Null message: Should recover ~P0 (20%)
- Random latent: Should recover ~P0 (20%)
- Shuffle messages: Should crater (<P0)

Usage (local testing):
    python modal/train_m2_codec.py --local --k 4 --epochs 2

Usage (Modal cloud):
    modal run modal/train_m2_codec.py --k 4 --epochs 10
    modal run modal/train_m2_codec.py --sweep  # Full k sweep
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import random

# Check if running on Modal
MODAL_AVAILABLE = False
try:
    import modal
    MODAL_AVAILABLE = True
except ImportError:
    pass


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class M2Config:
    """M2-SCALE training configuration."""
    # Model
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"

    # Codec architecture
    k_vectors: int = 16
    hidden_dim: int = 512

    # Training
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-3
    warmup_steps: int = 100

    # Data
    train_data_path: str = "data/m2_train.jsonl"
    val_seeds: Tuple[int, int] = (1000, 1100)
    test_seeds: Tuple[int, int] = (2000, 2100)

    # Evaluation
    eval_episodes: int = 50
    temperature: float = 0.3

    # Gates (FROZEN from m2_frozen_settings.yaml)
    p0_baseline: float = 0.20
    p1_baseline: float = 0.68
    gate1_threshold: float = 0.30  # P0 + 10pp
    gate2_threshold: float = 0.34  # 50% of P1


# ============================================================================
# Codec Architecture
# ============================================================================

def create_codec_model(d_model: int, k_vectors: int, hidden_dim: int = 512):
    """Create encoder-decoder codec for latent communication."""
    import torch
    import torch.nn as nn

    class LatentCodec(nn.Module):
        """
        Encoder-decoder for latent communication (Option A).

        Encoder: text embedding -> k latent vectors
        Decoder: k vectors -> prefix embeddings for receiver
        """

        def __init__(self, d_model: int, k_vectors: int, hidden_dim: int):
            super().__init__()
            self.k = k_vectors
            self.d = d_model

            # Encoder: pooled text embedding -> k latent vectors
            self.encoder = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, k_vectors * d_model),
            )

            # Decoder: k vectors -> single prefix embedding
            self.decoder = nn.Sequential(
                nn.Linear(k_vectors * d_model, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, d_model),
            )

        def encode(self, text_embedding: 'torch.Tensor') -> 'torch.Tensor':
            """Encode text embedding to k latent vectors."""
            # text_embedding: (batch, d_model)
            encoded = self.encoder(text_embedding)  # (batch, k * d_model)
            return encoded.view(-1, self.k, self.d)  # (batch, k, d_model)

        def decode(self, latent: 'torch.Tensor') -> 'torch.Tensor':
            """Decode k latent vectors to prefix embedding."""
            # latent: (batch, k, d_model)
            flat = latent.view(-1, self.k * self.d)  # (batch, k * d_model)
            return self.decoder(flat)  # (batch, d_model)

        def forward(self, text_embedding: 'torch.Tensor') -> Tuple['torch.Tensor', 'torch.Tensor']:
            """Full encode-decode pass."""
            latent = self.encode(text_embedding)
            reconstructed = self.decode(latent)
            return latent, reconstructed

    return LatentCodec(d_model, k_vectors, hidden_dim)


# ============================================================================
# Training Loop
# ============================================================================

def train_codec(
    config: M2Config,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Train the codec on M2 dataset.

    Returns training history and final checkpoint path.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm import tqdm

    print("=" * 60)
    print(f"M2-SCALE Training: k={config.k_vectors}")
    print("=" * 60)

    # Load tokenizer and model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device != "cuda":
        model = model.to(device)
    model.eval()

    # Freeze base model
    for param in model.parameters():
        param.requires_grad = False

    d_model = model.config.hidden_size
    print(f"Model loaded: d_model={d_model}")

    # Create codec
    codec = create_codec_model(d_model, config.k_vectors, config.hidden_dim)
    codec = codec.to(device)
    n_params = sum(p.numel() for p in codec.parameters())
    print(f"Codec parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    # Load training data
    print(f"Loading training data from {config.train_data_path}...")

    class M2Dataset(Dataset):
        def __init__(self, data_path: str):
            self.samples = []
            with open(data_path, 'r') as f:
                for line in f:
                    self.samples.append(json.loads(line))
            print(f"  Loaded {len(self.samples)} samples")

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return self.samples[idx]

    dataset = M2Dataset(config.train_data_path)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda x: x,  # Return list of dicts
    )

    # Optimizer
    optimizer = torch.optim.AdamW(codec.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs)

    # Training loop
    history = {'loss': [], 'epoch': []}

    print(f"\nTraining for {config.epochs} epochs...")
    for epoch in range(config.epochs):
        codec.train()
        epoch_loss = 0
        n_batches = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.epochs}"):
            # Get text embeddings for sender messages
            messages = [s['sender_message'] for s in batch]
            inputs = tokenizer(
                messages,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                # Use last hidden state, mean pooled
                hidden = outputs.hidden_states[-1]  # (batch, seq, d_model)
                # Mask padding
                mask = inputs['attention_mask'].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)  # (batch, d_model)

            # Forward through codec
            latent, reconstructed = codec(pooled)

            # Reconstruction loss
            loss = F.mse_loss(reconstructed, pooled)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        history['loss'].append(avg_loss)
        history['epoch'].append(epoch + 1)

        print(f"  Epoch {epoch+1}: loss={avg_loss:.6f}")

    # Save checkpoint
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_path = f"checkpoints/m2_codec_k{config.k_vectors}_{timestamp}.pt"
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        'k_vectors': config.k_vectors,
        'hidden_dim': config.hidden_dim,
        'd_model': d_model,
        'state_dict': codec.state_dict(),
        'config': asdict(config),
        'history': history,
    }, checkpoint_path)

    print(f"\nCheckpoint saved: {checkpoint_path}")

    return {
        'checkpoint_path': checkpoint_path,
        'history': history,
        'final_loss': history['loss'][-1],
    }


# ============================================================================
# Evaluation with Ablations
# ============================================================================

def evaluate_codec(
    checkpoint_path: str,
    config: M2Config,
    device: str = "cuda",
    run_ablations: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate trained codec with required ablations.

    Ablations:
    1. Normal (trained codec)
    2. Null message (no latent)
    3. Random latent
    4. Shuffled messages (if requested)
    """
    import torch

    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    k_vectors = checkpoint['k_vectors']
    d_model = checkpoint['d_model']

    print(f"Loaded: k={k_vectors}, d_model={d_model}")

    results = {}

    # TODO: Implement actual evaluation loop
    # For now, placeholder
    print("\n[Evaluation not yet implemented - requires full inference loop]")
    print("Will evaluate on fresh test seeds (2000-2099)")

    results['normal'] = {'success_rate': 0.0, 'n_episodes': 0}

    if run_ablations:
        print("\nRunning ablations...")
        results['null_message'] = {'success_rate': 0.0, 'expected': config.p0_baseline}
        results['random_latent'] = {'success_rate': 0.0, 'expected': config.p0_baseline}

    return results


def check_gates(results: Dict[str, Any], config: M2Config) -> Dict[str, bool]:
    """Check if training passes the pre-committed gates."""
    gates = {}

    normal_sr = results.get('normal', {}).get('success_rate', 0)

    # Gate 1: Sanity (L1 > P0 + 10pp)
    gates['gate1_sanity'] = normal_sr >= config.gate1_threshold
    print(f"\nGate 1 (Sanity): {normal_sr:.1%} >= {config.gate1_threshold:.1%}? "
          f"{'PASS' if gates['gate1_sanity'] else 'FAIL'}")

    # Gate 2: Retention (L1 >= 50% of P1) - only check at k>=16
    if config.k_vectors >= 16:
        gates['gate2_retention'] = normal_sr >= config.gate2_threshold
        print(f"Gate 2 (Retention): {normal_sr:.1%} >= {config.gate2_threshold:.1%}? "
              f"{'PASS' if gates['gate2_retention'] else 'FAIL'}")

    # Ablation checks
    null_sr = results.get('null_message', {}).get('success_rate', 0)
    random_sr = results.get('random_latent', {}).get('success_rate', 0)

    # Null and random should be ~P0
    gates['ablation_null'] = abs(null_sr - config.p0_baseline) < 0.15
    gates['ablation_random'] = abs(random_sr - config.p0_baseline) < 0.15

    print(f"\nAblation (null ~P0): {null_sr:.1%} ~ {config.p0_baseline:.1%}? "
          f"{'OK' if gates['ablation_null'] else 'UNEXPECTED'}")
    print(f"Ablation (random ~P0): {random_sr:.1%} ~ {config.p0_baseline:.1%}? "
          f"{'OK' if gates['ablation_random'] else 'UNEXPECTED'}")

    return gates


# ============================================================================
# Modal App Definition
# ============================================================================

if MODAL_AVAILABLE:
    app = modal.App("m2-codec-training")

    # Define the image with dependencies
    image = (
        modal.Image.debian_slim(python_version="3.10")
        .pip_install(
            "torch>=2.0.0",
            "transformers>=4.35.0",
            "tqdm",
            "numpy",
        )
    )

    @app.function(
        gpu="A100",
        timeout=3600,
        image=image,
    )
    def train_on_modal(k: int, epochs: int, data_path: str) -> Dict:
        """Train codec on Modal A100."""
        config = M2Config(
            k_vectors=k,
            epochs=epochs,
            train_data_path=data_path,
        )

        # Train
        train_result = train_codec(config, device="cuda")

        # Evaluate
        eval_result = evaluate_codec(
            train_result['checkpoint_path'],
            config,
            device="cuda",
        )

        # Check gates
        gates = check_gates(eval_result, config)

        return {
            'k': k,
            'train': train_result,
            'eval': eval_result,
            'gates': gates,
        }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="M2-SCALE Codec Training")
    parser.add_argument("--k", type=int, default=16, help="Number of latent vectors")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--data", type=str, default="data/m2_train.jsonl")
    parser.add_argument("--local", action="store_true", help="Run locally (not on Modal)")
    parser.add_argument("--sweep", action="store_true", help="Run full k sweep")
    parser.add_argument("--device", type=str, default="mps")

    args = parser.parse_args()

    if args.sweep:
        k_values = [4, 8, 16, 32, 64]
        print(f"Running k sweep: {k_values}")

        if args.local:
            for k in k_values:
                config = M2Config(k_vectors=k, epochs=args.epochs, train_data_path=args.data)
                train_result = train_codec(config, device=args.device)
                eval_result = evaluate_codec(train_result['checkpoint_path'], config, device=args.device)
                check_gates(eval_result, config)
        else:
            if not MODAL_AVAILABLE:
                print("Modal not available. Install with: pip install modal")
                sys.exit(1)
            # Run on Modal in parallel
            with app.run():
                results = []
                for k in k_values:
                    result = train_on_modal.remote(k, args.epochs, args.data)
                    results.append(result)
                # Collect results
                for r in results:
                    print(f"\nk={r['k']}: {r['gates']}")

    else:
        # Single k training
        config = M2Config(k_vectors=args.k, epochs=args.epochs, train_data_path=args.data)

        if args.local:
            train_result = train_codec(config, device=args.device)
            eval_result = evaluate_codec(train_result['checkpoint_path'], config, device=args.device)
            gates = check_gates(eval_result, config)
        else:
            if not MODAL_AVAILABLE:
                print("Modal not available. Install with: pip install modal")
                sys.exit(1)
            with app.run():
                result = train_on_modal.remote(args.k, args.epochs, args.data)
                print(f"\nResult: {result}")


if __name__ == "__main__":
    main()
