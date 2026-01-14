#!/usr/bin/env python3
"""
M2: Continuous Codec Training on Modal

This script trains an encoder-decoder to convert text messages
into latent packets that can be injected as prefix embeddings.

Usage:
    # Local test (CPU)
    modal run modal/train_codec.py --local

    # GPU training
    modal run modal/train_codec.py

    # Sweep k values
    modal run modal/train_codec.py --k 4 8 16 32
"""

import modal

# Define the Modal app
app = modal.App("lpca-codec-training")

# Container image with all dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "torch>=2.0",
    "transformers>=4.40",
    "numpy",
    "tqdm",
    "scipy",
    "pandas",
])

# Shared volume for checkpoints and data
volume = modal.Volume.from_name("lpca-data", create_if_missing=True)


@app.function(
    gpu="a100-40gb",
    image=image,
    timeout=7200,  # 2 hours max
    volumes={"/data": volume},
)
def train_codec(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    k_vectors: int = 16,
    hidden_dim: int = 1024,
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 3e-4,
    data_path: str = "/data/p1_episodes.jsonl",
):
    """
    Train continuous codec on P1 episodes.

    The codec learns to:
    1. Encode: text message -> k latent vectors
    2. Decode: k latent vectors -> prefix embeddings for receiver

    Loss: behavior cloning (match P1 outputs) + reconstruction
    """
    import json
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm import tqdm
    from pathlib import Path

    print(f"=" * 60)
    print(f"M2: Continuous Codec Training")
    print(f"=" * 60)
    print(f"Model: {model_name}")
    print(f"k vectors: {k_vectors}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"=" * 60)

    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load model and tokenizer
    print(f"\nLoading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    d_model = model.config.hidden_size
    print(f"Model loaded: d_model={d_model}")

    # Define codec architecture
    class LatentCodec(nn.Module):
        """Encoder-decoder for continuous latent packets."""

        def __init__(self, d_model: int, k_vectors: int, hidden_dim: int = 1024):
            super().__init__()
            self.k = k_vectors
            self.d = d_model

            # Encoder: text embedding -> k latent vectors
            self.encoder = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, k_vectors * d_model),
            )

            # Decoder: k latent vectors -> prefix embedding
            self.decoder = nn.Sequential(
                nn.Linear(k_vectors * d_model, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, d_model),
            )

        def encode(self, text_embedding: torch.Tensor) -> torch.Tensor:
            """Encode text embedding to k latent vectors."""
            # text_embedding: (batch, d_model)
            encoded = self.encoder(text_embedding)  # (batch, k * d_model)
            return encoded.view(-1, self.k, self.d)  # (batch, k, d_model)

        def decode(self, latent: torch.Tensor) -> torch.Tensor:
            """Decode k latent vectors to prefix embedding."""
            # latent: (batch, k, d_model)
            flat = latent.view(-1, self.k * self.d)  # (batch, k * d_model)
            return self.decoder(flat)  # (batch, d_model)

        def forward(self, text_embedding: torch.Tensor) -> tuple:
            """Full encode-decode pass."""
            latent = self.encode(text_embedding)
            reconstructed = self.decode(latent)
            return latent, reconstructed

    # Initialize codec
    codec = LatentCodec(d_model, k_vectors, hidden_dim).to(device)
    codec = codec.float()  # Train in fp32
    print(f"Codec parameters: {sum(p.numel() for p in codec.parameters()):,}")

    # Load training data
    data_file = Path(data_path)
    if not data_file.exists():
        print(f"\nNo training data found at {data_path}")
        print("Please run data collection first:")
        print("  modal run modal/collect_data.py")
        return {"status": "error", "message": "No training data"}

    print(f"\nLoading training data from {data_path}...")
    episodes = []
    with open(data_file) as f:
        for line in f:
            episodes.append(json.loads(line))
    print(f"Loaded {len(episodes)} episodes")

    # Prepare dataset
    class P1Dataset(Dataset):
        """Dataset of P1 episodes for codec training."""

        def __init__(self, episodes, tokenizer, model, max_length=256):
            self.data = []
            for ep in tqdm(episodes, desc="Preparing data"):
                if not ep.get("success"):
                    continue

                # Extract sender message and receiver response
                message = ep.get("sender_message", "")
                response = ep.get("receiver_response", "")

                if message and response:
                    # Tokenize
                    msg_ids = tokenizer.encode(message, add_special_tokens=False)[:max_length]
                    resp_ids = tokenizer.encode(response, add_special_tokens=False)[:max_length]

                    self.data.append({
                        "message_ids": msg_ids,
                        "response_ids": resp_ids,
                    })

            print(f"Prepared {len(self.data)} training examples")

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    dataset = P1Dataset(episodes, tokenizer, model)

    if len(dataset) < 10:
        print("Not enough successful episodes for training")
        return {"status": "error", "message": "Insufficient training data"}

    # Training loop
    optimizer = torch.optim.AdamW(codec.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * len(dataset))

    print(f"\nStarting training...")
    for epoch in range(epochs):
        codec.train()
        total_loss = 0
        n_batches = 0

        for i, example in enumerate(tqdm(dataset, desc=f"Epoch {epoch+1}/{epochs}")):
            # Get message embedding from model
            msg_ids = torch.tensor([example["message_ids"]], device=device)

            with torch.no_grad():
                outputs = model(msg_ids, output_hidden_states=True)
                # Use last hidden state, mean-pooled
                msg_embedding = outputs.hidden_states[-1].mean(dim=1).float()  # (1, d_model)

            # Forward through codec
            latent, reconstructed = codec(msg_embedding)

            # Reconstruction loss
            loss = F.mse_loss(reconstructed, msg_embedding)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch+1}: avg_loss={avg_loss:.6f}")

        # Save checkpoint
        checkpoint_path = f"/data/codec_k{k_vectors}_epoch{epoch+1}.pt"
        torch.save({
            "epoch": epoch + 1,
            "k_vectors": k_vectors,
            "hidden_dim": hidden_dim,
            "d_model": d_model,
            "state_dict": codec.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": avg_loss,
        }, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    # Final save
    final_path = f"/data/codec_k{k_vectors}_final.pt"
    torch.save({
        "k_vectors": k_vectors,
        "hidden_dim": hidden_dim,
        "d_model": d_model,
        "state_dict": codec.state_dict(),
    }, final_path)
    volume.commit()

    print(f"\nTraining complete!")
    print(f"Final model saved to: {final_path}")

    return {
        "status": "success",
        "k_vectors": k_vectors,
        "final_loss": avg_loss,
        "checkpoint": final_path,
    }


@app.function(
    gpu="a10g",  # Cheaper GPU for data collection
    image=image,
    timeout=3600,
    volumes={"/data": volume},
)
def collect_p1_data(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    n_episodes: int = 100,
    output_path: str = "/data/p1_episodes.jsonl",
):
    """
    Collect successful P1 episodes for codec training.

    Runs text-based communication episodes and saves successful ones
    with sender messages and receiver responses.
    """
    import json
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from pathlib import Path

    print(f"Collecting P1 episodes: {n_episodes}")
    print(f"Model: {model_name}")

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    # TODO: Implement actual P1 episode collection
    # This should use the split_synthetic environment and text protocol
    # For now, create placeholder data structure

    episodes = []
    for i in range(n_episodes):
        # Placeholder - replace with actual episode running
        episodes.append({
            "episode_id": f"p1_{i:04d}",
            "success": False,  # Will be set by actual evaluation
            "sender_message": "",
            "receiver_response": "",
        })

    # Save
    output_file = Path(output_path)
    with open(output_file, "w") as f:
        for ep in episodes:
            f.write(json.dumps(ep) + "\n")

    volume.commit()
    print(f"Saved {len(episodes)} episodes to {output_path}")

    return {"n_episodes": len(episodes), "output_path": output_path}


@app.local_entrypoint()
def main(
    k: list[int] = [16],
    collect_data: bool = False,
    n_episodes: int = 100,
):
    """
    Main entry point for Modal.

    Usage:
        # Collect data first
        modal run modal/train_codec.py --collect-data --n-episodes 200

        # Train with default k=16
        modal run modal/train_codec.py

        # Train with multiple k values
        modal run modal/train_codec.py --k 4 8 16 32
    """
    if collect_data:
        print("Collecting P1 training data...")
        result = collect_p1_data.remote(n_episodes=n_episodes)
        print(f"Data collection result: {result}")
        return

    # Train codec for each k value
    for k_val in k:
        print(f"\n{'='*60}")
        print(f"Training codec with k={k_val}")
        print(f"{'='*60}")

        result = train_codec.remote(k_vectors=k_val)
        print(f"Result: {result}")
