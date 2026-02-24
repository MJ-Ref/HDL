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
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import random

# Ensure repository-local imports (e.g., `lpca`) work when invoked as
# `python modal/train_m2_codec.py`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
    codec_variant: str = "legacy"  # legacy | m2v2
    codebook_size: int = 256
    vq_beta: float = 0.25

    # Training
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-3
    warmup_steps: int = 100
    curriculum_warmup_steps: int = 500
    curriculum_ramp_steps: int = 1500
    base_seed: int = 1000
    output_path: Optional[str] = None

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
# Embedding Utilities
# ============================================================================


def get_last_token_embedding(hidden_states, attention_mask):
    """Extract embedding of the last actual token (not padding).

    For decoder-only models like Gemma/Qwen, the last token contains
    more discriminative information than mean pooling.

    Args:
        hidden_states: (batch, seq_len, hidden_size)
        attention_mask: (batch, seq_len)

    Returns:
        (batch, hidden_size) - embedding of last actual token
    """
    import torch

    # Get position of last actual token for each sample
    seq_lengths = attention_mask.sum(dim=1) - 1  # (batch,) - indices of last tokens
    batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    last_token_emb = hidden_states[batch_indices, seq_lengths]  # (batch, hidden_size)
    return last_token_emb


# ============================================================================
# Codec Architecture
# ============================================================================


def create_codec_model(
    d_model: int,
    k_vectors: int,
    hidden_dim: int = 512,
    codec_variant: str = "legacy",
    codebook_size: int = 256,
    vq_beta: float = 0.25,
):
    """Create encoder-decoder codec for latent communication."""
    import torch
    import torch.nn as nn

    if codec_variant == "m2v2":
        from lpca.training.m2v2 import M2V2Codec

        class M2V2CompatCodec(nn.Module):
            """
            Compatibility wrapper exposing legacy encode/decode API over M2v2 core.

            This keeps evaluation and ablation plumbing unchanged while enabling
            a discrete VQ bottleneck + MI regularization path during training.
            """

            def __init__(
                self,
                d_model: int,
                k_vectors: int,
                hidden_dim: int,
                codebook_size: int,
                vq_beta: float,
            ):
                super().__init__()
                self.k = k_vectors
                self.d = d_model
                self.core = M2V2Codec(
                    d_model=d_model,
                    k_vectors=k_vectors,
                    hidden_dim=hidden_dim,
                    codebook_size=codebook_size,
                    vq_beta=vq_beta,
                )
                self.prefix_norm = nn.LayerNorm(d_model)
                self.scale_gate = nn.Parameter(torch.tensor(0.5))
                self.register_buffer("target_norm", torch.tensor(1.0))
                self.latest_vq_loss = torch.tensor(0.0)
                self.latest_assignment_probs: Optional[torch.Tensor] = None

            def encode(self, text_embedding: "torch.Tensor") -> "torch.Tensor":
                slots = self.core.encoder(text_embedding).view(-1, self.k, self.d)
                bottleneck = self.core.bottleneck(slots)
                self.latest_vq_loss = bottleneck.vq_loss
                self.latest_assignment_probs = bottleneck.assignment_probs
                return bottleneck.quantized

            def decode(self, latent: "torch.Tensor") -> "torch.Tensor":
                flat = latent.view(-1, self.k * self.d)
                prefix = self.core.decoder(flat).view(-1, self.k, self.d)
                prefix = self.prefix_norm(prefix)
                prefix = prefix * (self.target_norm / (self.d**0.5))
                prefix = prefix * self.scale_gate
                return prefix

            def reconstruct(self, latent: "torch.Tensor") -> "torch.Tensor":
                flat = latent.view(-1, self.k * self.d)
                return self.core.reconstructor(flat)

            def forward(
                self, text_embedding: "torch.Tensor"
            ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
                latent = self.encode(text_embedding)
                prefix_embeddings = self.decode(latent)
                reconstructed = self.reconstruct(latent)
                return latent, prefix_embeddings, reconstructed

        return M2V2CompatCodec(
            d_model=d_model,
            k_vectors=k_vectors,
            hidden_dim=hidden_dim,
            codebook_size=codebook_size,
            vq_beta=vq_beta,
        )

    class LatentCodec(nn.Module):
        """
        Encoder-decoder for latent communication (Option A).

        Encoder: text embedding -> k latent vectors
        Decoder: k vectors -> prefix embeddings for receiver

        Prefix calibration:
        - LayerNorm on output ensures embeddings have proper scale
        - Learned scale gate starts small (near null) and grows during training
        """

        def __init__(self, d_model: int, k_vectors: int, hidden_dim: int):
            super().__init__()
            self.k = k_vectors
            self.d = d_model
            self.training_noise_scale = 0.0  # Disabled - let diversity loss handle it

            # Encoder: pooled text embedding -> k latent vectors
            # NO dropout - let reconstruction loss enforce input-dependent outputs
            self.encoder = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, k_vectors * d_model),
            )

            # Decoder: k vectors -> k prefix embeddings (NOT 1!)
            # This gives the model k prefix tokens to convey information
            # NO dropout - fully deterministic
            self.decoder = nn.Sequential(
                nn.Linear(k_vectors * d_model, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, k_vectors * d_model),  # Output k vectors, not 1
            )

            # Reconstructor: latent -> original message embedding
            # Forces encoder to retain message-specific information
            self.reconstructor = nn.Sequential(
                nn.Linear(k_vectors * d_model, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, d_model),  # Reconstruct original embedding
            )

            # === PREFIX CALIBRATION ===
            # LayerNorm ensures output vectors have normalized scale
            self.prefix_norm = nn.LayerNorm(d_model)

            # Learned scale gate: starts moderate (0.5) so prefixes have influence
            # Previous 0.1 was too conservative - prefixes had no effect on model
            self.scale_gate = nn.Parameter(torch.tensor(0.5))

            # Target norm: will be set from model's token embeddings during first forward
            self.register_buffer("target_norm", torch.tensor(1.0))

        def encode(self, text_embedding: "torch.Tensor") -> "torch.Tensor":
            """Encode text embedding to k latent vectors."""
            # text_embedding: (batch, d_model)
            encoded = self.encoder(text_embedding)  # (batch, k * d_model)
            # Add noise during training to break collapse
            if self.training and self.training_noise_scale > 0:
                noise = torch.randn_like(encoded) * self.training_noise_scale
                encoded = encoded + noise
            return encoded.view(-1, self.k, self.d)  # (batch, k, d_model)

        def decode(self, latent: "torch.Tensor") -> "torch.Tensor":
            """Identity decoder with calibration: latent vectors ARE the prefix embeddings.

            This removes the decoder bottleneck that was causing collapse.
            The encoder directly produces prefix embeddings.

            Calibration:
            - LayerNorm ensures proper scale per-vector
            - Scale gate controls overall influence (starts small, grows during training)
            - Target norm rescales to match typical token embedding magnitude
            """
            # latent: (batch, k, d_model)
            # Apply LayerNorm per-vector to normalize scale
            prefix = self.prefix_norm(latent)

            # Rescale to match target token embedding norm
            # LayerNorm outputs have norm ~sqrt(d_model), we want target_norm
            prefix = prefix * (self.target_norm / (self.d**0.5))

            # Apply learned scale gate (starts at 0.1, can grow during training)
            prefix = prefix * self.scale_gate

            return prefix

        def reconstruct(self, latent: "torch.Tensor") -> "torch.Tensor":
            """Reconstruct original message embedding from latent."""
            # latent: (batch, k, d_model)
            flat = latent.view(-1, self.k * self.d)  # (batch, k * d_model)
            return self.reconstructor(flat)  # (batch, d_model)

        def forward(
            self, text_embedding: "torch.Tensor"
        ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            """Full encode-decode pass with reconstruction."""
            latent = self.encode(text_embedding)
            prefix_embeddings = self.decode(latent)  # (batch, k, d_model)
            reconstructed = self.reconstruct(latent)  # (batch, d_model)
            return latent, prefix_embeddings, reconstructed

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
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm import tqdm
    from lpca.training.m2v2 import CurriculumScheduler, combine_m2v2_losses

    print("=" * 60)
    print(
        f"M2-SCALE Training: k={config.k_vectors}, "
        f"codec_variant={config.codec_variant}"
    )
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
    codec = create_codec_model(
        d_model=d_model,
        k_vectors=config.k_vectors,
        hidden_dim=config.hidden_dim,
        codec_variant=config.codec_variant,
        codebook_size=config.codebook_size,
        vq_beta=config.vq_beta,
    )
    codec = codec.to(device)

    # Set target norm from model's actual token embeddings
    with torch.no_grad():
        token_embeds = model.get_input_embeddings().weight
        target_norm = token_embeds.norm(dim=1).mean().item()
        codec.target_norm.fill_(target_norm)
    print(f"Prefix target norm set to: {target_norm:.4f}")

    n_params = sum(p.numel() for p in codec.parameters())
    print(f"Codec parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    # Load training data
    print(f"Loading training data from {config.train_data_path}...")

    class M2Dataset(Dataset):
        def __init__(self, data_path: str):
            self.samples = []
            with open(data_path, "r") as f:
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

    # Training loop with VALUE-WEIGHTED CE + CONTRASTIVE + RECONSTRUCTION loss
    # Key insight: KL over "ANSWER: {"x1": 0, ...}" is dominated by formatting tokens
    # Solution: Upweight value tokens (actual digits) + contrastive term for shuffle
    # CRITICAL: Reconstruction loss forces encoder to retain message-specific information
    history = {
        "loss": [],
        "ce_loss": [],
        "contrastive_loss": [],
        "diversity_loss": [],
        "recon_loss": [],
        "help_loss": [],
        "mi_loss": [],
        "vq_loss": [],
        "info_preserve_loss": [],
        "mean_cosim": [],
        "epoch": [],
        "tokens_per_sample": [],
    }

    import re

    def extract_answer_with_value_mask(
        receiver_output: str,
    ) -> Optional[Tuple[List[int], List[float]]]:
        """
        Extract answer tokens AND create a weight mask that upweights value tokens.

        Returns (tokens, weights) where weights[i] >> 1.0 for actual value digits.
        """
        # Look for ANSWER: {"x1": V1, "x2": V2, "x3": V3, "x4": V4} pattern
        match = re.search(r"ANSWER:\s*(\{[^}]+\})", receiver_output)
        if not match:
            return None

        answer_json = match.group(1)
        answer_text = "ANSWER: " + answer_json
        tokens = tokenizer.encode(answer_text, add_special_tokens=False)

        if len(tokens) < 5:
            return None

        # Create weight mask: 1.0 for formatting, VALUE_WEIGHT for actual values
        VALUE_WEIGHT = 10.0  # Upweight value tokens 10x
        weights = [1.0] * len(tokens)

        # Decode each token and check if it's a value digit (0, 1, 2)
        # Also upweight tokens that come right after ": " (the value position)
        decoded_tokens = [tokenizer.decode([t]) for t in tokens]
        for i, tok_text in enumerate(decoded_tokens):
            # Check if token is a digit that could be a value
            tok_stripped = tok_text.strip()
            if tok_stripped in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                # Check context: is this after a colon? (value position)
                if i > 0:
                    prev_text = tokenizer.decode(tokens[:i])
                    if prev_text.rstrip().endswith(":"):
                        weights[i] = VALUE_WEIGHT

        return tokens, weights

    # Collect all messages for shuffling (negative sampling)
    all_messages = [
        s["sender_message"] for s in dataset.samples if s.get("sender_message")
    ]

    print(
        f"\nTraining for {config.epochs} epochs (VALUE-WEIGHTED CE + CONTRASTIVE + RECONSTRUCTION + INFO-PRESERVE)..."
    )
    # Loss hyperparameters
    CONTRASTIVE_MARGIN = 1.0  # Margin for contrastive loss
    CONTRASTIVE_WEIGHT = 0.5  # Weight of contrastive term
    DIVERSITY_WEIGHT = 0.0  # Disabled - info preservation handles this
    DIVERSITY_THRESHOLD = 0.5  # N/A when weight is 0
    RECONSTRUCTION_WEIGHT = 20.0  # Strong - helps with input-dependent encoding
    INFO_PRESERVE_WEIGHT = 10.0  # Force encoder to preserve pairwise input similarities
    HELP_MARGIN = 0.2  # Correct prefix should be better than null by this margin (value-weighted NLL)
    HELP_WEIGHT = 1.0  # Weight for help loss: enforces correct > null
    curriculum = CurriculumScheduler(
        warmup_steps=config.curriculum_warmup_steps,
        semantic_ramp_steps=config.curriculum_ramp_steps,
    )
    global_step = 0

    for epoch in range(config.epochs):
        codec.train()
        epoch_loss = 0
        epoch_ce = 0
        epoch_contrastive = 0
        epoch_diversity = 0
        epoch_recon = 0  # Track reconstruction loss
        epoch_info_preserve = 0  # Track info preservation loss
        epoch_help = 0  # Track help loss (correct > null)
        epoch_mi = 0
        epoch_vq = 0
        epoch_cosim = 0  # Track mean cosine similarity for debugging
        epoch_tokens = 0
        n_batches = 0

        # Shuffle messages for negative sampling at start of each epoch
        shuffled_messages = all_messages.copy()
        random.shuffle(shuffled_messages)
        shuffle_idx = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.epochs}"):
            batch_loss = 0
            batch_ce = 0
            batch_contrastive = 0
            batch_recon = 0
            batch_help = 0  # Track help loss within batch
            batch_mi = 0
            batch_vq = 0
            batch_tokens = 0
            n_samples = 0
            batch_prefixes = []  # Collect prefixes for diversity loss
            batch_msg_pooled = []  # Collect input embeddings for info preservation loss
            batch_latents = []  # Collect latent embeddings for info preservation loss

            for sample in batch:
                receiver_context = sample["receiver_context"]
                sender_message = sample["sender_message"]
                receiver_output = sample.get("receiver_output", "")

                # Only train on samples with actual ANSWER output + value weights
                result = extract_answer_with_value_mask(receiver_output)
                if result is None:
                    continue  # Skip samples without clear answer

                answer_tokens, token_weights = result

                # Limit answer length for efficiency
                answer_tokens = answer_tokens[:64]
                token_weights = token_weights[:64]
                answer_tensor = torch.tensor([answer_tokens], device=device)
                weight_tensor = torch.tensor(
                    [token_weights], device=device, dtype=torch.float32
                )
                n_answer_tokens = len(answer_tokens)

                # Get shuffled (wrong) message for contrastive negative
                wrong_message = shuffled_messages[shuffle_idx % len(shuffled_messages)]
                # Ensure it's actually different
                while wrong_message == sender_message and len(shuffled_messages) > 1:
                    shuffle_idx += 1
                    wrong_message = shuffled_messages[
                        shuffle_idx % len(shuffled_messages)
                    ]
                shuffle_idx += 1

                # === STUDENT: Agent B with no text (will get prefix) + answer tokens ===
                student_prompt = f"""{receiver_context}

Your partner sent you encoded information (injected as a learned prefix).
Based on your constraints and the encoded context, provide an ANSWER.
"""

                # Tokenize prompt (without answer)
                student_prompt_inputs = tokenizer(
                    student_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    add_special_tokens=True,
                ).to(device)

                # Concatenate prompt + answer tokens for teacher-forcing
                student_input_ids = torch.cat(
                    [student_prompt_inputs["input_ids"], answer_tensor], dim=1
                )
                student_mask = torch.cat(
                    [
                        student_prompt_inputs["attention_mask"],
                        torch.ones(
                            1,
                            n_answer_tokens,
                            device=device,
                            dtype=student_prompt_inputs["attention_mask"].dtype,
                        ),
                    ],
                    dim=1,
                )

                # === ENCODE CORRECT MESSAGE ===
                msg_inputs = tokenizer(
                    sender_message,
                    return_tensors="pt",
                    truncation=True,
                    max_length=256,
                ).to(device)

                with torch.no_grad():
                    msg_outputs = model(**msg_inputs, output_hidden_states=True)
                    msg_hidden = msg_outputs.hidden_states[-1]
                    # Use last-token pooling instead of mean pooling for better discrimination
                    msg_pooled = get_last_token_embedding(
                        msg_hidden, msg_inputs["attention_mask"]
                    )

                # Encode correct message through codec → k prefix embeddings + reconstruction
                # Cast to float32 for codec (model outputs float16 on CUDA)
                latent, prefix_embeddings, reconstructed = codec(
                    msg_pooled.float()
                )  # prefix_embeddings: (1, k, d_model)
                k_prefix = prefix_embeddings.shape[1]

                # Reconstruction loss: force encoder to retain message-specific info
                # Use MSE loss - stricter than cosine similarity, forces exact reconstruction
                # Normalize both to unit norm first to avoid scale issues
                # Cast target to float32 to match codec output
                recon_normed = F.normalize(reconstructed, dim=1)
                target_normed = F.normalize(msg_pooled.detach().float(), dim=1)
                recon_loss = F.mse_loss(recon_normed, target_normed)

                # Collect flattened prefix for diversity loss (keep gradients!)
                prefix_flat = prefix_embeddings.view(1, -1)  # (1, k*d_model)
                batch_prefixes.append(prefix_flat)

                # Collect for info preservation loss
                # msg_pooled is detached (no_grad), latent has gradients
                # Cast to float32 to match codec output
                batch_msg_pooled.append(msg_pooled.detach().float())  # (1, d_model)
                batch_latents.append(latent.view(1, -1))  # (1, k*d_model)

                # === ENCODE WRONG MESSAGE (for contrastive) ===
                wrong_msg_inputs = tokenizer(
                    wrong_message,
                    return_tensors="pt",
                    truncation=True,
                    max_length=256,
                ).to(device)

                with torch.no_grad():
                    wrong_msg_outputs = model(
                        **wrong_msg_inputs, output_hidden_states=True
                    )
                    wrong_hidden = wrong_msg_outputs.hidden_states[-1]
                    # Use last-token pooling for wrong message too
                    wrong_pooled = get_last_token_embedding(
                        wrong_hidden, wrong_msg_inputs["attention_mask"]
                    )

                # Encode wrong message through codec (ignore reconstruction for wrong)
                wrong_latent, wrong_prefix_embeddings, _ = codec(wrong_pooled.float())

                # === STUDENT FORWARD WITH CORRECT PREFIX ===
                student_embeds = model.get_input_embeddings()(student_input_ids)
                soft_prefix = prefix_embeddings.to(
                    student_embeds.dtype
                )  # (1, k, d_model)
                combined_embeds = torch.cat([soft_prefix, student_embeds], dim=1)

                soft_mask = torch.ones(
                    1, k_prefix, device=device, dtype=student_mask.dtype
                )
                combined_mask = torch.cat([soft_mask, student_mask], dim=1)

                student_outputs = model(
                    inputs_embeds=combined_embeds, attention_mask=combined_mask
                )
                student_prompt_len = (
                    student_prompt_inputs["input_ids"].shape[1] + k_prefix
                )
                student_logits = student_outputs.logits[
                    :, student_prompt_len - 1 : -1, :
                ]  # (1, n_answer_tokens, vocab)

                # === STUDENT FORWARD WITH WRONG PREFIX (for contrastive) ===
                wrong_soft_prefix = wrong_prefix_embeddings.to(student_embeds.dtype)
                wrong_combined_embeds = torch.cat(
                    [wrong_soft_prefix, student_embeds], dim=1
                )

                wrong_student_outputs = model(
                    inputs_embeds=wrong_combined_embeds, attention_mask=combined_mask
                )
                wrong_student_logits = wrong_student_outputs.logits[
                    :, student_prompt_len - 1 : -1, :
                ]

                # === VALUE-WEIGHTED CROSS-ENTROPY LOSS (correct prefix) ===
                # Targets are the answer tokens (shifted by 1 for next-token prediction)
                targets = answer_tensor  # (1, n_answer_tokens)
                log_probs = F.log_softmax(
                    student_logits, dim=-1
                )  # (1, n_answer_tokens, vocab)

                # Gather log probs for correct tokens
                target_log_probs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(
                    -1
                )  # (1, n_answer_tokens)

                # Apply value weights (upweight value tokens)
                weighted_nll = -target_log_probs * weight_tensor  # (1, n_answer_tokens)
                ce_loss_pos = (
                    weighted_nll.sum() / weight_tensor.sum()
                )  # Normalize by total weight

                # === CE FOR WRONG PREFIX ===
                wrong_log_probs = F.log_softmax(wrong_student_logits, dim=-1)
                wrong_target_log_probs = wrong_log_probs.gather(
                    2, targets.unsqueeze(-1)
                ).squeeze(-1)
                weighted_wrong_nll = -wrong_target_log_probs * weight_tensor
                ce_loss_neg = weighted_wrong_nll.sum() / weight_tensor.sum()

                # === CONTRASTIVE MARGIN LOSS ===
                # Want: CE_neg - CE_pos > margin (wrong prefix should make answer harder)
                # Loss: max(0, margin - (CE_neg - CE_pos))
                contrastive_loss = F.relu(
                    CONTRASTIVE_MARGIN - (ce_loss_neg - ce_loss_pos)
                )

                # === HELP LOSS (correct > null) ===
                # Forward pass with null prefix (zeros)
                null_prefix = torch.zeros_like(prefix_embeddings)
                null_soft_prefix = null_prefix.to(student_embeds.dtype)
                null_combined_embeds = torch.cat(
                    [null_soft_prefix, student_embeds], dim=1
                )

                null_student_outputs = model(
                    inputs_embeds=null_combined_embeds, attention_mask=combined_mask
                )
                null_student_logits = null_student_outputs.logits[
                    :, student_prompt_len - 1 : -1, :
                ]

                # Compute value-weighted NLL for null
                null_log_probs = F.log_softmax(null_student_logits, dim=-1)
                null_target_log_probs = null_log_probs.gather(
                    2, targets.unsqueeze(-1)
                ).squeeze(-1)
                weighted_null_nll = -null_target_log_probs * weight_tensor
                ce_loss_null = weighted_null_nll.sum() / weight_tensor.sum()

                # Want: ce_loss_pos < ce_loss_null (correct prefix should be BETTER than null)
                # Loss: max(0, ce_loss_pos - ce_loss_null + margin)
                # This is zero when correct is better than null by at least margin
                help_loss = F.relu(ce_loss_pos - ce_loss_null + HELP_MARGIN)

                if config.codec_variant == "m2v2":
                    if codec.latest_assignment_probs is None:
                        raise RuntimeError(
                            "m2v2 codec must expose assignment_probs during training"
                        )
                    weights = curriculum.weights(global_step)
                    loss_terms = combine_m2v2_losses(
                        nll_pos=ce_loss_pos.unsqueeze(0),
                        nll_shuffle=ce_loss_neg.unsqueeze(0),
                        recon_loss=recon_loss,
                        vq_loss=codec.latest_vq_loss,
                        assignment_probs=codec.latest_assignment_probs,
                        curriculum_weights=weights,
                        margin=CONTRASTIVE_MARGIN,
                    )
                    # Preserve explicit "correct > null" pressure from legacy path.
                    total_loss = loss_terms["total"] + HELP_WEIGHT * help_loss
                    ce_loss_pos = loss_terms["ce"]
                    contrastive_loss = loss_terms["anti_shuffle"]
                    batch_mi += loss_terms["mi"].item()
                    batch_vq += loss_terms["vq"].item()
                else:
                    # === TOTAL LOSS ===
                    total_loss = (
                        ce_loss_pos
                        + CONTRASTIVE_WEIGHT * contrastive_loss
                        + RECONSTRUCTION_WEIGHT * recon_loss
                        + HELP_WEIGHT * help_loss
                    )

                batch_loss += total_loss
                batch_ce += ce_loss_pos.item()
                batch_contrastive += contrastive_loss.item()
                batch_recon += recon_loss.item()
                batch_help += help_loss.item()
                batch_tokens += n_answer_tokens
                n_samples += 1

            # === DIVERSITY LOSS (penalize similar prefixes within batch) ===
            batch_diversity_loss = torch.tensor(0.0, device=device)
            batch_mean_cosim = 0.0
            if len(batch_prefixes) >= 2:
                # Stack all prefixes: (n_samples, k*d_model)
                prefix_matrix = torch.cat(batch_prefixes, dim=0)
                # Normalize for cosine similarity
                prefix_normed = F.normalize(prefix_matrix, dim=1)
                # Pairwise cosine similarities
                cosine_sim = torch.mm(prefix_normed, prefix_normed.t())  # (n, n)
                # Get upper triangular (excluding diagonal)
                n = cosine_sim.shape[0]
                mask = torch.triu(
                    torch.ones(n, n, dtype=torch.bool, device=device), diagonal=1
                )
                pairwise_sims = cosine_sim[mask]
                batch_mean_cosim = pairwise_sims.mean().item()  # Track for debugging
                # Penalize similarities above threshold
                # Loss = mean(max(0, sim - threshold))
                diversity_violation = F.relu(pairwise_sims - DIVERSITY_THRESHOLD)
                batch_diversity_loss = diversity_violation.mean()

            # === INFO PRESERVATION LOSS (force encoder to preserve input similarity structure) ===
            batch_info_preserve_loss = torch.tensor(0.0, device=device)
            if len(batch_msg_pooled) >= 2:
                # Stack inputs and latents
                input_matrix = torch.cat(batch_msg_pooled, dim=0)  # (n, d_model)
                latent_matrix = torch.cat(batch_latents, dim=0)  # (n, k*d_model)

                # Compute pairwise cosine similarities
                input_normed = F.normalize(input_matrix, dim=1)
                latent_normed = F.normalize(latent_matrix, dim=1)

                input_sim = torch.mm(input_normed, input_normed.t())  # (n, n)
                latent_sim = torch.mm(latent_normed, latent_normed.t())  # (n, n)

                # Get upper triangular (excluding diagonal)
                n = input_sim.shape[0]
                mask = torch.triu(
                    torch.ones(n, n, dtype=torch.bool, device=device), diagonal=1
                )

                input_pairwise = input_sim[mask]  # Flatten to 1D
                latent_pairwise = latent_sim[mask]

                # Loss: MSE between input similarities and latent similarities
                # This forces the encoder to preserve relative distances
                batch_info_preserve_loss = F.mse_loss(
                    latent_pairwise, input_pairwise.detach()
                )

            # Average over batch and backprop
            if n_samples > 0:
                loss = (
                    batch_loss / n_samples
                    + DIVERSITY_WEIGHT * batch_diversity_loss
                    + INFO_PRESERVE_WEIGHT * batch_info_preserve_loss
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(codec.parameters(), max_norm=1.0)
                optimizer.step()
                global_step += 1

                epoch_loss += loss.item()
                epoch_ce += batch_ce / n_samples
                epoch_contrastive += batch_contrastive / n_samples
                epoch_diversity += batch_diversity_loss.item()
                epoch_recon += batch_recon / n_samples
                epoch_help += batch_help / n_samples
                epoch_mi += batch_mi / n_samples
                epoch_vq += batch_vq / n_samples
                epoch_info_preserve += batch_info_preserve_loss.item()
                epoch_cosim += batch_mean_cosim  # Track mean cosine similarity
                epoch_tokens += batch_tokens
                n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        avg_ce = epoch_ce / max(n_batches, 1)
        avg_contrastive = epoch_contrastive / max(n_batches, 1)
        avg_diversity = epoch_diversity / max(n_batches, 1)
        avg_recon = epoch_recon / max(n_batches, 1)
        avg_help = epoch_help / max(n_batches, 1)
        avg_mi = epoch_mi / max(n_batches, 1)
        avg_vq = epoch_vq / max(n_batches, 1)
        avg_info_preserve = epoch_info_preserve / max(n_batches, 1)
        avg_cosim = epoch_cosim / max(n_batches, 1)
        avg_tokens = epoch_tokens / max(n_batches * config.batch_size, 1)
        history["loss"].append(avg_loss)
        history["ce_loss"].append(avg_ce)
        history["contrastive_loss"].append(avg_contrastive)
        history["diversity_loss"].append(avg_diversity)
        history["recon_loss"].append(avg_recon)
        history["help_loss"].append(avg_help)
        history["mean_cosim"].append(avg_cosim)
        history["mi_loss"].append(avg_mi)
        history["vq_loss"].append(avg_vq)
        history["info_preserve_loss"].append(avg_info_preserve)
        history["epoch"].append(epoch + 1)
        history["tokens_per_sample"].append(avg_tokens)

        print(
            f"  Epoch {epoch+1}: loss={avg_loss:.4f}, ce={avg_ce:.4f}, "
            f"contrast={avg_contrastive:.4f}, help={avg_help:.4f}, "
            f"recon={avg_recon:.4f}, info={avg_info_preserve:.4f}, "
            f"mi={avg_mi:.4f}, vq={avg_vq:.4f}, "
            f"cosim={avg_cosim:.4f}"
        )

    # Save checkpoint (use /data volume on Modal for persistence)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if config.output_path:
        checkpoint_root = Path(config.output_path).expanduser().resolve().parent
        checkpoint_path = str(
            checkpoint_root
            / (
                f"m2_{config.codec_variant}_k{config.k_vectors}"
                f"_seed{config.base_seed}_{timestamp}.pt"
            )
        )
    elif Path("/data").exists():
        # Modal: save to persistent volume
        checkpoint_path = f"/data/checkpoints/m2_{config.codec_variant}_k{config.k_vectors}_{timestamp}.pt"
    else:
        # Local: save to local checkpoints dir
        checkpoint_path = (
            f"checkpoints/m2_{config.codec_variant}_k{config.k_vectors}_{timestamp}.pt"
        )
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "k_vectors": config.k_vectors,
            "hidden_dim": config.hidden_dim,
            "d_model": d_model,
            "codec_variant": config.codec_variant,
            "codebook_size": config.codebook_size,
            "vq_beta": config.vq_beta,
            "state_dict": codec.state_dict(),
            "config": asdict(config),
            "history": history,
        },
        checkpoint_path,
    )

    print(f"\nCheckpoint saved: {checkpoint_path}")

    return {
        "checkpoint_path": checkpoint_path,
        "history": history,
        "final_loss": history["loss"][-1],
    }


# ============================================================================
# Diagnostic Checks (run before A100 time)
# ============================================================================


def run_diagnostics(
    checkpoint_path: str,
    config: M2Config,
    device: str = "cuda",
    n_samples: int = 20,
) -> Dict[str, Any]:
    """
    Quick diagnostic checks to verify training is learning semantic content.

    1. Prefix collapse check: Are different messages producing similar prefixes?
    2. Deterministic semantic check: Does correct prefix make answer more likely than shuffle?

    Run this locally before spending A100 time on full training.
    """
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import json
    import re

    print("\n" + "=" * 60)
    print("DIAGNOSTIC CHECKS")
    print("=" * 60)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    k_vectors = checkpoint["k_vectors"]
    d_model = checkpoint["d_model"]
    hidden_dim = checkpoint["hidden_dim"]

    print(f"Loaded: k={k_vectors}, d_model={d_model}")

    # Load model and tokenizer
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

    # Recreate codec and load weights
    codec = create_codec_model(
        d_model=d_model,
        k_vectors=k_vectors,
        hidden_dim=hidden_dim,
        codec_variant=checkpoint.get("codec_variant", config.codec_variant),
        codebook_size=checkpoint.get("codebook_size", config.codebook_size),
        vq_beta=checkpoint.get("vq_beta", config.vq_beta),
    )
    # Handle loading old checkpoints that don't have prefix_norm/scale_gate
    try:
        codec.load_state_dict(checkpoint["state_dict"])
    except RuntimeError:
        # Old checkpoint without new calibration params - load what we can
        print("  Note: Loading old checkpoint format, using default calibration")
        codec.load_state_dict(checkpoint["state_dict"], strict=False)
    codec = codec.to(device)
    codec.eval()

    # Set target norm from model's token embeddings
    with torch.no_grad():
        token_embeds = model.get_input_embeddings().weight
        target_norm = token_embeds.norm(dim=1).mean().item()
        codec.target_norm.fill_(target_norm)

    # Load some training samples
    print(f"Loading {n_samples} samples...")
    samples = []
    with open(config.train_data_path, "r") as f:
        for i, line in enumerate(f):
            if i >= n_samples:
                break
            sample = json.loads(line)
            if "ANSWER" in sample.get("receiver_output", ""):
                samples.append(sample)

    if len(samples) < 5:
        print("WARNING: Not enough samples with ANSWER found")
        return {"error": "insufficient_samples"}

    print(f"  Found {len(samples)} samples with ANSWER")

    # ========== CHECK 1: PREFIX COLLAPSE ==========
    print("\n--- Check 1: Prefix Collapse ---")
    print("Computing prefix embeddings for different messages...")

    prefix_embeddings_list = []
    pooled_embeddings_list = []
    latent_embeddings_list = []
    messages = []

    for sample in samples[: min(20, len(samples))]:
        msg = sample["sender_message"]
        messages.append(msg)

        # Encode message
        msg_inputs = tokenizer(
            msg, return_tensors="pt", truncation=True, max_length=256
        ).to(device)
        with torch.no_grad():
            msg_outputs = model(**msg_inputs, output_hidden_states=True)
            hidden = msg_outputs.hidden_states[-1]
            # Use last-token pooling for better discrimination
            pooled = get_last_token_embedding(hidden, msg_inputs["attention_mask"])

            # Get prefix embeddings
            latent, prefix_emb, _ = codec(pooled.float())
            # Flatten k prefix vectors to single vector for comparison
            prefix_flat = prefix_emb.view(1, -1).cpu()
            prefix_embeddings_list.append(prefix_flat)
            pooled_embeddings_list.append(pooled.float().cpu())
            latent_embeddings_list.append(latent.view(1, -1).float().cpu())

    # Debug: Check pooled embedding similarities
    pooled_matrix = torch.cat(pooled_embeddings_list, dim=0)
    pooled_normed = F.normalize(pooled_matrix, dim=1)
    pooled_sim = torch.mm(pooled_normed, pooled_normed.t())
    n_p = pooled_sim.shape[0]
    pooled_mask = ~torch.eye(n_p, dtype=torch.bool)
    pooled_off_diag = pooled_sim[pooled_mask]
    print(
        f"  DEBUG - Pooled embeddings cosim: mean={pooled_off_diag.mean():.4f}, min={pooled_off_diag.min():.4f}, max={pooled_off_diag.max():.4f}"
    )

    # Debug: Check latent similarities (encoder output)
    latent_matrix = torch.cat(latent_embeddings_list, dim=0)
    latent_normed = F.normalize(latent_matrix, dim=1)
    latent_sim = torch.mm(latent_normed, latent_normed.t())
    n_l = latent_sim.shape[0]
    latent_mask = ~torch.eye(n_l, dtype=torch.bool)
    latent_off_diag = latent_sim[latent_mask]
    print(
        f"  DEBUG - Latent (encoder output) cosim: mean={latent_off_diag.mean():.4f}, min={latent_off_diag.min():.4f}, max={latent_off_diag.max():.4f}"
    )

    # Compute pairwise cosine similarities
    prefix_matrix = torch.cat(prefix_embeddings_list, dim=0)  # (n_samples, k*d_model)
    prefix_normed = F.normalize(prefix_matrix, dim=1)
    cosine_sim = torch.mm(prefix_normed, prefix_normed.t())

    # Get off-diagonal similarities (excluding self-similarity)
    n = cosine_sim.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool)
    off_diag_sims = cosine_sim[mask]

    mean_sim = off_diag_sims.mean().item()
    min_sim = off_diag_sims.min().item()
    max_sim = off_diag_sims.max().item()

    print("  Prefix cosine similarity (across different messages):")
    print(f"    Mean: {mean_sim:.4f}")
    print(f"    Min:  {min_sim:.4f}")
    print(f"    Max:  {max_sim:.4f}")

    if mean_sim > 0.95:
        print("  ⚠️ WARNING: Prefixes are nearly identical (collapse detected)")
        print(
            "     The codec is learning a constant 'magic prompt', not semantic encoding"
        )
    elif mean_sim > 0.8:
        print("  ⚠️ CAUTION: Prefixes are quite similar, limited differentiation")
    else:
        print("  ✓ Prefixes show meaningful differentiation")

    # ========== CHECK 2: DETERMINISTIC SEMANTIC CHECK ==========
    # Now with value-weighted, value-only, format-only NLL + JSON-parse rate
    print("\n--- Check 2: Semantic Check (value-weighted metrics) ---")
    print("Comparing log-probs: correct prefix vs shuffled prefix vs null")

    # Helper to create value mask (same logic as training)
    VALUE_WEIGHT = 10.0

    def create_value_mask(answer_tokens, tokenizer):
        """Create weight mask: 1.0 for format tokens, VALUE_WEIGHT for value digits."""
        weights = [1.0] * len(answer_tokens)
        value_mask = [False] * len(answer_tokens)  # True = value token
        decoded_tokens = [tokenizer.decode([t]) for t in answer_tokens]
        for i, tok_text in enumerate(decoded_tokens):
            tok_stripped = tok_text.strip()
            if tok_stripped in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                if i > 0:
                    prev_text = tokenizer.decode(answer_tokens[:i])
                    if prev_text.rstrip().endswith(":"):
                        weights[i] = VALUE_WEIGHT
                        value_mask[i] = True
        return weights, value_mask

    # Metrics storage - now with multiple NLL types
    metrics = {
        "correct": {
            "unweighted": [],
            "weighted": [],
            "value_only": [],
            "format_only": [],
        },
        "shuffle": {
            "unweighted": [],
            "weighted": [],
            "value_only": [],
            "format_only": [],
        },
        "null": {"unweighted": [], "weighted": [], "value_only": [], "format_only": []},
    }
    json_parse_results = {"correct": [], "shuffle": [], "null": []}

    # Also track prefix norms to detect OOD embeddings
    prefix_norms = []
    # Get reference: typical token embedding norm
    with torch.no_grad():
        sample_embeds = model.get_input_embeddings().weight[:1000]  # First 1000 tokens
        ref_norm = sample_embeds.norm(dim=1).mean().item()
    print(f"  Reference token embedding norm: {ref_norm:.4f}")

    for i, sample in enumerate(samples[: min(10, len(samples))]):
        receiver_context = sample["receiver_context"]
        sender_message = sample["sender_message"]
        receiver_output = sample.get("receiver_output", "")

        # Extract answer
        match = re.search(r"ANSWER:\s*(\{[^}]+\})", receiver_output)
        if not match:
            continue

        answer_text = "ANSWER: " + match.group(1)
        answer_tokens = tokenizer.encode(answer_text, add_special_tokens=False)
        answer_tensor = torch.tensor([answer_tokens], device=device)

        # Create value mask
        weights, value_mask = create_value_mask(answer_tokens, tokenizer)
        weight_tensor = torch.tensor([weights], device=device, dtype=torch.float32)
        value_mask_tensor = torch.tensor([value_mask], device=device)

        # Student prompt
        student_prompt = f"""{receiver_context}

Your partner sent you encoded information (injected as a learned prefix).
Based on your constraints and the encoded context, provide an ANSWER.
"""
        student_inputs = tokenizer(
            student_prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(device)
        student_input_ids = torch.cat(
            [student_inputs["input_ids"], answer_tensor], dim=1
        )

        # Get correct message embedding and prefix
        msg_inputs = tokenizer(
            sender_message, return_tensors="pt", truncation=True, max_length=256
        ).to(device)
        with torch.no_grad():
            msg_outputs = model(**msg_inputs, output_hidden_states=True)
            hidden = msg_outputs.hidden_states[-1]
            pooled = get_last_token_embedding(hidden, msg_inputs["attention_mask"])
            latent, correct_prefix, _ = codec(pooled.float())

        # Track prefix norm
        prefix_norm = correct_prefix.norm(dim=-1).mean().item()
        prefix_norms.append(prefix_norm)

        # Get shuffled message prefix
        shuffle_msg = samples[(i + 7) % len(samples)]["sender_message"]
        shuffle_msg_inputs = tokenizer(
            shuffle_msg, return_tensors="pt", truncation=True, max_length=256
        ).to(device)
        with torch.no_grad():
            shuffle_outputs = model(**shuffle_msg_inputs, output_hidden_states=True)
            shuffle_hidden = shuffle_outputs.hidden_states[-1]
            shuffle_pooled = get_last_token_embedding(
                shuffle_hidden, shuffle_msg_inputs["attention_mask"]
            )
            _, shuffle_prefix, _ = codec(shuffle_pooled.float())

        # Null prefix (zeros)
        null_prefix = torch.zeros_like(correct_prefix)

        # Compute NLL for each condition
        student_embeds = model.get_input_embeddings()(student_input_ids)
        k_prefix = correct_prefix.shape[1]
        prompt_len = student_inputs["input_ids"].shape[1] + k_prefix

        conditions = [
            ("correct", correct_prefix),
            ("shuffle", shuffle_prefix),
            ("null", null_prefix),
        ]

        for cond_name, prefix in conditions:
            soft_prefix = prefix.to(student_embeds.dtype)
            combined = torch.cat([soft_prefix, student_embeds], dim=1)
            attn_mask = torch.ones(1, combined.shape[1], device=device)

            with torch.no_grad():
                outputs = model(inputs_embeds=combined, attention_mask=attn_mask)
                logits = outputs.logits[:, prompt_len - 1 : -1, :]

                log_probs = F.log_softmax(logits, dim=-1)
                target_log_probs = log_probs.gather(
                    2, answer_tensor.unsqueeze(-1)
                ).squeeze(-1)

                # 1. Unweighted NLL (old metric)
                nll_unweighted = -target_log_probs.mean().item()
                metrics[cond_name]["unweighted"].append(nll_unweighted)

                # 2. Value-weighted NLL (matches training objective)
                weighted_nll = -target_log_probs * weight_tensor
                nll_weighted = (weighted_nll.sum() / weight_tensor.sum()).item()
                metrics[cond_name]["weighted"].append(nll_weighted)

                # 3. Value-only NLL (just the digits that matter)
                if value_mask_tensor.any():
                    value_nll = -target_log_probs[value_mask_tensor].mean().item()
                else:
                    value_nll = float("nan")
                metrics[cond_name]["value_only"].append(value_nll)

                # 4. Format-only NLL (punctuation, keys, etc)
                format_mask = ~value_mask_tensor
                if format_mask.any():
                    format_nll = -target_log_probs[format_mask].mean().item()
                else:
                    format_nll = float("nan")
                metrics[cond_name]["format_only"].append(format_nll)

            # JSON parse rate under greedy decoding
            with torch.no_grad():
                # Greedy generation from the prompt
                gen_prompt = student_prompt + "\nANSWER: "
                gen_inputs = tokenizer(
                    gen_prompt, return_tensors="pt", truncation=True, max_length=512
                ).to(device)
                gen_embeds = model.get_input_embeddings()(gen_inputs["input_ids"])
                gen_combined = torch.cat([soft_prefix, gen_embeds], dim=1)
                gen_mask = torch.ones(1, gen_combined.shape[1], device=device)

                # Forward pass to get KV cache
                gen_outputs = model(
                    inputs_embeds=gen_combined, attention_mask=gen_mask, use_cache=True
                )
                past_kv = gen_outputs.past_key_values

                # Greedy decode up to 64 tokens
                generated_ids = []
                next_logits = gen_outputs.logits[:, -1, :]
                for _ in range(64):
                    next_token = next_logits.argmax(dim=-1, keepdim=True)
                    generated_ids.append(next_token.item())
                    if next_token.item() == tokenizer.eos_token_id:
                        break
                    # Extend attention mask
                    curr_len = gen_combined.shape[1] + len(generated_ids)
                    ext_mask = torch.ones(1, curr_len, device=device)
                    out = model(
                        input_ids=next_token,
                        attention_mask=ext_mask,
                        past_key_values=past_kv,
                        use_cache=True,
                    )
                    past_kv = out.past_key_values
                    next_logits = out.logits[:, -1, :]

                gen_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                # Try to parse JSON
                json_match = re.search(r"\{[^}]+\}", gen_text)
                parsed = False
                if json_match:
                    try:
                        json.loads(json_match.group())
                        parsed = True
                    except json.JSONDecodeError:
                        pass
                json_parse_results[cond_name].append(parsed)

    # Aggregate results
    def safe_mean(lst):
        valid = [x for x in lst if x == x]  # filter NaN
        return sum(valid) / len(valid) if valid else float("nan")

    results_table = {}
    for cond in ["correct", "shuffle", "null"]:
        results_table[cond] = {
            "unweighted_nll": safe_mean(metrics[cond]["unweighted"]),
            "weighted_nll": safe_mean(metrics[cond]["weighted"]),
            "value_only_nll": safe_mean(metrics[cond]["value_only"]),
            "format_only_nll": safe_mean(metrics[cond]["format_only"]),
            "json_parse_rate": sum(json_parse_results[cond])
            / len(json_parse_results[cond])
            if json_parse_results[cond]
            else 0,
        }

    # Print results
    print(
        f"\n  Prefix norm: mean={safe_mean(prefix_norms):.4f} (ref={ref_norm:.4f}, ratio={safe_mean(prefix_norms)/ref_norm:.2f}x)"
    )

    print("\n  NLL Metrics (lower = better):")
    print(
        f"  {'Condition':<12} {'Unweight':<10} {'Weighted':<10} {'Value':<10} {'Format':<10} {'JSON%':<8}"
    )
    print(f"  {'-'*60}")
    for cond in ["correct", "shuffle", "null"]:
        r = results_table[cond]
        print(
            f"  {cond:<12} {r['unweighted_nll']:<10.4f} {r['weighted_nll']:<10.4f} "
            f"{r['value_only_nll']:<10.4f} {r['format_only_nll']:<10.4f} {r['json_parse_rate']*100:<8.1f}"
        )

    # Compute gaps using weighted NLL (matches training)
    gap_shuffle_weighted = (
        results_table["shuffle"]["weighted_nll"]
        - results_table["correct"]["weighted_nll"]
    )
    gap_null_weighted = (
        results_table["null"]["weighted_nll"] - results_table["correct"]["weighted_nll"]
    )
    gap_shuffle_value = (
        results_table["shuffle"]["value_only_nll"]
        - results_table["correct"]["value_only_nll"]
    )

    print("\n  Gaps (positive = correct is better):")
    print(f"    Shuffle gap (weighted): {gap_shuffle_weighted:.4f}")
    print(f"    Shuffle gap (value-only): {gap_shuffle_value:.4f}")
    print(f"    Null gap (weighted): {gap_null_weighted:.4f}")

    # Check prefix norm
    norm_ratio = safe_mean(prefix_norms) / ref_norm
    if norm_ratio > 3.0:
        print(
            f"  ⚠️ WARNING: Prefix norm is {norm_ratio:.1f}x reference - OOD soft prompts!"
        )
        print("     This may hurt format tokens while still being message-dependent")
    elif norm_ratio < 0.3:
        print(
            f"  ⚠️ WARNING: Prefix norm is {norm_ratio:.1f}x reference - too small to influence"
        )

    if gap_shuffle_weighted > 0.1:
        print(
            "  ✓ Shuffle makes answer harder (weighted) - semantic dependence detected!"
        )
    elif gap_shuffle_weighted > 0:
        print("  ⚠️ Small shuffle gap - weak semantic signal")
    else:
        print("  ❌ No shuffle gap - codec not encoding semantic content")

    return {
        "prefix_collapse": {
            "mean_cosine_sim": mean_sim,
            "min_sim": min_sim,
            "max_sim": max_sim,
        },
        "prefix_norm": {
            "mean": safe_mean(prefix_norms),
            "reference": ref_norm,
            "ratio": norm_ratio,
        },
        "semantic_check": {
            "correct": results_table["correct"],
            "shuffle": results_table["shuffle"],
            "null": results_table["null"],
            "shuffle_gap_weighted": gap_shuffle_weighted,
            "shuffle_gap_value_only": gap_shuffle_value,
            "null_gap_weighted": gap_null_weighted,
        },
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

    Ablations (all required):
    1. Normal (trained codec with injection)
    2. Null message (zero latent)
    3. Random latent
    4. Shuffle (message from different episode)

    VALIDITY REQUIREMENTS:
    - All conditions use same message_cache for paired comparisons
    - Agent B never sees message_A text (no leak)
    - Shuffle must crater below P0 (proves semantic dependence)
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\n" + "=" * 60)
    print("EVALUATION (with validity checks)")
    print("=" * 60)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    k_vectors = checkpoint["k_vectors"]
    d_model = checkpoint["d_model"]
    hidden_dim = checkpoint["hidden_dim"]

    print(f"Loaded: k={k_vectors}, d_model={d_model}")

    # Load model and tokenizer
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

    # Recreate codec and load weights
    codec = create_codec_model(
        d_model=d_model,
        k_vectors=k_vectors,
        hidden_dim=hidden_dim,
        codec_variant=checkpoint.get("codec_variant", config.codec_variant),
        codebook_size=checkpoint.get("codebook_size", config.codebook_size),
        vq_beta=checkpoint.get("vq_beta", config.vq_beta),
    )
    # Handle loading old checkpoints that don't have prefix_norm/scale_gate
    try:
        codec.load_state_dict(checkpoint["state_dict"])
    except RuntimeError:
        print("  Note: Loading old checkpoint format, using default calibration")
        codec.load_state_dict(checkpoint["state_dict"], strict=False)
    codec = codec.to(device)
    codec.eval()

    # Set target norm from model's token embeddings
    with torch.no_grad():
        token_embeds = model.get_input_embeddings().weight
        target_norm = token_embeds.norm(dim=1).mean().item()
        codec.target_norm.fill_(target_norm)

    results = {}

    # Test on fresh seeds
    test_start, test_end = config.test_seeds
    n_episodes = min(config.eval_episodes, test_end - test_start)
    seed_list = list(range(test_start, test_start + n_episodes))

    print(
        f"\nEvaluating on {n_episodes} fresh episodes (seeds {test_start}-{test_start+n_episodes-1})"
    )

    # P2 FIX: Shared message cache for paired comparisons
    message_cache: Dict[int, str] = {}

    # Phase 1: Run normal condition (also populates message cache)
    print("\nNormal (with injection):")
    success_count = 0
    for seed in seed_list:
        success = _run_codec_episode(
            seed=seed,
            model=model,
            tokenizer=tokenizer,
            codec=codec,
            device=device,
            temperature=config.temperature,
            ablation=None,
            message_cache=message_cache,
            seed_list=seed_list,
        )
        if success:
            success_count += 1

    success_rate = success_count / n_episodes
    results["normal"] = {
        "success_rate": success_rate,
        "successes": success_count,
        "n_episodes": n_episodes,
    }
    print(f"  Normal: {success_rate:.1%} ({success_count}/{n_episodes})")

    if run_ablations:
        print("\nRunning ablations (using cached messages for paired comparison)...")

        # Null message ablation (zero latent)
        null_success = 0
        for seed in seed_list:
            success = _run_codec_episode(
                seed=seed,
                model=model,
                tokenizer=tokenizer,
                codec=codec,
                device=device,
                temperature=config.temperature,
                ablation="null",
                message_cache=message_cache,
                seed_list=seed_list,
            )
            if success:
                null_success += 1
        null_rate = null_success / n_episodes
        results["null_message"] = {
            "success_rate": null_rate,
            "successes": null_success,
            "n_episodes": n_episodes,
            "expected": config.p0_baseline,
        }
        print(
            f"  Null message: {null_rate:.1%} ({null_success}/{n_episodes}) [expected ~{config.p0_baseline:.0%}]"
        )

        # Random latent ablation
        random_success = 0
        for seed in seed_list:
            success = _run_codec_episode(
                seed=seed,
                model=model,
                tokenizer=tokenizer,
                codec=codec,
                device=device,
                temperature=config.temperature,
                ablation="random",
                message_cache=message_cache,
                seed_list=seed_list,
            )
            if success:
                random_success += 1
        random_rate = random_success / n_episodes
        results["random_latent"] = {
            "success_rate": random_rate,
            "successes": random_success,
            "n_episodes": n_episodes,
            "expected": config.p0_baseline,
        }
        print(
            f"  Random latent: {random_rate:.1%} ({random_success}/{n_episodes}) [expected ~{config.p0_baseline:.0%}]"
        )

        # P1 FIX: Shuffle ablation (REQUIRED - proves semantic dependence)
        shuffle_success = 0
        for seed in seed_list:
            success = _run_codec_episode(
                seed=seed,
                model=model,
                tokenizer=tokenizer,
                codec=codec,
                device=device,
                temperature=config.temperature,
                ablation="shuffle",
                message_cache=message_cache,
                seed_list=seed_list,
            )
            if success:
                shuffle_success += 1
        shuffle_rate = shuffle_success / n_episodes
        results["shuffle"] = {
            "success_rate": shuffle_rate,
            "successes": shuffle_success,
            "n_episodes": n_episodes,
            "expected": "below_p0",  # Should crater below P0
        }
        print(
            f"  Shuffle: {shuffle_rate:.1%} ({shuffle_success}/{n_episodes}) [expected < {config.p0_baseline:.0%}]"
        )

        # === GATE SWEEP: Test different scale_gate values ===
        # This reveals if we're capacity-limited (higher gate helps) or need better content
        print("\n--- Gate Sweep (testing different scale_gate values) ---")
        original_gate = codec.scale_gate.item()
        gate_values = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
        gate_results = {}

        for gate_val in gate_values:
            with torch.no_grad():
                codec.scale_gate.fill_(gate_val)

            gate_success = 0
            for seed in seed_list[:20]:  # Use subset for speed
                success = _run_codec_episode(
                    seed=seed,
                    model=model,
                    tokenizer=tokenizer,
                    codec=codec,
                    device=device,
                    temperature=config.temperature,
                    ablation="normal",
                    message_cache=message_cache,
                    seed_list=seed_list,
                )
                if success:
                    gate_success += 1
            gate_rate = gate_success / 20
            gate_results[gate_val] = gate_rate
            print(f"  scale_gate={gate_val:.2f}: {gate_rate:.1%} ({gate_success}/20)")

        # Restore original gate
        with torch.no_grad():
            codec.scale_gate.fill_(original_gate)

        results["gate_sweep"] = gate_results

        # === COMMUNICATION-MATTERS SUBSET ===
        # Report results on episodes where P1 succeeds but Null fails
        # These are the episodes where communication actually matters
        print("\n--- Communication-Matters Subset ---")
        print("(Episodes where text communication helps - i.e., P1-solvable)")

        # Count episodes where null failed but normal succeeded
        comm_matters_seeds = []
        for seed in seed_list:
            # Run null
            null_success = _run_codec_episode(
                seed=seed,
                model=model,
                tokenizer=tokenizer,
                codec=codec,
                device=device,
                temperature=config.temperature,
                ablation="null",
                message_cache=message_cache,
                seed_list=seed_list,
            )
            # Run normal (restore gate first)
            with torch.no_grad():
                codec.scale_gate.fill_(original_gate)
            normal_success = _run_codec_episode(
                seed=seed,
                model=model,
                tokenizer=tokenizer,
                codec=codec,
                device=device,
                temperature=config.temperature,
                ablation="normal",
                message_cache=message_cache,
                seed_list=seed_list,
            )
            if not null_success:  # Null failed = communication matters
                comm_matters_seeds.append((seed, normal_success))

        n_comm_matters = len(comm_matters_seeds)
        n_helped = sum(1 for _, success in comm_matters_seeds if success)
        if n_comm_matters > 0:
            help_rate = n_helped / n_comm_matters
            print(f"  Episodes where Null fails: {n_comm_matters}/{n_episodes}")
            print(
                f"  Normal success on Null-failing episodes: {help_rate:.1%} ({n_helped}/{n_comm_matters})"
            )
            results["comm_matters"] = {
                "n_episodes": n_comm_matters,
                "normal_successes": n_helped,
                "help_rate": help_rate,
            }
        else:
            print("  No episodes where Null fails (all solvable without communication)")
            results["comm_matters"] = {
                "n_episodes": 0,
                "note": "all solvable without communication",
            }

    return results


def _run_codec_episode(
    seed: int,
    model,
    tokenizer,
    codec,
    device: str,
    temperature: float,
    ablation: Optional[str] = None,
    message_cache: Optional[Dict[int, str]] = None,
    seed_list: Optional[List[int]] = None,
) -> bool:
    """
    Run a single codec-based episode with proper soft-prefix injection.

    ablation: None (normal), 'null' (zero latent), 'random' (random latent), 'shuffle' (wrong message)

    VALIDITY REQUIREMENTS:
    - Agent B prompt NEVER contains message_A for latent conditions
    - prefix_embedding is actually injected via inputs_embeds
    - Shuffle uses message from different episode (deterministic rotation)
    """
    import torch
    import re

    # P0 FIX: Fail fast if env not available (never return placeholder results)
    try:
        # Add both Modal path (/root) and local repo root to sys.path
        if "/root" not in sys.path:
            sys.path.insert(0, "/root")
        # For local evaluation, add the repo root (parent of modal/)
        repo_root = str(Path(__file__).parent.parent)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from lpca.envs.split_synthetic import SplitSyntheticEnv
    except ImportError as e:
        raise RuntimeError(
            f"LPCA env required for evaluation (install or mount lpca package): {e}"
        )

    env = SplitSyntheticEnv(difficulty="easy")
    env.select_environment("constraint_satisfaction")
    task = env.reset(seed)

    # Agent A: Generate message from obs_A
    # Use deterministic generation for paired comparisons
    torch.manual_seed(seed * 1000)  # Deterministic per seed
    prompt_A = f"""You are Agent A in a collaborative task.

Your observation:
{task.obs_A}

Send a MESSAGE to your partner to help solve the task together.
Start with "MESSAGE:" followed by the key information from your observation."""

    inputs_A = tokenizer(prompt_A, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs_A = model.generate(
            **inputs_A,
            max_new_tokens=256,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    response_A = tokenizer.decode(
        outputs_A[0][inputs_A["input_ids"].shape[1] :], skip_special_tokens=True
    )

    # Extract message
    message_A = response_A
    if "MESSAGE:" in response_A:
        message_A = response_A.split("MESSAGE:")[-1].strip()

    if not message_A or len(message_A.strip()) == 0:
        message_A = "No information available"

    # P2 FIX: Cache message for paired comparisons
    if message_cache is not None:
        if seed in message_cache:
            message_A = message_cache[seed]  # Reuse cached message
        else:
            message_cache[seed] = message_A  # Cache for other conditions

    # P1 FIX: Shuffle uses message from different episode
    message_for_encoding = message_A
    if ablation == "shuffle" and seed_list is not None:
        idx = seed_list.index(seed)
        shuffled_idx = (idx + 17) % len(seed_list)  # Rotate by 17 in index space
        shuffled_seed = seed_list[shuffled_idx]
        if message_cache is not None and shuffled_seed in message_cache:
            message_for_encoding = message_cache[shuffled_seed]
        else:
            # Generate message for shuffled seed (should already be cached)
            message_for_encoding = f"Shuffled placeholder {shuffled_seed}"

    # Encode message using codec
    msg_inputs = tokenizer(
        message_for_encoding, return_tensors="pt", truncation=True, max_length=256
    ).to(device)
    with torch.no_grad():
        msg_outputs = model(**msg_inputs, output_hidden_states=True)
        hidden = msg_outputs.hidden_states[-1]
        # Use last-token pooling for better discrimination
        pooled = get_last_token_embedding(hidden, msg_inputs["attention_mask"])

        # Apply ablation to latent/prefix
        # CRITICAL: null must use actual zeros (not decode(0) which has LayerNorm bias)
        # This matches training where null_prefix = torch.zeros_like(prefix_embeddings)
        if ablation == "null":
            # Direct zero embeddings - matches training null exactly
            prefix_embeddings = torch.zeros(1, codec.k, codec.d, device=device)
        elif ablation == "random":
            # Random latent through decoder (tests if decoder structure matters)
            latent = torch.randn(1, codec.k, codec.d, device=device)
            prefix_embeddings = codec.decode(latent)
        else:
            # Normal or shuffle: use actual codec encoding + decoding
            latent = codec.encode(pooled.float())
            prefix_embeddings = codec.decode(latent)

        k_prefix = codec.k

    # P1 FIX: Agent B prompt NEVER contains message_A for latent conditions
    # This is the NO TEXT LEAK guarantee
    prompt_B = f"""You are Agent B in a collaborative task.

Your observation:
{task.obs_B}

Your partner sent you encoded information (injected as a learned prefix).
Based on your observation and the encoded context, provide an ANSWER.

ANSWER in format {{"x1": <value>, "x2": <value>, "x3": <value>, "x4": <value>}}:"""

    # P1 FIX: Actually inject prefix_embeddings via inputs_embeds
    inputs_B = tokenizer(prompt_B, return_tensors="pt").to(device)

    # Get input embeddings for prompt
    input_embeds_B = model.get_input_embeddings()(inputs_B["input_ids"])

    # Cast k prefix embeddings to model dtype (float16 on CUDA)
    soft_prefix = prefix_embeddings.to(input_embeds_B.dtype)  # (1, k, d_model)

    # Concatenate: [soft_prefix (k tokens), prompt_embeddings]
    combined_embeds = torch.cat([soft_prefix, input_embeds_B], dim=1)

    # Create attention mask for combined input (k prefix tokens + prompt)
    soft_mask = torch.ones(
        1, k_prefix, device=device, dtype=inputs_B["attention_mask"].dtype
    )
    combined_mask = torch.cat([soft_mask, inputs_B["attention_mask"]], dim=1)

    # P2 FIX: Deterministic generation for Agent B (paired comparisons)
    torch.manual_seed(seed * 1000 + 1)  # Different seed than Agent A

    # Generate with injected embeddings using past_key_values to maintain context
    with torch.no_grad():
        # First forward pass with embeddings to get initial logits AND cache
        outputs_B = model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            use_cache=True,
        )
        logits = outputs_B.logits
        past_key_values = outputs_B.past_key_values

        # Sample first token
        probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Continue generation autoregressively with KV cache
        generated_ids = [next_token.item()]
        current_ids = next_token

        for _ in range(127):  # max_new_tokens - 1
            # Pass past_key_values to maintain injected context
            # Attention mask must cover all positions: prefix + prompt + generated tokens
            current_seq_len = combined_mask.shape[1] + len(generated_ids)
            extended_mask = torch.ones(
                1, current_seq_len, device=device, dtype=combined_mask.dtype
            )

            outputs = model(
                input_ids=current_ids,
                attention_mask=extended_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values

            next_probs = torch.softmax(outputs.logits[:, -1, :] / temperature, dim=-1)
            next_token = torch.multinomial(next_probs, num_samples=1)
            generated_ids.append(next_token.item())
            current_ids = next_token

            # Stop at EOS
            if next_token.item() == tokenizer.eos_token_id:
                break

    response_B = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Extract answer
    answer = None
    json_match = re.search(r"\{[^}]+\}", response_B)
    if json_match:
        try:
            answer = json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    if answer is None:
        return False

    # Verify
    result = env.verify(answer)
    return result.success


def _run_text_baseline_episode(
    seed: int,
    model,
    tokenizer,
    device: str,
    temperature: float,
) -> bool:
    """
    Run P1 baseline: Agent B sees message_A in text form (no injection).
    Used for plumbing proof comparison.
    """
    import torch
    import re

    try:
        # Add both Modal path (/root) and local repo root to sys.path
        if "/root" not in sys.path:
            sys.path.insert(0, "/root")
        repo_root = str(Path(__file__).parent.parent)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from lpca.envs.split_synthetic import SplitSyntheticEnv
    except ImportError as e:
        raise RuntimeError(f"LPCA env required for evaluation: {e}")

    env = SplitSyntheticEnv(difficulty="easy")
    env.select_environment("constraint_satisfaction")
    task = env.reset(seed)

    # Agent A: Generate message (deterministic)
    torch.manual_seed(seed * 1000)
    prompt_A = f"""You are Agent A in a collaborative task.

Your observation:
{task.obs_A}

Send a MESSAGE to your partner to help solve the task together.
Start with "MESSAGE:" followed by the key information from your observation."""

    inputs_A = tokenizer(prompt_A, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs_A = model.generate(
            **inputs_A,
            max_new_tokens=256,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    response_A = tokenizer.decode(
        outputs_A[0][inputs_A["input_ids"].shape[1] :], skip_special_tokens=True
    )
    message_A = response_A
    if "MESSAGE:" in message_A:
        message_A = message_A.split("MESSAGE:", 1)[1].strip()

    # Agent B: Sees message_A in text (P1 baseline)
    torch.manual_seed(seed * 1000 + 1)
    prompt_B = f"""You are Agent B in a collaborative task.

Your observation:
{task.obs_B}

MESSAGE from your partner:
{message_A}

Based on your observation and your partner's message, provide an ANSWER.

ANSWER in format {{"x1": <value>, "x2": <value>, "x3": <value>, "x4": <value>}}:"""

    inputs_B = tokenizer(prompt_B, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs_B = model.generate(
            **inputs_B,
            max_new_tokens=128,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    response_B = tokenizer.decode(
        outputs_B[0][inputs_B["input_ids"].shape[1] :], skip_special_tokens=True
    )

    # Extract answer
    answer = None
    json_match = re.search(r"\{[^}]+\}", response_B)
    if json_match:
        try:
            import json

            answer = json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    if answer is None:
        return False

    result = env.verify(answer)
    return result.success


def run_plumbing_proof(
    checkpoint_path: str,
    config: M2Config,
    device: str = "cuda",
    n_episodes: int = 10,
) -> Dict[str, Any]:
    """
    Run plumbing proof to verify injection actually influences generation.

    Compares:
    1. P1 baseline: Text present, no injection (expect ~68% from E1)
    2. L1 injection: Text absent, injection on (measures injection effectiveness)

    Decision rule:
    - If L1 ≈ P1 AND no-leak check passes → injection works (replacing text function)
    - If L1 collapses to ~P0 (~20%) → injection not influencing generation → fix before proceeding
    """
    import torch

    print("\n" + "=" * 60)
    print("PLUMBING PROOF: Verifying Injection Effectiveness")
    print("=" * 60)

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer

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

    # Load codec
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    k_vectors = checkpoint["k_vectors"]
    d_model = checkpoint["d_model"]
    hidden_dim = checkpoint["hidden_dim"]

    codec = create_codec_model(
        d_model=d_model,
        k_vectors=k_vectors,
        hidden_dim=hidden_dim,
        codec_variant=checkpoint.get("codec_variant", config.codec_variant),
        codebook_size=checkpoint.get("codebook_size", config.codebook_size),
        vq_beta=checkpoint.get("vq_beta", config.vq_beta),
    )
    # Handle loading old checkpoints that don't have prefix_norm/scale_gate
    try:
        codec.load_state_dict(checkpoint["state_dict"])
    except RuntimeError:
        print("  Note: Loading old checkpoint format, using default calibration")
        codec.load_state_dict(checkpoint["state_dict"], strict=False)
    codec = codec.to(device)
    codec.eval()

    # Set target norm from model's token embeddings
    with torch.no_grad():
        token_embeds = model.get_input_embeddings().weight
        target_norm = token_embeds.norm(dim=1).mean().item()
        codec.target_norm.fill_(target_norm)

    print(f"Loaded codec: k={k_vectors}, d_model={d_model}")
    print(f"Testing on {n_episodes} episodes")

    # Use fixed test seeds
    test_seeds = list(range(3000, 3000 + n_episodes))

    # Condition 1: P1 baseline (text present, no injection)
    print("\n--- Condition 1: P1 Baseline (text present, no injection) ---")
    p1_successes = 0
    for seed in test_seeds:
        success = _run_text_baseline_episode(
            seed=seed,
            model=model,
            tokenizer=tokenizer,
            device=device,
            temperature=config.temperature,
        )
        p1_successes += int(success)
        print(f"  Seed {seed}: {'✓' if success else '✗'}")

    p1_rate = p1_successes / n_episodes
    p1_ci = wilson_ci(p1_successes, n_episodes)
    print(
        f"P1 Baseline: {p1_successes}/{n_episodes} = {p1_rate:.1%} (95% CI: [{p1_ci[0]:.1%}, {p1_ci[1]:.1%}])"
    )

    # Condition 2: L1 injection (text absent, injection on)
    print("\n--- Condition 2: L1 Injection (text absent, injection on) ---")
    l1_successes = 0
    for seed in test_seeds:
        success = _run_codec_episode(
            seed=seed,
            model=model,
            tokenizer=tokenizer,
            codec=codec,
            device=device,
            temperature=config.temperature,
            ablation=None,  # Normal injection
            message_cache={},  # Fresh cache
            seed_list=test_seeds,
        )
        l1_successes += int(success)
        print(f"  Seed {seed}: {'✓' if success else '✗'}")

    l1_rate = l1_successes / n_episodes
    l1_ci = wilson_ci(l1_successes, n_episodes)
    print(
        f"L1 Injection: {l1_successes}/{n_episodes} = {l1_rate:.1%} (95% CI: [{l1_ci[0]:.1%}, {l1_ci[1]:.1%}])"
    )

    # Decision
    print("\n" + "=" * 60)
    print("PLUMBING PROOF DECISION")
    print("=" * 60)

    # P0 reference: ~20% from E1 results
    p0_reference = 0.20
    p0_ci_approx = (0.11, 0.33)  # From E1 n=50 results

    # Check if L1 is closer to P1 or P0
    l1_vs_p1_gap = abs(l1_rate - p1_rate)
    l1_vs_p0_gap = abs(l1_rate - p0_reference)

    print(f"\nP1 Baseline:    {p1_rate:.1%} (expected ~68%)")
    print(f"L1 Injection:   {l1_rate:.1%}")
    print(f"P0 Reference:   {p0_reference:.1%} (from E1)")
    print(f"\nGap L1-P1: {l1_vs_p1_gap:.1%}")
    print(f"Gap L1-P0: {l1_vs_p0_gap:.1%}")

    injection_works = l1_vs_p0_gap > l1_vs_p1_gap and l1_rate > p0_ci_approx[1]

    if injection_works:
        print("\n✅ PASS: Injection appears to influence generation")
        print("   L1 is closer to P1 than P0, suggesting codec conveys information")
        print("   Proceed to full Gate 1 evaluation")
    else:
        print("\n❌ FAIL: Injection may not be working")
        print(
            "   L1 is closer to P0 than P1, suggesting codec is not conveying information"
        )
        print("   Check: Is prefix_embedding being used? Is KV cache working?")

    return {
        "p1_baseline": {
            "successes": p1_successes,
            "n": n_episodes,
            "rate": p1_rate,
            "ci": p1_ci,
        },
        "l1_injection": {
            "successes": l1_successes,
            "n": n_episodes,
            "rate": l1_rate,
            "ci": l1_ci,
        },
        "p0_reference": p0_reference,
        "injection_works": injection_works,
        "l1_vs_p1_gap": l1_vs_p1_gap,
        "l1_vs_p0_gap": l1_vs_p0_gap,
    }


def wilson_ci(successes: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Wilson score confidence interval for binomial proportion.
    More accurate than normal approximation, especially for extreme proportions.
    """
    from scipy import stats

    if n == 0:
        return (0.0, 1.0)

    p_hat = successes / n
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    z2 = z * z

    denominator = 1 + z2 / n
    centre = (p_hat + z2 / (2 * n)) / denominator
    margin = z * ((p_hat * (1 - p_hat) / n + z2 / (4 * n * n)) ** 0.5) / denominator

    return (max(0.0, centre - margin), min(1.0, centre + margin))


def ci_overlap(ci1: Tuple[float, float], ci2: Tuple[float, float]) -> bool:
    """Check if two confidence intervals overlap."""
    return ci1[0] <= ci2[1] and ci2[0] <= ci1[1]


def check_gates(results: Dict[str, Any], config: M2Config) -> Dict[str, bool]:
    """
    Check if training passes the pre-committed gates using CI-aware criteria.

    VALIDITY REQUIREMENTS:
    - Gate passes only if CI_low >= threshold (not point estimate)
    - Ablations pass if CI overlaps P0 CI (null, random) or CI_high < CI_low(P0) (shuffle)
    """
    gates = {}

    # Get results with counts for CI calculation
    normal = results.get("normal", {})
    normal_sr = normal.get("success_rate", 0)
    normal_n = normal.get("n_episodes", 50)
    normal_successes = normal.get("successes", int(normal_sr * normal_n))

    # P2 FIX: CI-aware gating
    normal_ci = wilson_ci(normal_successes, normal_n)
    print(
        f"\nNormal: {normal_sr:.1%} (95% CI: [{normal_ci[0]:.1%}, {normal_ci[1]:.1%}])"
    )

    # Gate 1: Sanity - CI_low must meet threshold
    gates["gate1_sanity"] = normal_ci[0] >= config.gate1_threshold
    print(
        f"Gate 1 (Sanity): CI_low {normal_ci[0]:.1%} >= {config.gate1_threshold:.1%}? "
        f"{'PASS' if gates['gate1_sanity'] else 'FAIL'}"
    )

    # Gate 2: Retention (only check at k>=16)
    if config.k_vectors >= 16:
        gates["gate2_retention"] = normal_ci[0] >= config.gate2_threshold
        print(
            f"Gate 2 (Retention): CI_low {normal_ci[0]:.1%} >= {config.gate2_threshold:.1%}? "
            f"{'PASS' if gates['gate2_retention'] else 'FAIL'}"
        )

    # P0 baseline CI (for ablation comparison)
    # Using hypothetical n=50 at p=0.20 for P0 reference
    p0_ci = wilson_ci(int(config.p0_baseline * normal_n), normal_n)
    print(
        f"\nP0 baseline reference: {config.p0_baseline:.1%} (CI: [{p0_ci[0]:.1%}, {p0_ci[1]:.1%}])"
    )

    # Ablation checks with CI-aware criteria
    print("\nAblation checks:")

    # Null ablation: CI should overlap P0 CI
    null = results.get("null_message", {})
    null_sr = null.get("success_rate", 0)
    null_n = null.get("n_episodes", normal_n)
    null_successes = null.get("successes", int(null_sr * null_n))
    null_ci = wilson_ci(null_successes, null_n)
    gates["ablation_null"] = ci_overlap(null_ci, p0_ci)
    print(
        f"  Null: {null_sr:.1%} (CI: [{null_ci[0]:.1%}, {null_ci[1]:.1%}]) "
        f"overlaps P0? {'OK' if gates['ablation_null'] else 'UNEXPECTED'}"
    )

    # Random ablation: CI should overlap P0 CI
    rand = results.get("random_latent", {})
    random_sr = rand.get("success_rate", 0)
    random_n = rand.get("n_episodes", normal_n)
    random_successes = rand.get("successes", int(random_sr * random_n))
    random_ci = wilson_ci(random_successes, random_n)
    gates["ablation_random"] = ci_overlap(random_ci, p0_ci)
    print(
        f"  Random: {random_sr:.1%} (CI: [{random_ci[0]:.1%}, {random_ci[1]:.1%}]) "
        f"overlaps P0? {'OK' if gates['ablation_random'] else 'UNEXPECTED'}"
    )

    # Shuffle ablation: CI_high must be < CI_low(P0) (proves semantic dependence)
    shuffle = results.get("shuffle", {})
    if shuffle:
        shuffle_sr = shuffle.get("success_rate", 0)
        shuffle_n = shuffle.get("n_episodes", normal_n)
        shuffle_successes = shuffle.get("successes", int(shuffle_sr * shuffle_n))
        shuffle_ci = wilson_ci(shuffle_successes, shuffle_n)
        # Shuffle should crater: CI_high(shuffle) < CI_low(P0)
        gates["ablation_shuffle"] = shuffle_ci[1] < p0_ci[0]
        print(
            f"  Shuffle: {shuffle_sr:.1%} (CI: [{shuffle_ci[0]:.1%}, {shuffle_ci[1]:.1%}]) "
            f"CI_high < P0_CI_low? {'OK' if gates['ablation_shuffle'] else 'UNEXPECTED'}"
        )
    else:
        gates["ablation_shuffle"] = False
        print("  Shuffle: NOT RUN (required ablation missing)")

    # Overall validity
    all_ablations_pass = (
        gates.get("ablation_null", False)
        and gates.get("ablation_random", False)
        and gates.get("ablation_shuffle", False)
    )
    gates["all_ablations_valid"] = all_ablations_pass

    print(f"\nOverall ablation validity: {'PASS' if all_ablations_pass else 'FAIL'}")

    return gates


# ============================================================================
# Modal App Definition
# ============================================================================

if MODAL_AVAILABLE:
    app = modal.App("m2-codec-training")

    # Persistent volume for training data
    volume = modal.Volume.from_name("m2-training-data", create_if_missing=True)

    # Define the image with dependencies and lpca package
    image = (
        modal.Image.debian_slim(python_version="3.10")
        .pip_install(
            "torch>=2.0.0",
            "transformers>=4.35.0",
            "accelerate>=0.20.0",
            "tqdm",
            "numpy",
            "scipy",
        )
        .add_local_dir("lpca", remote_path="/root/lpca")
    )

    @app.function(
        gpu="A100",
        timeout=7200,  # 2 hours
        image=image,
        volumes={"/data": volume},
    )
    def train_on_modal(
        k: int,
        epochs: int,
        data_path: str = "/data/m2_train.jsonl",
        eval_only: str = None,
        eval_episodes: int = 50,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        base_seed: int = 1000,
        codec_variant: str = "legacy",
        codebook_size: int = 256,
        vq_beta: float = 0.25,
        curriculum_warmup_steps: int = 500,
        curriculum_ramp_steps: int = 1500,
    ) -> Dict:
        """Train codec on Modal A100, or run eval-only on existing checkpoint."""
        eval_episodes = max(1, eval_episodes)
        config = M2Config(
            model_name=model_name,
            k_vectors=k,
            epochs=epochs,
            train_data_path=data_path,
            eval_episodes=eval_episodes,
            test_seeds=(base_seed, base_seed + eval_episodes),
            val_seeds=(base_seed + 100000, base_seed + 100000 + eval_episodes),
            codec_variant=codec_variant,
            codebook_size=codebook_size,
            vq_beta=vq_beta,
            curriculum_warmup_steps=curriculum_warmup_steps,
            curriculum_ramp_steps=curriculum_ramp_steps,
            base_seed=base_seed,
        )

        if eval_only:
            # Eval-only mode
            eval_result = evaluate_codec(
                eval_only,
                config,
                device="cuda",
            )
            gates = check_gates(eval_result, config)
            return {
                "k": k,
                "eval": eval_result,
                "gates": gates,
                "eval_only": True,
            }

        # Train
        train_result = train_codec(config, device="cuda")

        # Evaluate
        eval_result = evaluate_codec(
            train_result["checkpoint_path"],
            config,
            device="cuda",
        )

        # Check gates
        gates = check_gates(eval_result, config)

        return {
            "k": k,
            "train": train_result,
            "eval": eval_result,
            "gates": gates,
        }


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="M2-SCALE Codec Training")
    parser.add_argument("--k", type=int, default=16, help="Number of latent vectors")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--data", type=str, default="data/m2_train.jsonl")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Backbone model id",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=1000,
        help="Base seed for deterministic eval seed ranges",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path for final eval JSON artifact",
    )
    parser.add_argument(
        "--codec-variant",
        type=str,
        default="legacy",
        choices=["legacy", "m2v2"],
        help="Codec architecture variant",
    )
    parser.add_argument(
        "--codebook-size",
        type=int,
        default=256,
        help="VQ codebook size for m2v2 variant",
    )
    parser.add_argument(
        "--vq-beta",
        type=float,
        default=0.25,
        help="VQ commitment weight for m2v2 variant",
    )
    parser.add_argument(
        "--curriculum-warmup-steps",
        type=int,
        default=500,
        help="Curriculum warmup steps for m2v2 variant",
    )
    parser.add_argument(
        "--curriculum-ramp-steps",
        type=int,
        default=1500,
        help="Curriculum semantic ramp steps for m2v2 variant",
    )
    parser.add_argument(
        "--local", action="store_true", help="Run locally (not on Modal)"
    )
    parser.add_argument("--sweep", action="store_true", help="Run full k sweep")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument(
        "--plumbing-proof",
        type=str,
        metavar="CHECKPOINT",
        help="Run plumbing proof on a trained checkpoint to verify injection works",
    )
    parser.add_argument(
        "--plumbing-episodes",
        type=int,
        default=10,
        help="Number of episodes for plumbing proof (default: 10)",
    )
    parser.add_argument(
        "--diagnostics",
        type=str,
        metavar="CHECKPOINT",
        help="Run diagnostic checks on a checkpoint (prefix collapse + semantic check)",
    )
    parser.add_argument(
        "--diagnostics-samples",
        type=int,
        default=20,
        help="Number of samples for diagnostics (default: 20)",
    )
    parser.add_argument(
        "--eval-only",
        type=str,
        metavar="CHECKPOINT",
        help="Run evaluation only on a checkpoint (no training)",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=50,
        help="Number of episodes for evaluation (default: 50)",
    )

    args = parser.parse_args()

    def build_config(
        *,
        k_vectors: Optional[int] = None,
        epochs: Optional[int] = None,
        eval_episodes: Optional[int] = None,
        output_path: Optional[str] = None,
    ) -> M2Config:
        eval_eps = args.eval_episodes if eval_episodes is None else eval_episodes
        eval_eps = max(1, eval_eps)
        base_seed = int(args.base_seed)
        return M2Config(
            model_name=args.model,
            k_vectors=args.k if k_vectors is None else k_vectors,
            epochs=args.epochs if epochs is None else epochs,
            train_data_path=args.data,
            eval_episodes=eval_eps,
            test_seeds=(base_seed, base_seed + eval_eps),
            val_seeds=(base_seed + 100000, base_seed + 100000 + eval_eps),
            codec_variant=args.codec_variant,
            codebook_size=args.codebook_size,
            vq_beta=args.vq_beta,
            curriculum_warmup_steps=args.curriculum_warmup_steps,
            curriculum_ramp_steps=args.curriculum_ramp_steps,
            base_seed=base_seed,
            output_path=output_path if output_path is not None else args.output,
        )

    def write_json(path: str, payload: Dict[str, Any]) -> None:
        out = Path(path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"\nResults saved to: {out}")

    def output_for_k(k: int) -> Optional[str]:
        if not args.output:
            return None
        out = Path(args.output)
        suffix = out.suffix if out.suffix else ".json"
        return str(out.with_name(f"{out.stem}_k{k}{suffix}"))

    # Plumbing proof mode
    if args.plumbing_proof:
        config = build_config(k_vectors=args.k)
        result = run_plumbing_proof(
            checkpoint_path=args.plumbing_proof,
            config=config,
            device=args.device,
            n_episodes=args.plumbing_episodes,
        )

        proof_path = args.output or args.plumbing_proof.replace(
            ".pt", "_plumbing_proof.json"
        )
        # Convert tuples to lists for JSON
        result_json = {
            "p1_baseline": {
                **result["p1_baseline"],
                "ci": list(result["p1_baseline"]["ci"]),
            },
            "l1_injection": {
                **result["l1_injection"],
                "ci": list(result["l1_injection"]["ci"]),
            },
            "p0_reference": result["p0_reference"],
            "injection_works": result["injection_works"],
            "l1_vs_p1_gap": result["l1_vs_p1_gap"],
            "l1_vs_p0_gap": result["l1_vs_p0_gap"],
        }
        write_json(proof_path, result_json)
        return

    # Diagnostics mode
    if args.diagnostics:
        config = build_config(k_vectors=args.k)
        result = run_diagnostics(
            checkpoint_path=args.diagnostics,
            config=config,
            device=args.device,
            n_samples=args.diagnostics_samples,
        )

        diag_path = args.output or args.diagnostics.replace(".pt", "_diagnostics.json")
        write_json(diag_path, result)
        return

    # Eval-only mode
    if args.eval_only:
        if args.local:
            config = build_config(k_vectors=args.k, eval_episodes=args.eval_episodes)
            eval_result = evaluate_codec(
                checkpoint_path=args.eval_only,
                config=config,
                device=args.device,
            )
            gates = check_gates(eval_result, config)

            eval_path = args.output or args.eval_only.replace(".pt", "_eval.json")
            write_json(
                eval_path,
                {
                    "eval": eval_result,
                    "gates": gates,
                    "config": asdict(config),
                },
            )
        else:
            # Run on Modal
            if not MODAL_AVAILABLE:
                print("Modal not available. Install with: pip install modal")
                sys.exit(1)
            with app.run():
                result = train_on_modal.remote(
                    args.k,
                    0,
                    "/data/m2_train.jsonl",
                    eval_only=args.eval_only,
                    eval_episodes=args.eval_episodes,
                    model_name=args.model,
                    base_seed=args.base_seed,
                    codec_variant=args.codec_variant,
                    codebook_size=args.codebook_size,
                    vq_beta=args.vq_beta,
                    curriculum_warmup_steps=args.curriculum_warmup_steps,
                    curriculum_ramp_steps=args.curriculum_ramp_steps,
                )
                print(f"\nResult: {result}")
                eval_path = args.output or args.eval_only.replace(".pt", "_eval.json")
                write_json(
                    eval_path,
                    {
                        **result,
                        "config": asdict(
                            build_config(
                                k_vectors=args.k,
                                eval_episodes=args.eval_episodes,
                            )
                        ),
                    },
                )
        return

    if args.sweep:
        k_values = [4, 8, 16, 32, 64]
        print(f"Running k sweep: {k_values}")

        if args.local:
            for k in k_values:
                config = build_config(
                    k_vectors=k,
                    epochs=args.epochs,
                    output_path=output_for_k(k),
                )
                train_result = train_codec(config, device=args.device)
                eval_result = evaluate_codec(
                    train_result["checkpoint_path"], config, device=args.device
                )
                gates = check_gates(eval_result, config)
                eval_path = config.output_path or train_result[
                    "checkpoint_path"
                ].replace(".pt", "_eval.json")
                write_json(
                    eval_path,
                    {
                        "train": train_result,
                        "eval": eval_result,
                        "gates": gates,
                        "config": asdict(config),
                    },
                )
        else:
            if not MODAL_AVAILABLE:
                print("Modal not available. Install with: pip install modal")
                sys.exit(1)
            # Run on Modal in parallel (use Modal volume path)
            with app.run():
                results = []
                for k in k_values:
                    result = train_on_modal.remote(
                        k,
                        args.epochs,
                        "/data/m2_train.jsonl",
                        model_name=args.model,
                        base_seed=args.base_seed,
                        codec_variant=args.codec_variant,
                        codebook_size=args.codebook_size,
                        vq_beta=args.vq_beta,
                        curriculum_warmup_steps=args.curriculum_warmup_steps,
                        curriculum_ramp_steps=args.curriculum_ramp_steps,
                    )
                    results.append(result)
                # Collect results
                for r in results:
                    print(f"\nk={r['k']}: {r['gates']}")
                if args.output:
                    write_json(
                        args.output,
                        {
                            "results": results,
                            "config": asdict(build_config()),
                        },
                    )

    else:
        # Single k training
        config = build_config(k_vectors=args.k, epochs=args.epochs)

        if args.local:
            train_result = train_codec(config, device=args.device)
            eval_result = evaluate_codec(
                train_result["checkpoint_path"], config, device=args.device
            )
            gates = check_gates(eval_result, config)
            eval_path = config.output_path or train_result["checkpoint_path"].replace(
                ".pt", "_eval.json"
            )
            write_json(
                eval_path,
                {
                    "train": train_result,
                    "eval": eval_result,
                    "gates": gates,
                    "config": asdict(config),
                },
            )
        else:
            if not MODAL_AVAILABLE:
                print("Modal not available. Install with: pip install modal")
                sys.exit(1)
            with app.run():
                # Use Modal volume path, not local path
                result = train_on_modal.remote(
                    args.k,
                    args.epochs,
                    "/data/m2_train.jsonl",
                    model_name=args.model,
                    base_seed=args.base_seed,
                    codec_variant=args.codec_variant,
                    codebook_size=args.codebook_size,
                    vq_beta=args.vq_beta,
                    curriculum_warmup_steps=args.curriculum_warmup_steps,
                    curriculum_ramp_steps=args.curriculum_ramp_steps,
                )
                print(f"\nResult: {result}")
                if args.output:
                    write_json(
                        args.output,
                        {
                            **result,
                            "config": asdict(build_config()),
                        },
                    )


if __name__ == "__main__":
    main()
