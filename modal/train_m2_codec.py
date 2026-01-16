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

    # Training loop with KL distillation
    # Goal: Train codec so Agent B with prefix behaves like Agent B with text message
    history = {'loss': [], 'kl_loss': [], 'epoch': []}

    print(f"\nTraining for {config.epochs} epochs (KL distillation)...")
    for epoch in range(config.epochs):
        codec.train()
        epoch_loss = 0
        epoch_kl = 0
        n_batches = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.epochs}"):
            batch_loss = 0
            batch_kl = 0
            n_samples = 0

            for sample in batch:
                receiver_context = sample['receiver_context']
                sender_message = sample['sender_message']

                # === TEACHER: Agent B with full text message ===
                teacher_prompt = f"""{receiver_context}

MESSAGE from your partner:
{sender_message}

Based on your constraints and your partner's message, provide an ANSWER.
ANSWER:"""

                # === STUDENT: Agent B with no text (will get prefix) ===
                student_prompt = f"""{receiver_context}

Your partner sent you encoded information (injected as a learned prefix).
Based on your constraints and the encoded context, provide an ANSWER.
ANSWER:"""

                # Tokenize
                teacher_inputs = tokenizer(
                    teacher_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                ).to(device)

                student_inputs = tokenizer(
                    student_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                ).to(device)

                # Encode sender message for codec input
                msg_inputs = tokenizer(
                    sender_message,
                    return_tensors="pt",
                    truncation=True,
                    max_length=256,
                ).to(device)

                # Get message embedding (pooled last hidden state)
                with torch.no_grad():
                    msg_outputs = model(**msg_inputs, output_hidden_states=True)
                    msg_hidden = msg_outputs.hidden_states[-1]
                    msg_mask = msg_inputs['attention_mask'].unsqueeze(-1).float()
                    msg_pooled = (msg_hidden * msg_mask).sum(dim=1) / msg_mask.sum(dim=1)

                # Encode message through codec → prefix embedding
                latent, prefix_embedding = codec(msg_pooled)  # prefix_embedding: (1, d_model)

                # Get student embeddings and prepend prefix
                student_embeds = model.get_input_embeddings()(student_inputs['input_ids'])
                # Cast prefix to match model dtype (float16 on CUDA)
                soft_prefix = prefix_embedding.unsqueeze(1).to(student_embeds.dtype)  # (1, 1, d_model)
                combined_embeds = torch.cat([soft_prefix, student_embeds], dim=1)

                # Attention mask for combined input
                soft_mask = torch.ones(1, 1, device=device, dtype=student_inputs['attention_mask'].dtype)
                combined_mask = torch.cat([soft_mask, student_inputs['attention_mask']], dim=1)

                # Forward passes
                with torch.no_grad():
                    # Teacher forward (frozen, just get logits)
                    teacher_outputs = model(**teacher_inputs)
                    teacher_logits = teacher_outputs.logits[:, -1, :]  # Last token prediction

                # Student forward (gradient flows through codec)
                student_outputs = model(
                    inputs_embeds=combined_embeds,
                    attention_mask=combined_mask,
                )
                student_logits = student_outputs.logits[:, -1, :]  # Last token prediction

                # KL divergence loss: student should match teacher distribution
                # Use temperature for softer distributions
                temperature = 2.0
                teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
                student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
                kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')

                # Scale by temperature^2 as per distillation convention
                kl_loss = kl_loss * (temperature ** 2)

                batch_loss += kl_loss
                batch_kl += kl_loss.item()
                n_samples += 1

            # Average over batch and backprop
            if n_samples > 0:
                loss = batch_loss / n_samples
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_kl += batch_kl / n_samples
                n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        avg_kl = epoch_kl / max(n_batches, 1)
        history['loss'].append(avg_loss)
        history['kl_loss'].append(avg_kl)
        history['epoch'].append(epoch + 1)

        print(f"  Epoch {epoch+1}: loss={avg_loss:.6f}, kl={avg_kl:.6f}")

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
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\n" + "=" * 60)
    print("EVALUATION (with validity checks)")
    print("=" * 60)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    k_vectors = checkpoint['k_vectors']
    d_model = checkpoint['d_model']
    hidden_dim = checkpoint['hidden_dim']

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
    codec = create_codec_model(d_model, k_vectors, hidden_dim)
    codec.load_state_dict(checkpoint['state_dict'])
    codec = codec.to(device)
    codec.eval()

    results = {}

    # Test on fresh seeds
    test_start, test_end = config.test_seeds
    n_episodes = min(config.eval_episodes, test_end - test_start)
    seed_list = list(range(test_start, test_start + n_episodes))

    print(f"\nEvaluating on {n_episodes} fresh episodes (seeds {test_start}-{test_start+n_episodes-1})")

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
    results['normal'] = {
        'success_rate': success_rate,
        'successes': success_count,
        'n_episodes': n_episodes,
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
                ablation='null',
                message_cache=message_cache,
                seed_list=seed_list,
            )
            if success:
                null_success += 1
        null_rate = null_success / n_episodes
        results['null_message'] = {
            'success_rate': null_rate,
            'successes': null_success,
            'n_episodes': n_episodes,
            'expected': config.p0_baseline,
        }
        print(f"  Null message: {null_rate:.1%} ({null_success}/{n_episodes}) [expected ~{config.p0_baseline:.0%}]")

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
                ablation='random',
                message_cache=message_cache,
                seed_list=seed_list,
            )
            if success:
                random_success += 1
        random_rate = random_success / n_episodes
        results['random_latent'] = {
            'success_rate': random_rate,
            'successes': random_success,
            'n_episodes': n_episodes,
            'expected': config.p0_baseline,
        }
        print(f"  Random latent: {random_rate:.1%} ({random_success}/{n_episodes}) [expected ~{config.p0_baseline:.0%}]")

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
                ablation='shuffle',
                message_cache=message_cache,
                seed_list=seed_list,
            )
            if success:
                shuffle_success += 1
        shuffle_rate = shuffle_success / n_episodes
        results['shuffle'] = {
            'success_rate': shuffle_rate,
            'successes': shuffle_success,
            'n_episodes': n_episodes,
            'expected': 'below_p0',  # Should crater below P0
        }
        print(f"  Shuffle: {shuffle_rate:.1%} ({shuffle_success}/{n_episodes}) [expected < {config.p0_baseline:.0%}]")

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
        if '/root' not in sys.path:
            sys.path.insert(0, '/root')
        from lpca.envs.split_synthetic import SplitSyntheticEnv
    except ImportError as e:
        raise RuntimeError(f"LPCA env required for evaluation (install or mount lpca package): {e}")

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
    response_A = tokenizer.decode(outputs_A[0][inputs_A['input_ids'].shape[1]:], skip_special_tokens=True)

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
    if ablation == 'shuffle' and seed_list is not None:
        idx = seed_list.index(seed)
        shuffled_idx = (idx + 17) % len(seed_list)  # Rotate by 17 in index space
        shuffled_seed = seed_list[shuffled_idx]
        if message_cache is not None and shuffled_seed in message_cache:
            message_for_encoding = message_cache[shuffled_seed]
        else:
            # Generate message for shuffled seed (should already be cached)
            message_for_encoding = f"Shuffled placeholder {shuffled_seed}"

    # Encode message using codec
    msg_inputs = tokenizer(message_for_encoding, return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        msg_outputs = model(**msg_inputs, output_hidden_states=True)
        hidden = msg_outputs.hidden_states[-1]
        mask = msg_inputs['attention_mask'].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)

        # Apply ablation to latent
        if ablation == 'null':
            latent = torch.zeros(1, codec.k, codec.d, device=device)
        elif ablation == 'random':
            latent = torch.randn(1, codec.k, codec.d, device=device)
        else:
            # Normal or shuffle: use actual codec encoding
            latent = codec.encode(pooled)

        # Decode to prefix embedding
        prefix_embedding = codec.decode(latent)  # Shape: (1, d_model)

    # P1 FIX: Agent B prompt NEVER contains message_A for latent conditions
    # This is the NO TEXT LEAK guarantee
    prompt_B = f"""You are Agent B in a collaborative task.

Your observation:
{task.obs_B}

Your partner sent you encoded information (injected as a learned prefix).
Based on your observation and the encoded context, provide an ANSWER.

ANSWER in format {{"x1": <value>, "x2": <value>, "x3": <value>, "x4": <value>}}:"""

    # P1 FIX: Actually inject prefix_embedding via inputs_embeds
    inputs_B = tokenizer(prompt_B, return_tensors="pt").to(device)

    # Get input embeddings for prompt
    input_embeds_B = model.get_input_embeddings()(inputs_B['input_ids'])

    # Expand prefix_embedding to match expected shape (1, 1, d_model) -> prepend as soft token
    # Cast to model dtype (float16 on CUDA)
    soft_prefix = prefix_embedding.unsqueeze(1).to(input_embeds_B.dtype)  # (1, 1, d_model)

    # Concatenate: [soft_prefix, prompt_embeddings]
    combined_embeds = torch.cat([soft_prefix, input_embeds_B], dim=1)

    # Create attention mask for combined input
    soft_mask = torch.ones(1, 1, device=device, dtype=inputs_B['attention_mask'].dtype)
    combined_mask = torch.cat([soft_mask, inputs_B['attention_mask']], dim=1)

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
            extended_mask = torch.ones(1, current_seq_len, device=device, dtype=combined_mask.dtype)

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
    json_match = re.search(r'\{[^}]+\}', response_B)
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
        if '/root' not in sys.path:
            sys.path.insert(0, '/root')
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
    response_A = tokenizer.decode(outputs_A[0][inputs_A['input_ids'].shape[1]:], skip_special_tokens=True)
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
    response_B = tokenizer.decode(outputs_B[0][inputs_B['input_ids'].shape[1]:], skip_special_tokens=True)

    # Extract answer
    answer = None
    json_match = re.search(r'\{[^}]+\}', response_B)
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
    k_vectors = checkpoint['k_vectors']
    d_model = checkpoint['d_model']
    hidden_dim = checkpoint['hidden_dim']

    codec = create_codec_model(d_model, k_vectors, hidden_dim)
    codec.load_state_dict(checkpoint['state_dict'])
    codec = codec.to(device)
    codec.eval()

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
    print(f"P1 Baseline: {p1_successes}/{n_episodes} = {p1_rate:.1%} (95% CI: [{p1_ci[0]:.1%}, {p1_ci[1]:.1%}])")

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
    print(f"L1 Injection: {l1_successes}/{n_episodes} = {l1_rate:.1%} (95% CI: [{l1_ci[0]:.1%}, {l1_ci[1]:.1%}])")

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
        print("   L1 is closer to P0 than P1, suggesting codec is not conveying information")
        print("   Check: Is prefix_embedding being used? Is KV cache working?")

    return {
        'p1_baseline': {'successes': p1_successes, 'n': n_episodes, 'rate': p1_rate, 'ci': p1_ci},
        'l1_injection': {'successes': l1_successes, 'n': n_episodes, 'rate': l1_rate, 'ci': l1_ci},
        'p0_reference': p0_reference,
        'injection_works': injection_works,
        'l1_vs_p1_gap': l1_vs_p1_gap,
        'l1_vs_p0_gap': l1_vs_p0_gap,
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
    normal = results.get('normal', {})
    normal_sr = normal.get('success_rate', 0)
    normal_n = normal.get('n_episodes', 50)
    normal_successes = normal.get('successes', int(normal_sr * normal_n))

    # P2 FIX: CI-aware gating
    normal_ci = wilson_ci(normal_successes, normal_n)
    print(f"\nNormal: {normal_sr:.1%} (95% CI: [{normal_ci[0]:.1%}, {normal_ci[1]:.1%}])")

    # Gate 1: Sanity - CI_low must meet threshold
    gates['gate1_sanity'] = normal_ci[0] >= config.gate1_threshold
    print(f"Gate 1 (Sanity): CI_low {normal_ci[0]:.1%} >= {config.gate1_threshold:.1%}? "
          f"{'PASS' if gates['gate1_sanity'] else 'FAIL'}")

    # Gate 2: Retention (only check at k>=16)
    if config.k_vectors >= 16:
        gates['gate2_retention'] = normal_ci[0] >= config.gate2_threshold
        print(f"Gate 2 (Retention): CI_low {normal_ci[0]:.1%} >= {config.gate2_threshold:.1%}? "
              f"{'PASS' if gates['gate2_retention'] else 'FAIL'}")

    # P0 baseline CI (for ablation comparison)
    # Using hypothetical n=50 at p=0.20 for P0 reference
    p0_ci = wilson_ci(int(config.p0_baseline * normal_n), normal_n)
    print(f"\nP0 baseline reference: {config.p0_baseline:.1%} (CI: [{p0_ci[0]:.1%}, {p0_ci[1]:.1%}])")

    # Ablation checks with CI-aware criteria
    print("\nAblation checks:")

    # Null ablation: CI should overlap P0 CI
    null = results.get('null_message', {})
    null_sr = null.get('success_rate', 0)
    null_n = null.get('n_episodes', normal_n)
    null_successes = null.get('successes', int(null_sr * null_n))
    null_ci = wilson_ci(null_successes, null_n)
    gates['ablation_null'] = ci_overlap(null_ci, p0_ci)
    print(f"  Null: {null_sr:.1%} (CI: [{null_ci[0]:.1%}, {null_ci[1]:.1%}]) "
          f"overlaps P0? {'OK' if gates['ablation_null'] else 'UNEXPECTED'}")

    # Random ablation: CI should overlap P0 CI
    rand = results.get('random_latent', {})
    random_sr = rand.get('success_rate', 0)
    random_n = rand.get('n_episodes', normal_n)
    random_successes = rand.get('successes', int(random_sr * random_n))
    random_ci = wilson_ci(random_successes, random_n)
    gates['ablation_random'] = ci_overlap(random_ci, p0_ci)
    print(f"  Random: {random_sr:.1%} (CI: [{random_ci[0]:.1%}, {random_ci[1]:.1%}]) "
          f"overlaps P0? {'OK' if gates['ablation_random'] else 'UNEXPECTED'}")

    # Shuffle ablation: CI_high must be < CI_low(P0) (proves semantic dependence)
    shuffle = results.get('shuffle', {})
    if shuffle:
        shuffle_sr = shuffle.get('success_rate', 0)
        shuffle_n = shuffle.get('n_episodes', normal_n)
        shuffle_successes = shuffle.get('successes', int(shuffle_sr * shuffle_n))
        shuffle_ci = wilson_ci(shuffle_successes, shuffle_n)
        # Shuffle should crater: CI_high(shuffle) < CI_low(P0)
        gates['ablation_shuffle'] = shuffle_ci[1] < p0_ci[0]
        print(f"  Shuffle: {shuffle_sr:.1%} (CI: [{shuffle_ci[0]:.1%}, {shuffle_ci[1]:.1%}]) "
              f"CI_high < P0_CI_low? {'OK' if gates['ablation_shuffle'] else 'UNEXPECTED'}")
    else:
        gates['ablation_shuffle'] = False
        print(f"  Shuffle: NOT RUN (required ablation missing)")

    # Overall validity
    all_ablations_pass = (gates.get('ablation_null', False) and
                          gates.get('ablation_random', False) and
                          gates.get('ablation_shuffle', False))
    gates['all_ablations_valid'] = all_ablations_pass

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
    def train_on_modal(k: int, epochs: int, data_path: str = "/data/m2_train.jsonl") -> Dict:
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
    parser.add_argument("--plumbing-proof", type=str, metavar="CHECKPOINT",
                        help="Run plumbing proof on a trained checkpoint to verify injection works")
    parser.add_argument("--plumbing-episodes", type=int, default=10,
                        help="Number of episodes for plumbing proof (default: 10)")

    args = parser.parse_args()

    # Plumbing proof mode
    if args.plumbing_proof:
        config = M2Config(k_vectors=args.k)
        result = run_plumbing_proof(
            checkpoint_path=args.plumbing_proof,
            config=config,
            device=args.device,
            n_episodes=args.plumbing_episodes,
        )

        # Save results
        import json
        proof_path = args.plumbing_proof.replace('.pt', '_plumbing_proof.json')
        with open(proof_path, 'w') as f:
            # Convert tuples to lists for JSON
            result_json = {
                'p1_baseline': {**result['p1_baseline'], 'ci': list(result['p1_baseline']['ci'])},
                'l1_injection': {**result['l1_injection'], 'ci': list(result['l1_injection']['ci'])},
                'p0_reference': result['p0_reference'],
                'injection_works': result['injection_works'],
                'l1_vs_p1_gap': result['l1_vs_p1_gap'],
                'l1_vs_p0_gap': result['l1_vs_p0_gap'],
            }
            json.dump(result_json, f, indent=2)
        print(f"\nResults saved to: {proof_path}")
        return

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
            # Run on Modal in parallel (use Modal volume path)
            with app.run():
                results = []
                for k in k_values:
                    result = train_on_modal.remote(k, args.epochs, "/data/m2_train.jsonl")
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
                # Use Modal volume path, not local path
                result = train_on_modal.remote(args.k, args.epochs, "/data/m2_train.jsonl")
                print(f"\nResult: {result}")


if __name__ == "__main__":
    main()
