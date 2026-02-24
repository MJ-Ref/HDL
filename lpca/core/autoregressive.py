"""
Autoregressive decoding helpers for latent-context experiments.

These utilities make sure generation preserves prefilled context by using
`past_key_values` after the initial forward pass.
"""

from typing import Any, Callable, List, Optional, Tuple

import torch


SamplerFn = Callable[[torch.Tensor, float], torch.Tensor]


def sample_next_token(next_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Sample the next token from logits with temperature scaling."""
    safe_temp = max(float(temperature), 1e-6)
    probs = torch.softmax(next_logits / safe_temp, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def prefill_with_embeds(
    model: Any,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Tuple[torch.Tensor, Any, int]:
    """
    Prefill model state from embeddings and return decoding state.

    Returns:
        next_logits: (batch, vocab) logits for the next token
        past_key_values: model cache to continue decoding
        base_seq_len: context length used to build future attention masks
    """
    with torch.no_grad():
        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=True,
        )

    return (
        outputs.logits[:, -1, :],
        outputs.past_key_values,
        int(attention_mask.shape[1]),
    )


def prefill_with_ids(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Tuple[torch.Tensor, Any, int]:
    """
    Prefill model state from token ids and return decoding state.

    Returns:
        next_logits: (batch, vocab) logits for the next token
        past_key_values: model cache to continue decoding
        base_seq_len: context length used to build future attention masks
    """
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )

    return (
        outputs.logits[:, -1, :],
        outputs.past_key_values,
        int(attention_mask.shape[1]),
    )


def decode_with_cache(
    model: Any,
    next_logits: torch.Tensor,
    past_key_values: Any,
    base_seq_len: int,
    max_new_tokens: int,
    temperature: float,
    eos_token_id: Optional[int],
    mask_dtype: torch.dtype,
    device: torch.device,
    sampler: Optional[SamplerFn] = None,
) -> List[int]:
    """
    Decode tokens while preserving context with `past_key_values`.

    The first token is sampled from `next_logits` (from prefill pass), and all
    subsequent tokens are generated using cached state.
    """
    sample_fn = sampler or sample_next_token

    generated_ids: List[int] = []
    current_logits = next_logits
    current_past = past_key_values

    for _ in range(max_new_tokens):
        next_token = sample_fn(current_logits, temperature)
        token_id = int(next_token.item())
        generated_ids.append(token_id)

        if eos_token_id is not None and token_id == eos_token_id:
            break

        current_len = base_seq_len + len(generated_ids)
        extended_mask = torch.ones(
            next_token.shape[0],
            current_len,
            device=device,
            dtype=mask_dtype,
        )

        with torch.no_grad():
            outputs = model(
                input_ids=next_token,
                attention_mask=extended_mask,
                past_key_values=current_past,
                use_cache=True,
            )

        current_past = outputs.past_key_values
        current_logits = outputs.logits[:, -1, :]

    return generated_ids
