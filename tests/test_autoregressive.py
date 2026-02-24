"""Regression tests for cache-preserving autoregressive decoding."""

from dataclasses import dataclass
from typing import Any, List

import torch

from lpca.core.autoregressive import decode_with_cache, prefill_with_embeds


@dataclass
class _FakeOutputs:
    logits: torch.Tensor
    past_key_values: Any


class _FakeModel:
    """Small fake model to validate cache decoding behavior."""

    def __init__(self, vocab_size: int = 8):
        self.vocab_size = vocab_size
        self.calls: List[dict] = []

    def __call__(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
    ):
        self.calls.append(
            {
                "input_ids": input_ids,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
            }
        )

        if inputs_embeds is not None:
            batch = inputs_embeds.shape[0]
            seq = inputs_embeds.shape[1]
        elif input_ids is not None:
            batch = input_ids.shape[0]
            seq = input_ids.shape[1]
        else:
            batch, seq = 1, 1

        logits = torch.zeros(batch, seq, self.vocab_size)
        next_past = f"past_{len(self.calls)}"
        return _FakeOutputs(logits=logits, past_key_values=next_past)


def test_prefill_with_embeds_uses_cache():
    model = _FakeModel(vocab_size=11)
    embeds = torch.randn(1, 4, 6)
    mask = torch.ones(1, 4, dtype=torch.long)

    next_logits, past, base_seq_len = prefill_with_embeds(model, embeds, mask)

    assert next_logits.shape == (1, 11)
    assert past == "past_1"
    assert base_seq_len == 4
    assert len(model.calls) == 1
    assert model.calls[0]["use_cache"] is True
    assert model.calls[0]["inputs_embeds"] is not None
    assert model.calls[0]["past_key_values"] is None


def test_decode_with_cache_preserves_prefill_context():
    model = _FakeModel(vocab_size=10)
    next_logits = torch.zeros(1, 10)

    sequence = [2, 4, 9]  # EOS is 9

    def sampler(_logits, _temperature):
        return torch.tensor([[sequence.pop(0)]], dtype=torch.long)

    generated = decode_with_cache(
        model=model,
        next_logits=next_logits,
        past_key_values="prefill_past",
        base_seq_len=5,
        max_new_tokens=10,
        temperature=0.3,
        eos_token_id=9,
        mask_dtype=torch.long,
        device=torch.device("cpu"),
        sampler=sampler,
    )

    # We sample 3 tokens and stop on EOS.
    assert generated == [2, 4, 9]

    # Model should be called only for tokens before EOS (2 calls here).
    assert len(model.calls) == 2
    assert model.calls[0]["past_key_values"] == "prefill_past"
    assert model.calls[1]["past_key_values"] == "past_1"
    assert model.calls[0]["use_cache"] is True
    assert model.calls[1]["use_cache"] is True

    # Attention mask must extend from original context length.
    assert model.calls[0]["attention_mask"].shape == (1, 6)
    assert model.calls[1]["attention_mask"].shape == (1, 7)
