"""Tests for M2v2 architecture utilities."""

import pytest

torch = pytest.importorskip("torch")

from lpca.training.m2v2 import (  # noqa: E402
    CurriculumScheduler,
    DiscreteBottleneck,
    M2V2Codec,
    anti_shuffle_margin_loss,
    combine_m2v2_losses,
    mutual_information_regularizer,
)


def test_discrete_bottleneck_shapes() -> None:
    bottleneck = DiscreteBottleneck(d_model=16, codebook_size=32)
    slots = torch.randn(4, 8, 16)
    out = bottleneck(slots)
    assert out.quantized.shape == (4, 8, 16)
    assert out.code_indices.shape == (4, 8)
    assert out.assignment_probs.shape == (4, 8, 32)
    assert out.code_indices.max().item() < 32
    assert out.vq_loss.ndim == 0


def test_anti_shuffle_margin_loss_behavior() -> None:
    nll_pos = torch.tensor([1.0, 1.2, 0.8])
    nll_shuffle_good = torch.tensor([2.1, 2.3, 2.0])
    nll_shuffle_bad = torch.tensor([1.1, 1.3, 0.9])
    loss_good = anti_shuffle_margin_loss(nll_pos, nll_shuffle_good, margin=1.0)
    loss_bad = anti_shuffle_margin_loss(nll_pos, nll_shuffle_bad, margin=1.0)
    assert loss_good.item() == pytest.approx(0.0)
    assert loss_bad.item() > 0.0


def test_mutual_information_regularizer_prefers_confident_codes() -> None:
    uniform = torch.full((2, 2, 4), 0.25)
    one_hot = torch.tensor(
        [
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        ]
    )
    mi_uniform = mutual_information_regularizer(uniform)
    mi_one_hot = mutual_information_regularizer(one_hot)
    assert mi_one_hot.item() < mi_uniform.item()


def test_m2v2_codec_forward_shapes() -> None:
    codec = M2V2Codec(d_model=32, k_vectors=4, hidden_dim=64, codebook_size=16)
    text_embedding = torch.randn(3, 32)
    out = codec(text_embedding)
    assert out.prefix_embeddings.shape == (3, 4, 32)
    assert out.reconstruction.shape == (3, 32)
    assert out.code_indices.shape == (3, 4)
    assert out.assignment_probs.shape == (3, 4, 16)


def test_combine_m2v2_losses_returns_all_terms() -> None:
    scheduler = CurriculumScheduler(warmup_steps=10, semantic_ramp_steps=20)
    weights = scheduler.weights(step=15)
    losses = combine_m2v2_losses(
        nll_pos=torch.tensor([1.0, 1.2]),
        nll_shuffle=torch.tensor([1.5, 1.7]),
        recon_loss=torch.tensor(0.4),
        vq_loss=torch.tensor(0.2),
        assignment_probs=torch.full((2, 2, 4), 0.25),
        curriculum_weights=weights,
        margin=1.0,
    )
    assert set(losses.keys()) == {"total", "ce", "anti_shuffle", "mi", "recon", "vq"}
    assert losses["total"].ndim == 0
