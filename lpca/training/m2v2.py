"""M2v2 architecture components for semantic codec training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CurriculumWeights:
    """Loss weights for a curriculum phase."""

    ce: float
    anti_shuffle: float
    mi: float
    recon: float
    vq: float


class CurriculumScheduler:
    """Simple piecewise-linear curriculum over optimization steps."""

    def __init__(
        self,
        warmup_steps: int = 500,
        semantic_ramp_steps: int = 1500,
    ) -> None:
        self.warmup_steps = max(1, warmup_steps)
        self.semantic_ramp_steps = max(1, semantic_ramp_steps)

    def weights(self, step: int) -> CurriculumWeights:
        if step < self.warmup_steps:
            return CurriculumWeights(
                ce=1.0,
                anti_shuffle=0.1,
                mi=0.0,
                recon=1.0,
                vq=0.25,
            )

        ramp_progress = min(
            1.0,
            max(0.0, (step - self.warmup_steps) / self.semantic_ramp_steps),
        )
        return CurriculumWeights(
            ce=1.0,
            anti_shuffle=0.1 + 0.9 * ramp_progress,
            mi=0.1 * ramp_progress,
            recon=1.0 - 0.5 * ramp_progress,
            vq=0.25,
        )


@dataclass
class BottleneckOutput:
    """Outputs from the discrete bottleneck."""

    quantized: torch.Tensor
    code_indices: torch.Tensor
    assignment_probs: torch.Tensor
    vq_loss: torch.Tensor


class DiscreteBottleneck(nn.Module):
    """Vector-quantized bottleneck with straight-through estimator."""

    def __init__(
        self,
        d_model: int,
        codebook_size: int = 256,
        beta: float = 0.25,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.codebook_size = codebook_size
        self.beta = beta
        self.codebook = nn.Parameter(torch.randn(codebook_size, d_model) * 0.02)

    def forward(
        self, slot_embeddings: torch.Tensor, temperature: float = 1.0
    ) -> BottleneckOutput:
        if slot_embeddings.ndim != 3:
            raise ValueError(
                f"slot_embeddings must have shape [batch, k, d], got {slot_embeddings.shape}"
            )
        batch, k, d = slot_embeddings.shape
        if d != self.d_model:
            raise ValueError(
                f"slot_embeddings last dim {d} must equal d_model={self.d_model}"
            )

        flat = slot_embeddings.reshape(-1, d)
        codebook = self.codebook
        distances = (
            (flat**2).sum(dim=1, keepdim=True)
            - 2.0 * flat @ codebook.t()
            + (codebook**2).sum(dim=1)
        )
        codes = distances.argmin(dim=1)
        one_hot = F.one_hot(codes, num_classes=self.codebook_size).float()
        quantized = one_hot @ codebook
        quantized = quantized.view(batch, k, d)

        quantized_st = slot_embeddings + (quantized - slot_embeddings).detach()

        probs = F.softmax(-distances / max(temperature, 1e-6), dim=-1)
        probs = probs.view(batch, k, self.codebook_size)

        commit = F.mse_loss(slot_embeddings, quantized.detach())
        codebook_loss = F.mse_loss(quantized, slot_embeddings.detach())
        vq_loss = codebook_loss + self.beta * commit

        return BottleneckOutput(
            quantized=quantized_st,
            code_indices=codes.view(batch, k),
            assignment_probs=probs,
            vq_loss=vq_loss,
        )


def anti_shuffle_margin_loss(
    nll_pos: torch.Tensor,
    nll_shuffle: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """Penalize when shuffled prefixes are not sufficiently worse than correct ones."""

    gap = nll_shuffle - nll_pos
    return F.relu(margin - gap).mean()


def mutual_information_regularizer(
    assignment_probs: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Return a loss term that *maximizes* I(message;code).

    Loss is negative MI, so minimizing this term increases MI.
    """

    if assignment_probs.ndim != 3:
        raise ValueError("assignment_probs must have shape [batch, k, codebook_size]")
    probs = assignment_probs.reshape(-1, assignment_probs.shape[-1])
    probs = probs.clamp_min(eps)
    marginal = probs.mean(dim=0)
    h_marginal = -(marginal * marginal.log()).sum()
    h_conditional = -(probs * probs.log()).sum(dim=1).mean()
    mi = h_marginal - h_conditional
    return -mi


@dataclass
class M2V2Output:
    """Forward outputs from M2v2 codec."""

    prefix_embeddings: torch.Tensor
    reconstruction: torch.Tensor
    code_indices: torch.Tensor
    assignment_probs: torch.Tensor
    vq_loss: torch.Tensor


class M2V2Codec(nn.Module):
    """Discrete M2v2 codec: encoder -> VQ bottleneck -> decoder/reconstructor."""

    def __init__(
        self,
        d_model: int,
        k_vectors: int = 16,
        hidden_dim: int = 512,
        codebook_size: int = 256,
        vq_beta: float = 0.25,
    ) -> None:
        super().__init__()
        self.k = k_vectors
        self.d = d_model
        flat_dim = k_vectors * d_model

        self.encoder = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, flat_dim),
        )
        self.bottleneck = DiscreteBottleneck(
            d_model=d_model,
            codebook_size=codebook_size,
            beta=vq_beta,
        )
        self.decoder = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, flat_dim),
        )
        self.reconstructor = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(
        self, text_embedding: torch.Tensor, temperature: float = 1.0
    ) -> M2V2Output:
        if text_embedding.ndim != 2:
            raise ValueError(
                f"text_embedding must have shape [batch, d], got {text_embedding.shape}"
            )
        slots = self.encoder(text_embedding).view(-1, self.k, self.d)
        bottleneck = self.bottleneck(slots, temperature=temperature)
        flat = bottleneck.quantized.reshape(-1, self.k * self.d)
        prefix = self.decoder(flat).view(-1, self.k, self.d)
        recon = self.reconstructor(flat)
        return M2V2Output(
            prefix_embeddings=prefix,
            reconstruction=recon,
            code_indices=bottleneck.code_indices,
            assignment_probs=bottleneck.assignment_probs,
            vq_loss=bottleneck.vq_loss,
        )


def combine_m2v2_losses(
    nll_pos: torch.Tensor,
    nll_shuffle: torch.Tensor,
    recon_loss: torch.Tensor,
    vq_loss: torch.Tensor,
    assignment_probs: torch.Tensor,
    curriculum_weights: CurriculumWeights,
    margin: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Combine CE, anti-shuffle, MI, reconstruction, and VQ objectives."""

    ce_loss = nll_pos.mean()
    anti_loss = anti_shuffle_margin_loss(
        nll_pos=nll_pos, nll_shuffle=nll_shuffle, margin=margin
    )
    mi_loss = mutual_information_regularizer(assignment_probs)

    total = (
        curriculum_weights.ce * ce_loss
        + curriculum_weights.anti_shuffle * anti_loss
        + curriculum_weights.mi * mi_loss
        + curriculum_weights.recon * recon_loss
        + curriculum_weights.vq * vq_loss
    )
    return {
        "total": total,
        "ce": ce_loss,
        "anti_shuffle": anti_loss,
        "mi": mi_loss,
        "recon": recon_loss,
        "vq": vq_loss,
    }
