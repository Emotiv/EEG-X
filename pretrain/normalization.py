"""Shared parameter-free normalization primitives for pretraining models."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    """Learned-scale RMSNorm with FP32 math and an explicit epsilon."""

    def __init__(self, width, eps=1.0e-6):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, inputs):
        return float32_rms_norm(inputs, weight=self.weight, eps=self.eps)


def float32_rms_norm(
    inputs: torch.Tensor,
    *,
    weight: torch.Tensor | None = None,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Use PyTorch's fused RMS kernel with FP32 math and the original output dtype."""
    return F.rms_norm(
        inputs.float(),
        (inputs.shape[-1],),
        weight.float() if weight is not None else None,
        eps,
    ).to(dtype=inputs.dtype)


def parameter_free_rms_norm(
    inputs: torch.Tensor,
    *,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Normalize the final axis in float32 with RMSNorm and no learned scale."""
    return float32_rms_norm(inputs, eps=eps)
