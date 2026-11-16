"""Shared parameter-free normalization primitives for pretraining models."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    """Learned-scale RMSNorm with FP32 math and an explicit epsilon."""

    def __init__(self, width, eps=1.0e-6, *, backend="torch"):
        super().__init__()
        if not isinstance(backend, str) or backend not in {"torch", "triton"}:
            raise ValueError("RMSNorm backend must be torch or triton")
        self.eps = float(eps)
        self.backend = backend
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, inputs):
        return _apply_rms_norm(inputs, weight=self.weight, eps=self.eps, backend=self.backend)


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
    backend: str = "torch",
) -> torch.Tensor:
    """Normalize the final axis in float32 with RMSNorm and no learned scale."""
    return _apply_rms_norm(inputs, weight=None, eps=eps, backend=backend)


def _apply_rms_norm(inputs, *, weight, eps, backend):
    """Dispatch the optional CUDA kernel and retain native fallback elsewhere."""
    if backend == "torch":
        return float32_rms_norm(inputs, weight=weight, eps=eps)
    if backend != "triton":
        raise ValueError("RMSNorm backend must be torch or triton")
    if not inputs.is_cuda:
        return float32_rms_norm(inputs, weight=weight, eps=eps)
    from .cuda import MAX_ELEMENTS, MAX_WEIGHT_ROWS, SUPPORTED_WIDTHS, fused_rms_norm

    supported_weight = weight is None or (
        weight.device == inputs.device
        and weight.dtype == torch.float32
        and inputs.ndim > 0
        and weight.shape == (inputs.shape[-1],)
        and weight.stride(0) in (0, 1)
    )
    # Keep indexing and the deterministic scale-gradient reduction within the
    # audited CUDA kernel bounds. Native RMSNorm handles other layouts/sizes.
    if (
        inputs.ndim > 0
        and inputs.numel() > 0
        and inputs.numel() < MAX_ELEMENTS
        and inputs.shape[-1] in SUPPORTED_WIDTHS
        and inputs.stride(-1) in (0, 1)
        and (weight is None or inputs.numel() // inputs.shape[-1] <= MAX_WEIGHT_ROWS)
        and inputs.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and supported_weight
        and isinstance(eps, (int, float))
        and math.isfinite(eps)
    ):
        return fused_rms_norm(inputs, weight=weight, eps=eps)
    return float32_rms_norm(inputs, weight=weight, eps=eps)
