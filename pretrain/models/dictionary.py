"""Differentiable winner counts with the manuscript's convolutional dictionary."""

import torch
from torch import nn
import torch.nn.functional as F


class SoftDiCT(nn.Module):
    """Keep fixed filters and temporal counts; soften only kernel assignments.

    Channels share the seeded dictionary but compete independently. Features use
    temporal sums, without raw/difference branches or response-amplitude weights.
    """

    def __init__(self, objective):
        super().__init__()
        from pretrain.objectives import validate_objective

        validate_objective(objective)
        self.groups = objective["groups"]
        self.k = objective["k"]
        self.temperature = objective["temperature"]
        self.kernel_sizes = tuple(objective["kernel_sizes"])
        self.dilations = tuple(objective["dilations"])
        generator = torch.Generator(device="cpu").manual_seed(objective["seed"])
        for size in self.kernel_sizes:
            weight = torch.randn(
                len(self.dilations), self.groups * self.k, 1, size, generator=generator
            )
            weight -= weight.mean(dim=-1, keepdim=True)
            weight /= weight.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            self.register_buffer(f"filters_{size}", weight)

    def forward(self, eeg):
        if eeg.ndim != 3 or min(eeg.shape) < 1:
            raise ValueError("Expected nonempty EEG [batch, channels, time]")
        with torch.autocast(device_type=eeg.device.type, enabled=False):
            batch, channels, samples = eeg.shape
            signal = eeg.float().reshape(batch * channels, 1, samples)
            features = []
            for size in self.kernel_sizes:
                filters = getattr(self, f"filters_{size}")
                for index, dilation in enumerate(self.dilations):
                    responses = F.conv1d(
                        signal,
                        filters[index],
                        dilation=dilation,
                        padding=(size - 1) * dilation // 2,
                    ).reshape(batch, channels, self.groups, self.k, samples)
                    counts = (responses / self.temperature).softmax(dim=3).sum(dim=-1)
                    features.append(counts.flatten(2))
            return torch.cat(features, dim=-1).flatten(1)
