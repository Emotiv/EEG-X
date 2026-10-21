"""Fixed XYZ features ported from the EEG-X2 scaling encoding."""

import hashlib
import math

import torch
from torch import nn


def encode_positions(normalized, width):
    """Frequency-major XYZ sin/cos pairs, with the original scaling gain."""
    k = width // 6
    if k < 1 or not torch.isfinite(normalized).all() or normalized.shape[-1] != 3:
        raise ValueError("Encoding requires finite XYZ positions and width >= 6")
    frequencies = 5.0 ** (-torch.linspace(0, 1, k, dtype=torch.float64, device=normalized.device))
    phases = normalized.double().unsqueeze(-2) * frequencies[:, None]
    features = torch.stack((phases.sin(), phases.cos()), dim=-1).flatten(-3)
    return nn.functional.pad(features / math.sqrt(3 * k), (0, width - 6 * k))


class FixedChannelEmbedding(nn.Module):
    """One vocabulary-wide coordinate frame, independent of the input headset."""

    def __init__(self, vocabulary, width):
        super().__init__()
        self.num_embeddings = vocabulary.embedding_capacity
        ids = torch.tensor([channel.index for channel in vocabulary.channels])
        positions = torch.tensor(
            [channel.position for channel in vocabulary.channels], dtype=torch.float64
        )
        if not torch.isfinite(positions).all():
            raise ValueError("Invalid nonfinite vocabulary positions")
        center = positions.mean(0)
        radius = (positions - center).norm(dim=-1).max()
        if not torch.isfinite(radius) or radius <= 0:
            raise ValueError("Invalid vocabulary radius")
        active = torch.zeros(self.num_embeddings, dtype=torch.bool)
        active[ids] = True
        self.register_buffer("active", active)
        weight = torch.zeros(self.num_embeddings, width)
        weight[ids] = encode_positions((positions - center) / radius, width).float()
        self.register_buffer("weight", weight)
        self.encoding_contract = {
            "version": "fixed_3d_v1",
            "center": center.tolist(),
            "radius": radius.item(),
            "width": width,
            "frequencies": (5.0 ** -torch.linspace(0, 1, width // 6, dtype=torch.float64)).tolist(),
            "ordering": "frequency_xyz_sin_cos",
            "gain": 1,
            "normalization": "1/sqrt(3K)",
            "padding": width % 6,
            "computation_dtype": "float64",
            "buffer_dtype": "float32",
            "buffer_sha256": hashlib.sha256(weight.numpy().tobytes()).hexdigest(),
        }

    def forward(self, indices):
        if indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("Physical channel identities must be integer IDs")
        if (indices < 0).any() or (indices >= self.num_embeddings).any():
            raise ValueError("Unknown physical channel identity")
        if not self.active[indices].all():
            raise ValueError("Inactive or reserved physical channel identity")
        ordered = indices.sort(-1).values
        if ordered.shape[-1] > 1 and (ordered[..., 1:] == ordered[..., :-1]).any():
            raise ValueError("Physical channel identities must be unique")
        return nn.functional.embedding(indices, self.weight)
