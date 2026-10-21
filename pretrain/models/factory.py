"""Revised EEG-X construction and all-RMS supervised readout."""

import copy

import torch
from torch import nn

from pretrain.models.eegx import EEGXReconstruction
from pretrain.normalization import RMSNorm
from pretrain.objectives import DEFAULT_HYDRA, validate_objective


GEOMETRIES = {
    "d8w64": (8, 64),
    "d4w64": (4, 64),
    "d4w128": (4, 128),
    "d4w512": (4, 512),
    "d8w128": (8, 128),
    "d8w512": (8, 512),
}


def resolve_model_config(config):
    """Materialize the selected primary geometry and default objective."""
    resolved = {"model": "eegx_hybrid_v2", "size": "d8w64", "dropout": 0.1, **copy.deepcopy(config)}
    if resolved["size"] not in GEOMETRIES:
        raise ValueError("An accepted EEG-X size is required; EEGX2 checkpoints are incompatible")
    depth, width = GEOMETRIES[resolved["size"]]
    expected = {"model": "eegx_hybrid_v2", "depth": depth, "width": width, "heads": 8}
    if any(key in resolved and resolved[key] != value for key, value in expected.items()):
        raise ValueError("Explicit EEG-X geometry differs from the selected size")
    resolved.update(expected)
    if "objective" not in resolved:
        resolved["objective"] = {
            **copy.deepcopy(DEFAULT_HYDRA), "seed": resolved.get("seed", DEFAULT_HYDRA["seed"])
        }
    validate_objective(resolved["objective"])
    return resolved


def create_model(config):
    model = EEGXReconstruction(resolve_model_config(config))
    assert_all_rms(model)
    return model


def assert_all_rms(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            raise ValueError(f"LayerNorm is outside the all-RMS contract: {name}")
        if hasattr(module, "qk_normalization") and module.qk_normalization != "rmsnorm":
            raise ValueError(f"Attention must use RMS Q/K normalization: {name}")


class Classifier(nn.Module):
    def __init__(self, encoder, classes):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            RMSNorm(encoder.embed_dim), nn.Dropout(0.1), nn.Linear(encoder.embed_dim, classes)
        )
        assert_all_rms(self)

    def forward(self, eeg, channel_ids):
        features = self.encoder.forward_emotiv_features(
            eeg, channel_ids, input_window_normalization="none"
        )
        return self.head(features)


def attention_backends(model):
    """Report the backend used by each attention module after a real forward."""
    return {
        name: module.actual_backend
        for name, module in model.named_modules()
        if getattr(module, "actual_backend", None) is not None
    }


def masks(window_ids, channel_ids, *, samples, seed):
    """Batch-independent masks paired across model sizes and objectives.

    Hash each window once, then mix explicit physical channel and time IDs.
    This avoids hashing every token in Python on the training hot path.
    """
    import hashlib

    ids = torch.as_tensor(channel_ids, dtype=torch.int64, device="cpu")
    if ids.ndim == 1:
        ids = ids.expand(len(window_ids), -1)
    if ids.ndim != 2 or ids.shape[0] != len(window_ids):
        raise ValueError("Channel identities must match windows")
    if samples < 32 or samples % 32:
        raise ValueError("Window length must be divisible by patch length")
    if any(len(set(row)) != len(row) for row in ids.tolist()):
        raise ValueError("Physical channels must be unique")
    base = torch.tensor(
        [
            int.from_bytes(hashlib.sha256(f"{seed}:{window}".encode()).digest()[:4], "little")
            for window in window_ids
        ],
        dtype=torch.int64,
    )
    time = torch.arange(samples // 32, dtype=torch.int64)
    values = base[:, None, None] + time[None, :, None] * 1_000_003 + ids[:, None, :] * 97_409
    modulus = (1 << 31) - 1
    values = values.remainder(modulus)
    values = values.bitwise_xor(values >> 16)
    values = (values * 2_246_822_519 + 3_266_489_917).remainder(modulus)
    values = values.bitwise_xor(values >> 13)
    values = (values * 3_266_489_917 + 668_265_263).remainder(modulus).flatten(1)
    # Stable ties follow explicit channel identity rather than input channel order.
    channel_order = ids.argsort(dim=1, stable=True)
    canonical = (time[None, :, None] * ids.shape[1] + channel_order[:, None, :]).flatten(1)
    canonical_values = values.gather(1, canonical)
    order = canonical.gather(1, canonical_values.argsort(dim=1, stable=True))
    mask = torch.ones_like(values, dtype=torch.float32)
    return mask.scatter(1, order[:, : values.shape[1] // 4], 0)
