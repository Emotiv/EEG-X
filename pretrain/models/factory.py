"""Revised EEG-X construction and all-RMS supervised readout."""

import copy
import re

import torch
from torch import nn

from pretrain.models.eegx import EEGXReconstruction
from pretrain.normalization import RMSNorm
from pretrain.objectives import DEFAULT_HYDRA, validate_objective


def resolve_model_config(config):
    """Resolve one explicitly configured EEG-X model, with no experiment grid."""
    supplied = copy.deepcopy(config)
    if "size" in supplied:
        match = re.fullmatch(r"d([1-9][0-9]*)w([1-9][0-9]*)", str(supplied["size"]))
        if match is None:
            raise ValueError(
                "EEG-X size must use d<depth>w<width>; EEGX2 checkpoints are incompatible"
            )
        for key, value in zip(("depth", "width"), map(int, match.groups())):
            if key in supplied and supplied[key] != value:
                raise ValueError("Explicit EEG-X geometry differs from the selected size")
            supplied[key] = value
    resolved = {
        "model": "eegx_hybrid_v2",
        "depth": 8,
        "width": 64,
        "heads": 8,
        "dropout": 0.1,
        "dictionary_backend": "torch",
        **supplied,
    }
    if resolved["model"] != "eegx_hybrid_v2":
        raise ValueError("The EEG-X hybrid model is required")
    if any(
        type(resolved[key]) is not int or resolved[key] < 1 for key in ("depth", "width", "heads")
    ):
        raise ValueError("EEG-X depth, width and heads must be positive integers")
    if resolved["heads"] != 8 or resolved["width"] % resolved["heads"]:
        raise ValueError("EEG-X requires eight heads and a width divisible by eight")
    resolved["size"] = f"d{resolved['depth']}w{resolved['width']}"
    normalization_backend = resolved.get("normalization_backend", "torch")
    if not isinstance(normalization_backend, str) or normalization_backend not in {"torch", "triton"}:
        raise ValueError("normalization_backend must be torch or triton")
    if "objective" not in resolved:
        resolved["objective"] = {
            **copy.deepcopy(DEFAULT_HYDRA),
            "seed": resolved.get("seed", DEFAULT_HYDRA["seed"]),
        }
    elif isinstance(resolved["objective"], dict):
        resolved["objective"].setdefault("seed", resolved.get("seed", DEFAULT_HYDRA["seed"]))
    validate_objective(resolved["objective"])
    if not isinstance(resolved["dictionary_backend"], str) or resolved[
        "dictionary_backend"
    ] not in {"torch", "triton"}:
        raise ValueError("dictionary_backend must be torch or triton")
    if (resolved["objective"] is not None
            and resolved["objective"]["name"] == "raw_hydra_patch_target_v1"
            and resolved["dictionary_backend"] != "torch"):
        raise ValueError("HYDRA patch targets use the torch backend")
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
            RMSNorm(encoder.embed_dim, backend=getattr(encoder, "normalization_backend", "torch")),
            nn.Dropout(0.1), nn.Linear(encoder.embed_dim, classes)
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
