"""Checkpoint-owned encoders and audited supervised optimizer groups."""

import re

import torch

from pretrain.pipeline.checkpoint import set_seed, stream_seed, software_contract
from pretrain.models.factory import Classifier, create_model


def load_encoder(checkpoint):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if any(payload["contract"].get(key) != value for key, value in software_contract().items()):
        raise ValueError("Selected checkpoint code or environment differs from this pipeline")
    model = create_model(payload["config"])
    model.load_state_dict(payload["model"], strict=True)
    return model.encoder, payload


def classifier(checkpoint, classes, seed):
    encoder, payload = load_encoder(checkpoint)
    set_seed(stream_seed(seed, "supervised_head"))
    return Classifier(encoder, classes), payload


def optimizer_groups(model):
    """Every parameter is assigned once; unknown names are an audit failure."""
    depth = len(model.encoder.contex_encoder.TRMs)
    maximum = depth + 2
    grouped = {}
    names = {}
    seen = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in seen:
            raise ValueError("A supervised parameter appears in multiple groups")
        seen.add(id(parameter))
        if name.startswith("head."):
            layer = maximum
        elif name.startswith(("encoder.raw_projection.", "encoder.spectral_projection.")):
            layer = 0
        elif match := re.match(r"encoder\.contex_encoder\.TRMs\.(\d+)\.", name):
            layer = int(match[1]) + 1
        elif name.startswith("encoder.norm."):
            layer = depth + 1
        else:
            raise ValueError(f"Unmapped supervised parameter: {name}")
        decay = 0.01 if parameter.ndim > 1 else 0.0
        key = (layer, decay)
        grouped.setdefault(key, []).append(parameter)
        names.setdefault(key, []).append(name)
    groups, audit = [], []
    for (layer, decay), parameters in sorted(grouped.items()):
        scale = 0.75 ** (maximum - layer)
        groups.append(
            {"params": parameters, "weight_decay": decay, "lr": 1e-3 * scale, "lr_scale": scale}
        )
        audit.append(
            {
                "layer": layer,
                "weight_decay": decay,
                "lr_scale": scale,
                "names": names[(layer, decay)],
                "elements": sum(p.numel() for p in parameters),
            }
        )
    if len(seen) != sum(1 for p in model.parameters() if p.requires_grad):
        raise ValueError("Incomplete supervised optimizer routing")
    return groups, audit
