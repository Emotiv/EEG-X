"""Checkpoint-owned encoders and audited supervised optimizer groups."""

import re

import torch

from pretrain.pipeline.checkpoint import dataset_identity, set_seed, stream_seed, software_contract
from pretrain.models.factory import Classifier, create_model, resolve_model_config


TRANSFER_TASKS = {"crowdsourced", "bciciv_2a", "kalunga", "physionet_mi"}


def validate_pretraining_model(payload, config, *, target_task=None):
    """A selected encoder must use the requested model and training seed."""
    expected = resolve_model_config(config)
    pretrained = payload["config"]
    keys = ("model", "depth", "width", "heads")
    if target_task not in TRANSFER_TASKS:
        keys += ("seed",)
    if any(pretrained.get(key) != expected.get(key) for key in keys):
        raise ValueError("The pretraining checkpoint must match the configured seed and model")


def validate_pretraining_source(payload, config, manifest, *, mode):
    """Enforce in-domain pairing and the agreed TUAB transfer source."""
    validate_pretraining_model(payload, config, target_task=manifest["task"])
    source = payload["contract"].get("dataset_identity")
    if not isinstance(source, dict) or any(
        not isinstance(source.get(key), str) or not source[key]
        for key in ("task", "release_sha256", "split_sha256")
    ):
        raise ValueError(
            "The pretraining checkpoint lacks source dataset provenance; train a fresh model"
        )
    target = dataset_identity(manifest)
    if mode not in {"linear_probe", "finetune"}:
        raise ValueError("Downstream mode must be linear_probe or finetune")
    if (
        config.get("purpose", "").startswith("paper_")
        and payload["contract"]["config"].get("purpose") != "paper_pretrain"
    ):
        raise ValueError("Paper evaluation requires a checkpoint from verified paper pretraining")
    transfer = manifest["task"] in TRANSFER_TASKS
    if transfer:
        if source["task"] != "tuab" or source.get("scope") != "tuab_source":
            raise ValueError("Cross-domain evaluation requires the shared TUAB source checkpoint")
        return
    if source != target or payload["contract"]["dataset"] != manifest["fingerprint"]:
        raise ValueError(
            "In-domain evaluation requires the same dataset and subject split as pretraining"
        )


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
