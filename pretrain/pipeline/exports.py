"""Local encoder release bundles with the exact data and numerical contracts."""

import json
import shutil
from pathlib import Path

import torch

from pretrain.channel_identity import sha256_file
from pretrain.pipeline.checkpoint import save
from pretrain.pipeline.downstream import load_encoder
from pretrain.metadata import atomic_json, fingerprint


def export(checkpoint, manifest, output):
    root = Path(output)
    encoder, payload = load_encoder(checkpoint)
    metadata = json.loads(Path(manifest).read_text())
    actual = fingerprint({k: v for k, v in metadata.items() if k != "fingerprint"})
    if metadata["fingerprint"] != actual or actual != payload["contract"]["dataset"]:
        raise ValueError("Release manifest differs from selected checkpoint")
    repository = Path(__file__).resolve().parents[2]
    if sha256_file(repository / "uv.lock") != payload["contract"]["uv_lock"]:
        raise ValueError("Release environment lock differs from selected checkpoint")
    if (
        sha256_file(repository / "pretrain" / "senloc_file" / "senloc.json")
        != metadata["vocabulary_sha256"]
    ):
        raise ValueError("Release channel vocabulary differs from selected checkpoint")
    root.mkdir(parents=True, exist_ok=False)
    save(
        root / "encoder.pth",
        {
            "encoder": encoder.state_dict(),
            "config": payload["config"],
            "contract": payload["contract"],
        },
    )
    atomic_json(
        root / "model.json",
        {
            "config": payload["config"],
            "contract": payload["contract"],
            "pretraining_checkpoint_sha256": sha256_file(checkpoint),
        },
    )
    atomic_json(root / "data.json", metadata)
    shutil.copy2(repository / "pretrain" / "senloc_file" / "senloc.json", root / "channels.json")
    shutil.copy2(repository / "uv.lock", root / "uv.lock")
    size = payload["config"]["size"]
    model_config = payload["model"]["_extra_state"]
    objective = "Soft DiCT plus raw MSE" if model_config["objective"] else "Raw MSE"
    card = f"""# EEG-X revised encoder

This encoder uses aligned raw and STFT magnitude patches with the EEG-X context encoder. Only completed revised experiments can supply paper results.

| Property | Value |
|---|---|
| Size | {size} |
| Objective | {objective} |
| Task | {metadata["task"]} |
| Seed | {payload["config"]["seed"]} |
| Sample rate | {metadata["sample_rate"]} Hz |
| Pretraining window | {metadata["window_samples"]} samples |
| Normalization | RMSNorm, including attention Q/K |
| Representation | Mean patch tokens; no special token |
| Spatial encoding | Fixed normalized XYZ sin/cos, base five |

The bundle records the source-backed channel order, preprocessing and fixed subject membership. Apply that preprocessing once to complete acquisition recordings before cutting windows.

Use the repository loader with its locked environment. Downstream measurements are saved with their predictions and logged to W&B when enabled.
"""
    (root / "MODEL_CARD.md").write_text(card)
    atomic_json(
        root / "checksums.json",
        {path.name: sha256_file(path) for path in sorted(root.iterdir()) if path.is_file()},
    )
    return root


def load(path):
    from pretrain.models.factory import create_model

    payload = torch.load(Path(path) / "encoder.pth", map_location="cpu", weights_only=True)
    model = create_model(payload["config"])
    model.encoder.load_state_dict(payload["encoder"], strict=True)
    return model.encoder, payload
