"""Atomic snapshots with explicit model, data and run contracts."""

import copy
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from pretrain.channel_identity import sha256_file
from pretrain.rng import capture_rng_state
from pretrain.compute import compute_policy
from pretrain.metadata import fingerprint


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def stream_seed(seed, stream):
    return int(fingerprint([int(seed), stream])[:15], 16) % (2**31 - 1)


def software_contract():
    root = Path(__file__).resolve().parents[2]
    sources = {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted((root / "pretrain").rglob("*.py"))
    }
    sources["Main.py"] = sha256_file(root / "Main.py")
    from importlib.metadata import version

    return {
        "source": fingerprint(sources),
        "sources": sources,
        "torch": str(torch.__version__),
        "python": sys.version,
        "uv_lock": sha256_file(root / "uv.lock"),
        "specification": sha256_file(root / "docs" / "spec.md"),
        "dependencies": {name: version(name) for name in ("numpy", "scipy", "mne", "scikit-learn")},
    }


def source_revision():
    """Record Git provenance without making commit labels resume requirements."""
    root = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}
    return {"commit": revision, "dirty": bool(status)}


def optimizer_settings(optimizer):
    """Save constructor defaults and effective parameter-group settings."""
    return {
        "class": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
        "defaults": copy.deepcopy(optimizer.defaults),
        "groups": [
            {key: copy.deepcopy(value) for key, value in group.items() if key != "params"}
            for group in optimizer.param_groups
        ],
    }


def dataset_identity(dataset):
    """Bind a checkpoint to the source task, release and subject assignment."""
    return {
        "task": dataset["task"],
        "scope": dataset.get("split_scope", "subject_holdout"),
        "release_sha256": dataset.get("release_id", dataset["fingerprint"]),
        "split_sha256": fingerprint(
            {
                "seed": dataset.get("split_seed"),
                "policy": dataset.get("split_policy"),
                "subjects": dataset["subjects"],
                "windows_sha256": dataset["windows_sha256"],
            }
        ),
    }


def contract(config, dataset, *, device="cpu"):
    resolved = torch.device(device)
    execution = {
        "device": str(resolved),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if resolved.type == "cuda" else None,
    }
    execution["requested_compute"] = compute_policy(config)
    execution["deterministic_algorithms"] = config.get("deterministic", True)
    execution["effective_compute"] = (
        compute_policy(config)
        if resolved.type == "cuda"
        else {"matmul": "ieee", "convolution": "ieee", "cudnn_benchmark": False}
    )
    if resolved.type == "cuda":
        execution["gpu"] = torch.cuda.get_device_name(resolved)
        execution["capability"] = list(torch.cuda.get_device_capability(resolved))
    return {
        **software_contract(),
        "config": copy.deepcopy(config),
        "dataset": dataset["fingerprint"],
        "dataset_identity": dataset_identity(dataset),
        "execution": execution,
    }


def save(path, payload):
    path = Path(path)
    partial = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, partial)
    with partial.open("rb") as handle:
        os.fsync(handle.fileno())
    partial.replace(path)


def restore_rng(payload):
    rng = payload["rng"]
    torch.set_rng_state(rng["torch_rng"].cpu())
    random.setstate(rng["python_rng"])
    state = rng["numpy_rng"]
    np.random.set_state(
        (
            state["algorithm"],
            np.asarray(state["keys"], dtype=np.uint32),
            state["position"],
            state["has_gauss"],
            state["cached_gaussian"],
        )
    )
    if rng["cuda_rng"]:
        torch.cuda.set_rng_state_all(rng["cuda_rng"])


def snapshot(model, optimizer, state, run_contract):
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "state": copy.deepcopy(state),
        "contract": run_contract,
        "rng": capture_rng_state(),
    }


def load(path, expected_contract):
    value = torch.load(path, map_location="cpu", weights_only=True)
    if value.get("contract") != expected_contract:
        raise ValueError("Resume rejected: code, data, configuration or environment changed")
    return value


def model_weights(model, config, run_contract):
    return {"model": model.state_dict(), "config": config, "contract": run_contract}


def write_history(path, values):
    path = Path(path)
    partial = path.with_suffix(".partial")
    partial.write_text("".join(json.dumps(item, allow_nan=False) + "\n" for item in values))
    partial.replace(path)
