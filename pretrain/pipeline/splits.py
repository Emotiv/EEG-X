"""Immutable per-seed subject membership over unchanged prepared signals."""

import copy
import json
from pathlib import Path

import numpy as np

from pretrain.channel_identity import sha256_file
from pretrain.pipeline.data import SPLITS, Windows, subject_split
from pretrain.metadata import atomic_json, fingerprint


def split_manifest(manifest, output, seed):
    source = Path(manifest).resolve()
    view = Windows(source, "train")
    metadata = copy.deepcopy(view.metadata)
    index = np.array(view.index, copy=True)
    used = set(index["record"].tolist())
    records = [r for i, r in enumerate(metadata["records"]) if i in used]
    assignments = subject_split(records, seed)
    for number, record in enumerate(metadata["records"]):
        record["path"] = str((source.parent / record["path"]).resolve())
        if sha256_file(record["path"]) != record["sha256"]:
            raise ValueError("Prepared signal changed before split generation")
        index["split"][index["record"] == number] = SPLITS.get(
            assignments.get(record["subject"]), 255
        )
    classes = metadata["classes"]
    for role, code in SPLITS.items():
        if np.unique(index["label"][index["split"] == code]).tolist() != classes:
            raise ValueError(f"Seed {seed} split {role} lacks class coverage")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    np.save(root / "windows.npy", index)
    metadata.update(
        subjects=assignments,
        split_seed=seed,
        split_policy="pooled_subject_70_10_20_v1",
        parent_manifest_sha256=sha256_file(source),
        windows_sha256=sha256_file(root / "windows.npy"),
        counts={role: int((index["split"] == code).sum()) for role, code in SPLITS.items()},
    )
    metadata.pop("fingerprint", None)
    metadata["fingerprint"] = fingerprint(metadata)
    atomic_json(root / "manifest.json", metadata)
    return root / "manifest.json"


def require_paper_source(metadata):
    """Local rasr=False never proves the processing history of an input."""
    evidence = metadata.get("non_rasr_evidence", {})
    if evidence.get("status") != "verified" or not evidence.get("sha256"):
        raise ValueError("Verified non-rASR source history is missing")
    if evidence.get("upstream_rasr") is not False or not evidence.get("authority"):
        raise ValueError("Non-rASR evidence needs an authority and explicit upstream policy")
    if metadata["preprocessing"].get("rasr") is not False:
        raise ValueError("Prepared inputs must not apply rASR")
    from pretrain.pipeline.preprocessing import profile

    for key, value in profile().items():
        if key != "name" and metadata["preprocessing"].get(key) != value:
            raise ValueError(f"Prepared preprocessing differs from the accepted contract: {key}")


def attach_source_evidence(metadata, evidence_path):
    evidence = json.loads(Path(evidence_path).read_text())
    expected = sorted({r["source"]["sha256"] for r in metadata["records"]})
    if sorted(evidence.get("source_sha256", [])) != expected:
        raise ValueError("Non-rASR evidence does not identify all source artifacts")
    metadata["non_rasr_evidence"] = {**evidence, "sha256": sha256_file(evidence_path)}
    require_paper_source(metadata)
    metadata.pop("fingerprint", None)
    metadata["fingerprint"] = fingerprint(metadata)
    return metadata


def validate_run_dataset(config, metadata):
    """Engineering checks cannot silently become paper experiments."""
    if config.get("purpose", "").startswith("paper_"):
        require_paper_source(metadata)
        if metadata.get("split_seed") != config["seed"]:
            raise ValueError("Paper experiments require the matching per-seed split")
