"""Copy prepared EEG unchanged and publish audited fixed subject memberships."""

import json
import os
from pathlib import Path
import shutil

import h5py
import numpy as np

from pretrain.channel_identity import sha256_file
from pretrain.metadata import atomic_json, fingerprint
from pretrain.pipeline.dataset import DatasetReference, read_metadata
from pretrain.pipeline.dataset.partitions import official_bank


def read_bank(handle, key):
    group = handle[f"splits/{key}"]
    return (
        json.loads(group.attrs["subjects"]),
        {r: group[f"{r}_idx"][:] for r in ("train", "validation", "test")},
        str(group.attrs["fingerprint"]),
    )


def write_release_metadata(output, protocol_path, manifest):
    """Keep the complete candidate-score evidence beside uploadable HDF5 files."""
    evidence = Path(protocol_path).parent / "split-selection-evidence.json"
    required = any(s["rule"] == "best_reference_test_bal_acc"
                   for s in manifest["protocol"]["datasets"].values())
    if required and not evidence.is_file():
        raise FileNotFoundError("Test-informed split selection requires its evidence JSON")
    if evidence.is_file():
        shutil.copy2(evidence, Path(output) / evidence.name)
        manifest["selection_evidence"] = {"file": evidence.name, "sha256": sha256_file(evidence)}
    atomic_json(Path(output) / "protocol.json", manifest["protocol"])
    atomic_json(Path(output) / "manifest.json", manifest)
    return manifest


def make_dataset(source, destination, protocol, selection):
    seed = selection["split_seed"]
    original = read_metadata(DatasetReference(source, seed))
    expected = selection.get("source_release")
    if expected and original["release_id"] != expected:
        raise ValueError(f"Reference release changed: {source.name}")
    shutil.copy2(source, destination)
    with h5py.File(destination, "r+") as handle:
        metadata = json.loads(handle["metadata"].asstr()[()])
        index = handle["window_index"][:]
        key = f"subject_holdout/{seed}"
        if selection["rule"] == "official":
            selected, _ = official_bank(metadata["records"], index, metadata["subject_groups"], seed)
            banks = {key: selected}
            if metadata["task"] == "tuab":
                source_bank, excluded = official_bank(
                    metadata["records"], index, metadata["subject_groups"], 241, source_only=True
                )
                banks["tuab_source/241"] = source_bank
                metadata["split_exclusions"] = {"tuab_source/241": excluded}
                metadata["source_split_policy"] = "official_train_only_80_20_v1"
            policy = "official_train_80_20_test_v1"
        else:
            banks = {key: read_bank(handle, key)}
            if selection.get("source_bank") != banks[key][2]:
                raise ValueError(f"Reference split changed: {source.name}")
            policy = "fixed_saved_subject_holdout_v1"
        subjects, rows, _ = banks[key]
        metadata.update(
            subjects=subjects,
            split_seed=seed,
            split_policy=policy,
            split_banks={k: b[2] for k, b in banks.items()},
            counts={r: len(v) for r, v in rows.items()},
            release_protocol={
                "name": protocol["name"],
                "split_rule": policy,
                "parent_release": original["release_id"],
                "selection": selection,
            },
        )
        metadata.pop("fingerprint")
        metadata["fingerprint"] = fingerprint(metadata)
        del handle["metadata"]
        handle.create_dataset("metadata", data=json.dumps(metadata, sort_keys=True),
                              dtype=h5py.string_dtype("utf-8"))
        del handle["splits"]
        for bank_key, (members, indices, digest) in banks.items():
            group = handle.require_group(f"splits/{bank_key}")
            group.attrs["subjects"] = json.dumps(members, sort_keys=True)
            group.attrs["fingerprint"] = digest
            for role, values in indices.items():
                group.create_dataset(f"{role}_idx", data=values, fletcher32=True)
    entry = {
        "file": destination.name,
        "parent_release": original["release_id"],
        "signal_catalog_sha256": metadata["windows_sha256"],
        "selection": selection,
        "banks": {},
    }
    for bank_key in banks:
        scope, bank_seed = bank_key.split("/")
        checked = read_metadata(DatasetReference(destination, int(bank_seed), scope))
        entry["release_id"] = checked["release_id"]
        entry["banks"][bank_key] = {
            "fingerprint": banks[bank_key][2],
            "subjects": checked["subjects"],
            "counts": checked["counts"],
        }
        if checked["preprocessing"] != original["preprocessing"]:
            raise ValueError("Signal preprocessing changed during split publication")
    # Read and hash every stored window. The old per-window digests bind the new
    # release to its exact EEG payload, including duplicate sample multiplicity.
    from pretrain.pipeline.dataset.schema import array_hash

    with h5py.File(destination, "r") as handle:
        for start in range(0, len(index), 128):
            values = handle["X"][start:start + 128]
            for offset, signal in enumerate(values):
                if not np.isfinite(signal).all() or array_hash(signal) != index[start + offset]["signal_sha256"].decode():
                    raise ValueError(f"Signal payload changed: {destination.name}, row {start + offset}")
    entry["sha256"] = sha256_file(destination)
    entry["bytes"] = destination.stat().st_size
    return entry


def build_release(source, output, protocol_path):
    source, output = Path(source).resolve(), Path(output).resolve()
    protocol = json.loads(Path(protocol_path).read_text())
    if output.exists():
        raise FileExistsError(f"Release already exists: {output}")
    staging = output.with_name(output.name + ".working")
    staging.mkdir(parents=True, exist_ok=False)
    manifest = {"protocol": protocol, "datasets": {}}
    try:
        for task, selection in protocol["datasets"].items():
            filename = task.replace("_", "-") + ".h5"
            print(f"RELEASE {task}: copy, validate splits and audit EEG", flush=True)
            manifest["datasets"][task] = make_dataset(
                source / filename, staging / filename, protocol, selection
            )
            atomic_json(staging / "manifest.json", manifest)
        write_release_metadata(staging, protocol_path, manifest)
        os.rename(staging, output)
    except BaseException:
        # Preserve partial files for diagnosis; they cannot be mistaken for the
        # complete release because its final directory has not been published.
        raise
    return manifest
