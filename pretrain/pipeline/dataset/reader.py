"""Lazy per-worker HDF5 window reads using immutable, persisted split indices."""

from dataclasses import dataclass
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from pretrain.channel_identity import load_channel_vocabulary
from pretrain.metadata import fingerprint
from .groups import subject_groups, validate_membership
from .provenance import validate_provenance
from .schema import INDEX_DTYPE, SPLITS, array_hash, bank_fingerprint


@dataclass(frozen=True)
class DatasetReference:
    path: Path
    seed: int = 241
    scope: str = "subject_holdout"

    def __post_init__(self):
        object.__setattr__(self, "path", Path(self.path).resolve())

    def __fspath__(self):
        return str(self.path)


def reference(value, seed=None, scope=None):
    existing = value if isinstance(value, DatasetReference) else DatasetReference(Path(value))
    return DatasetReference(
        existing.path,
        existing.seed if seed is None else seed,
        existing.scope if scope is None else scope,
    )


def _read(handle, selected):
    metadata = json.loads(handle["metadata"].asstr()[()])
    release = metadata["fingerprint"]
    if metadata.get("schema_version") != 2 or release != fingerprint(
        {k: v for k, v in metadata.items() if k != "fingerprint"}
    ):
        raise ValueError("Invalid HDF5 dataset metadata")
    if metadata["vocabulary_sha256"] != load_channel_vocabulary().fingerprint:
        raise ValueError("Channel vocabulary changed")
    if [json.loads(row) for row in handle["records"].asstr()[:]] != metadata["records"]:
        raise ValueError("Recording metadata changed")
    if handle["subjects"].asstr()[:].tolist() != metadata["source_subjects"]:
        raise ValueError("Subject metadata changed")
    for key, expected in [
        ("names", metadata["channels"]),
        ("ids", metadata["channel_ids"]),
        ("xyz", metadata["channel_xyz"]),
    ]:
        values = (
            handle[f"channels/{key}"].asstr()[:] if key == "names" else handle[f"channels/{key}"][:]
        )
        if not np.array_equal(values, expected):
            raise ValueError("Physical channel order or XYZ changed")
    index = handle["window_index"][:]
    if index.dtype != INDEX_DTYPE or array_hash(index) != metadata["windows_sha256"]:
        raise ValueError("Window index changed")
    if handle["X"].shape != (
        len(index),
        len(metadata["channels"]),
        metadata["window_samples"],
    ) or handle["X"].dtype != np.dtype("float32"):
        raise ValueError("Invalid HDF5 signal layout")
    if not np.array_equal(handle["y"][:], index["label"]):
        raise ValueError("Window labels changed")
    validate_provenance(metadata, index)
    key = f"{selected.scope}/{selected.seed}"
    if key not in metadata["split_banks"] or f"splits/{key}" not in handle:
        raise ValueError(f"HDF5 has no saved split {key}; prepare the requested seed first")
    group = handle[f"splits/{key}"]
    subjects = json.loads(group.attrs["subjects"])
    rows = {role: group[f"{role}_idx"][:] for role in SPLITS}
    digest = bank_fingerprint(subjects, rows)
    if digest != group.attrs["fingerprint"] or digest != metadata["split_banks"][key]:
        raise ValueError("Saved subject split changed")
    groups = subject_groups(metadata["records"], index)
    if "subject_groups" in metadata and groups != metadata["subject_groups"]:
        raise ValueError("Saved duplicate subject groups changed")
    validate_membership(groups, subjects)
    codes = np.asarray(
        [
            SPLITS[subjects[metadata["records"][int(number)]["subject"]]]
            for number in index["record"]
        ],
        dtype="u1",
    )
    for role, code in SPLITS.items():
        if not np.array_equal(rows[role], np.flatnonzero(codes == code)):
            raise ValueError("Window crosses declared subject membership")
    metadata.update(
        subjects=subjects,
        split_seed=selected.seed,
        split_scope=selected.scope,
        split_policy=(
            "grouped_tuab_source_80_20_v1"
            if "subject_groups" in metadata
            else "tuab_source_80_20_v1"
        )
        if selected.scope == "tuab_source"
        else metadata["split_policy"],
        counts={role: len(values) for role, values in rows.items()},
        release_id=release,
        windows_sha256=fingerprint([array_hash(index), digest]),
    )
    metadata.pop("fingerprint")
    metadata["fingerprint"] = fingerprint(metadata)
    return metadata, index, rows


def read_metadata(value):
    selected = reference(value)
    with h5py.File(selected.path, "r") as handle:
        return _read(handle, selected)[0]


class Windows(Dataset):
    def __init__(self, dataset, split, *, verify_signals=True, seed=None, scope=None):
        self.reference = reference(dataset, seed, scope)
        self.path = self.reference.path
        with h5py.File(self.path, "r") as handle:
            self.metadata, self.index, rows = _read(handle, self.reference)
        if self.metadata["label_evidence"].get("status") != "verified":
            raise ValueError("Unverified task labels")
        self.rows, self.split = rows[split], split
        self.ids = torch.tensor(self.metadata["channel_ids"], dtype=torch.long)
        self.verify_signals = verify_signals
        self._file, self._pid = None, None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if self._file is None or self._pid != os.getpid():
            self.close()
            self._file = h5py.File(self.path, "r")
            self._pid = os.getpid()
        row = int(self.rows[index])
        event = self.index[row]
        values = self._file["X"][row]
        if not np.isfinite(values).all() or (
            self.verify_signals and array_hash(values) != event["signal_sha256"].decode()
        ):
            raise ValueError("Prepared window changed")
        return {
            "eeg": torch.from_numpy(values),
            "channel_ids": self.ids,
            "label": int(event["label"]),
            "window_id": event["window_id"].decode(),
        }

    def close(self):
        if getattr(self, "_file", None) is not None:
            self._file.close()
        self._file, self._pid = None, None

    def __getstate__(self):
        return {**self.__dict__, "_file": None, "_pid": None}

    def __del__(self):
        self.close()
