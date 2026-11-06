"""Portable identities and frozen subject splits for prepared HDF5 datasets."""

import hashlib
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np

from pretrain.metadata import fingerprint


SEEDS = (241, 242, 243, 244, 245)
SPLITS = {"train": 0, "validation": 1, "test": 2}
TASK_WINDOWS = {
    "dreamer_arousal": 256,
    "stew": 256,
    "tuab": 1280,
    "tuev_binary": 256,
    "crowdsourced": 256,
    "bciciv_2a": 256,
    "kalunga": 256,
    "physionet_mi": 256,
}
INDEX_DTYPE = np.dtype(
    [
        ("record", "<u4"),
        ("start", "<u8"),
        ("label", "<i4"),
        ("window_id", "S64"),
        ("signal_sha256", "S64"),
    ]
)


def array_hash(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def portable(value):
    """Keep source evidence, excluding host-specific absolute path prefixes."""
    if isinstance(value, dict):
        return {str(key): portable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [portable(item) for item in value]
    if isinstance(value, str):
        if PureWindowsPath(value).is_absolute():
            return PureWindowsPath(value).name
        if Path(value).is_absolute():
            return Path(value).name
    return value


def recording_id(value):
    path = PurePosixPath(str(value).replace("\\", "/"))
    if path.is_absolute() or PureWindowsPath(str(value)).drive or ".." in path.parts:
        raise ValueError("Recording IDs must be source-relative, not host paths")
    return path.as_posix()


def bank_fingerprint(subjects, rows):
    return fingerprint(
        {"subjects": subjects, "indices": {role: array_hash(data) for role, data in rows.items()}}
    )


def subject_split(records, seed=241, *, source_only=False, groups=None):
    from .groups import assignments

    subjects = {record["subject"] for record in records}
    groups = {subject: subject for subject in subjects} if groups is None else groups
    return assignments(groups, seed, source_only=source_only)


def split_bank(records, index, seed, scope, groups=None):
    used = set(index["record"].tolist())
    assignments = subject_split(
        [record for i, record in enumerate(records) if i in used],
        seed,
        source_only=scope == "tuab_source",
        groups=groups,
    )
    codes = np.asarray(
        [SPLITS[assignments[records[int(i)]["subject"]]] for i in index["record"]], dtype="u1"
    )
    classes = np.unique(index["label"])
    rows = {role: np.flatnonzero(codes == code).astype("<u8") for role, code in SPLITS.items()}
    for role, selected in rows.items():
        if scope == "tuab_source" and role == "test":
            continue
        if not np.array_equal(np.unique(index["label"][selected]), classes):
            raise ValueError(f"Seed {seed} split {role} lacks class coverage")
    digest = bank_fingerprint(assignments, rows)
    return assignments, rows, digest
