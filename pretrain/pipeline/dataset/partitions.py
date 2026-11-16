"""Official TUH boundaries and explicit source-only subject exclusions."""

import numpy as np

from .groups import assignments
from .schema import SPLITS, bank_fingerprint


def official_subjects(records, groups):
    membership = {}
    for record in records:
        subject = record["subject"]
        if subject not in groups:
            continue
        role = record.get("official_split")
        if role not in {"train", "test"}:
            raise ValueError("Official train/test membership is missing")
        if membership.setdefault(subject, role) != role:
            raise ValueError("A subject crosses official train/test boundaries")
    linked = {}
    for subject, group in groups.items():
        if linked.setdefault(group, membership[subject]) != membership[subject]:
            raise ValueError("Duplicate-linked subjects cross official train/test boundaries")
    return membership


def official_bank(records, index, groups, seed, *, source_only=False):
    official = official_subjects(records, groups)
    training = {s: group for s, group in groups.items() if official[s] == "train"}
    subjects = assignments(training, seed, source_only=True)
    excluded = sorted(s for s in groups if official[s] == "test") if source_only else []
    if not source_only:
        subjects.update({s: "test" for s in groups if official[s] == "test"})
    rows = rows_for_subjects(records, index, subjects)
    return (subjects, rows, bank_fingerprint(subjects, rows)), excluded


def rows_for_subjects(records, index, subjects):
    codes = np.asarray(
        [SPLITS.get(subjects.get(records[int(i)]["subject"]), -1) for i in index["record"]]
    )
    rows = {role: np.flatnonzero(codes == code).astype("<u8") for role, code in SPLITS.items()}
    classes = np.unique(index["label"])
    for role, selected in rows.items():
        if role == "test" and not len(selected):
            continue
        if not np.array_equal(np.unique(index["label"][selected]), classes):
            raise ValueError(f"Fixed split {role} lacks class coverage")
    return rows


def validate_bank_membership(metadata, groups, subjects, key):
    """Only a declared official TUAB source bank may omit official test subjects."""
    excluded = set(metadata.get("split_exclusions", {}).get(key, []))
    if set(subjects) & excluded or set(subjects) | excluded != set(groups):
        raise ValueError("Saved subject groups do not cover the selected subjects")
    protocol = metadata.get("release_protocol", {})
    if excluded and not (
        key == "tuab_source/241"
        and metadata["task"] == "tuab"
        and protocol.get("split_rule") == "official_train_80_20_test_v1"
    ):
        raise ValueError("Only official TUAB source pretraining may exclude subjects")
    if protocol.get("split_rule") != "official_train_80_20_test_v1":
        return
    official = official_subjects(metadata["records"], groups)
    for subject, role in official.items():
        assigned = subjects.get(subject)
        if role == "train" and assigned not in {"train", "validation"}:
            raise ValueError("Official training subjects must stay in train/validation")
        if role == "test":
            expected = None if key.startswith("tuab_source/") else "test"
            if assigned != expected or (expected is None and subject not in excluded):
                raise ValueError("Official test subjects cannot enter training or validation")
