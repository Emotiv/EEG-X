"""Sample multiplicity and linked-subject isolation through real HDF5 readers."""

import json

import h5py
import numpy as np
import pytest

from pretrain.metadata import fingerprint
from pretrain.pipeline.dataset import DatasetReference, PreparedWriter, Windows, read_metadata
from pretrain.pipeline.dataset.schema import array_hash, bank_fingerprint
from pretrain.pipeline.preprocessing import profile


def linked_dataset(path):
    writer = PreparedWriter(
        path,
        task="fixture",
        window_samples=256,
        channels=["C1", "Pz"],
        label_evidence={"status": "verified"},
        preprocessing=profile(),
    )
    shared = np.random.default_rng(0).normal(size=(2, 768)).astype("float32")
    for subject in range(10):
        signal = np.random.default_rng(subject).normal(size=(2, 768)).astype("float32")
        if subject in (0, 1):
            signal[:] = shared
        elif subject == 3:
            signal[:, :256] = shared[:, :256]
        writer.add(
            signal,
            subject=str(subject),
            recording=f"record-{subject}",
            source={"sha256": fingerprint(subject)},
            events=[(0, 0), (256, 0), (512, 1)],
        )
    writer.finish()
    return path


def test_duplicate_records_and_windows_are_retained_and_transitively_grouped(tmp_path):
    path = linked_dataset(tmp_path / "grouped.h5")
    for seed in range(241, 246):
        metadata = read_metadata(DatasetReference(path, seed))
        assert sum(metadata["counts"].values()) == 30
        assert metadata["subjects"]["0"] == metadata["subjects"]["1"]
        assert metadata["subjects"]["0"] == metadata["subjects"]["3"]
        assert metadata["subject_groups"]["0"] == metadata["subject_groups"]["3"]
        assert metadata["excluded"] == []
        owners = {}
        for role in ("train", "validation", "test"):
            view = Windows(DatasetReference(path, seed), role)
            for number in range(len(view)):
                digest = array_hash(view[number]["eeg"].numpy())
                assert owners.setdefault(digest, role) == role
            view.close()


def test_reader_rejects_duplicate_subject_leak_even_with_rehashed_bank(tmp_path):
    path = linked_dataset(tmp_path / "grouped.h5")
    with h5py.File(path, "r+") as handle:
        metadata = json.loads(handle["metadata"].asstr()[()])
        group = handle["splits/subject_holdout/241"]
        subjects = json.loads(group.attrs["subjects"])
        subjects["1"] = "validation" if subjects["0"] != "validation" else "train"
        index = handle["window_index"][:]
        roles = {
            role: np.asarray(
                [
                    i
                    for i, row in enumerate(index)
                    if subjects[metadata["records"][int(row["record"])]["subject"]] == role
                ],
                dtype="<u8",
            )
            for role in ("train", "validation", "test")
        }
        for role, rows in roles.items():
            del group[f"{role}_idx"]
            group.create_dataset(f"{role}_idx", data=rows)
        digest = bank_fingerprint(subjects, roles)
        group.attrs["subjects"] = json.dumps(subjects, sort_keys=True)
        group.attrs["fingerprint"] = digest
        metadata["split_banks"]["subject_holdout/241"] = digest
        metadata.pop("fingerprint")
        metadata["fingerprint"] = fingerprint(metadata)
        handle["metadata"][()] = json.dumps(metadata, sort_keys=True)
    with pytest.raises(ValueError, match="Duplicate-linked subjects"):
        read_metadata(path)


def test_stride_controls_real_saved_windows_and_tail_accounting(tmp_path):
    writer = PreparedWriter(
        tmp_path / "overlap.h5",
        task="fixture",
        window_samples=256,
        channels=["C1", "Pz"],
        label_evidence={"status": "verified"},
        preprocessing=profile(),
        sampling_policy={"stride_samples": 32},
    )
    for subject in range(10):
        for label in (0, 1):
            writer.add(
                np.random.default_rng(subject * 2 + label).normal(size=(2, 547)),
                subject=str(subject),
                recording=f"record-{subject}-{label}",
                source={"sha256": fingerprint([subject, label])},
                label=label,
            )
    metadata = writer.finish()
    assert sum(metadata["counts"].values()) == 200
    assert len(metadata["excluded"]) == 20
    assert all(row["samples"] == 3 and row["start"] == 544 for row in metadata["excluded"])
    with h5py.File(writer.path, "r") as handle:
        assert set(handle["window_index"][:]["start"]) == set(range(0, 289, 32))


def test_timing_exception_does_not_waive_reference_count_validation(tmp_path):
    writer = PreparedWriter(
        tmp_path / "mismatch.h5",
        task="fixture",
        window_samples=256,
        channels=["C1", "Pz"],
        label_evidence={"status": "verified"},
        preprocessing=profile(),
        sampling_policy={
            "class_counts": [1, 1],
            "pilot": False,
            "exception": "duration_correction",
        },
    )
    for subject in range(10):
        writer.add(
            np.random.default_rng(subject).normal(size=(2, 512)),
            subject=str(subject),
            recording=f"record-{subject}",
            source={"sha256": fingerprint(subject)},
            events=[(0, 0), (256, 1)],
        )
    with pytest.raises(ValueError, match="Reference class counts differ"):
        writer.finish()
    assert not writer.path.exists()
