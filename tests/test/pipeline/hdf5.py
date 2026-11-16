"""Check portable data identity, persisted membership and lazy worker reads."""

import pickle
import shutil

import h5py
import numpy as np
import pytest
import torch

from pretrain.metadata import fingerprint
from pretrain.pipeline.data import PreparedWriter, Windows, batches
from pretrain.pipeline.dataset import DatasetReference, read_metadata
from pretrain.pipeline.preprocessing import profile
from pretrain.pipeline.splits import select_split
from tests.test.pipeline.data import prepared_fixture


def write_dataset(path, *, reverse=False, task="fixture", events=None, separator="/"):
    writer = PreparedWriter(
        path,
        task=task,
        window_samples=256,
        channels=["C1", "Pz"],
        label_evidence={"status": "verified"},
        preprocessing=profile(),
    )
    order = reversed(range(10)) if reverse else range(10)
    for subject in order:
        values = np.random.default_rng(subject).normal(size=(2, 1536)).astype("float32")
        writer.add(
            values,
            subject=str(subject),
            recording=f"session{separator}record-{subject}",
            source={
                "path": str(path.parent / f"raw-{subject}.edf"),
                "sha256": fingerprint(subject),
            },
            events=events or [(0, 0), (256, 1), (512, 0), (768, 1)],
        )
    writer.finish()
    return path


def test_rebuilding_and_relocating_preserves_rows_ids_and_all_seed_splits(tmp_path):
    left = write_dataset(tmp_path / "left/dataset.h5")
    right = write_dataset(tmp_path / "right/dataset.h5", reverse=True)
    assert read_metadata(left) == read_metadata(right)
    with h5py.File(left, "r") as a, h5py.File(right, "r") as b:
        np.testing.assert_array_equal(a["X"][:], b["X"][:])
        np.testing.assert_array_equal(a["window_index"][:], b["window_index"][:])
        for seed in range(241, 246):
            for role in ("train", "validation", "test"):
                key = f"splits/subject_holdout/{seed}/{role}_idx"
                np.testing.assert_array_equal(a[key][:], b[key][:])
    moved = tmp_path / "moved.h5"
    shutil.copy2(left, moved)
    all_ids = []
    for seed in (241, 242):
        identities = set()
        for role in ("train", "validation", "test"):
            view = Windows(DatasetReference(moved, seed), role)
            identities.update(view[i]["window_id"] for i in range(len(view)))
            view.close()
        all_ids.append(identities)
    assert all_ids[0] == all_ids[1]
    assert read_metadata(left) == read_metadata(moved)


def test_dataloader_workers_open_h5_independently_and_keep_shuffled_order(tmp_path):
    path = prepared_fixture(tmp_path / "data")
    view = Windows(path, "train")
    first = view[0]
    restored = pickle.loads(pickle.dumps(view))
    assert restored._file is None
    torch.testing.assert_close(restored[0]["eeg"], first["eeg"])
    view.close()
    restored.close()
    outputs = []
    for workers in (0, 2):
        current = Windows(path, "train")
        identities = []
        for batch in batches(current, batch_size=5, seed=99, workers=workers):
            assert batch["eeg"].shape[1:] == (2, 256)
            identities.extend(batch["window_id"])
        outputs.append(identities)
        current.close()
    assert outputs[0] == outputs[1]


def test_saved_split_mutation_and_unprepared_seed_are_rejected(tmp_path):
    path = prepared_fixture(tmp_path / "data")
    with pytest.raises(ValueError, match="no saved split"):
        Windows(DatasetReference(path, 999), "train")
    with h5py.File(path, "r+") as handle:
        handle["splits/subject_holdout/241/train_idx"][0] = handle[
            "splits/subject_holdout/241/test_idx"
        ][0]
    with pytest.raises(ValueError, match="split changed"):
        Windows(path, "train")


def test_source_split_reuses_the_same_file_without_copying_eeg(tmp_path):
    path = write_dataset(tmp_path / "tuab.h5", task="tuab")
    before = path.read_bytes()
    selected = select_split(path, tmp_path / "run", 241, scope="tuab_source")
    metadata = read_metadata(selected)
    assert selected.path == path.resolve()
    assert metadata["counts"] == {"train": 32, "validation": 8, "test": 0}
    assert set(metadata["subjects"].values()) == {"train", "validation"}
    assert path.read_bytes() == before
    assert [p.name for p in (tmp_path / "run").iterdir()] == ["split.json"]


def test_event_windows_preserve_original_overlap_and_labels(tmp_path):
    path = write_dataset(
        tmp_path / "events.h5", events=[(0, 0), (128, 0), (512, 1), (640, 0), (1024, 1)]
    )
    metadata = read_metadata(path)
    assert sum(metadata["counts"].values()) == 50
    assert metadata["excluded"] == []
    with h5py.File(path, "r") as handle:
        assert set(handle["window_index"][:]["start"]) == {0, 128, 512, 640, 1024}


def test_invalid_dataset_is_not_published(tmp_path):
    path = tmp_path / "invalid.h5"
    writer = PreparedWriter(
        path,
        task="fixture",
        window_samples=256,
        channels=["C1", "Pz"],
        label_evidence={"status": "verified"},
        preprocessing=profile(),
    )
    writer.add(
        np.ones((2, 256)), subject="one", recording="one", source={"sha256": "a" * 64}, label=0
    )
    with pytest.raises(ValueError):
        writer.finish()
    assert not path.exists()
    assert not list(tmp_path.glob("*.working"))
    assert not list(tmp_path.glob("*.partial"))


def test_windows_and_posix_recording_ids_produce_the_same_catalog(tmp_path):
    left = write_dataset(tmp_path / "posix.h5")
    right = write_dataset(tmp_path / "windows.h5", separator="\\")
    assert read_metadata(left) == read_metadata(right)
    with h5py.File(left, "r") as a, h5py.File(right, "r") as b:
        np.testing.assert_array_equal(a["window_index"][:], b["window_index"][:])


def test_duplicate_annotations_are_accounted_for(tmp_path):
    path = write_dataset(tmp_path / "duplicates.h5", events=[(0, 0), (0, 0), (256, 1)])
    metadata = read_metadata(path)
    assert sum(record["event_count"] for record in metadata["records"]) == 30
    assert sum(metadata["counts"].values()) == 30
    assert metadata["excluded"] == []
    with h5py.File(path, "r") as handle:
        index = handle["window_index"][:]
        assert len(set(index["window_id"])) == 30
        assert len(set(index["signal_sha256"])) == 20


def test_final_metadata_is_validated_before_publication(tmp_path):
    path = tmp_path / "invalid-final.h5"
    writer = PreparedWriter(
        path,
        task="fixture",
        window_samples=256,
        channels=["C1", "Pz"],
        label_evidence={"status": "verified"},
        preprocessing=profile(),
    )
    for subject in range(10):
        values = np.random.default_rng(subject).normal(size=(2, 512))
        writer.add(
            values,
            subject=str(subject),
            recording=f"record-{subject}",
            source={"sha256": fingerprint(subject)},
            events=[(0, 0), (256, 1)],
        )

    def invalid(metadata):
        metadata["window_samples"] = 288
        return metadata

    with pytest.raises(ValueError, match="signal layout"):
        writer.finish(finalize=invalid)
    assert not path.exists()
    assert not list(tmp_path.glob("*.working"))
    assert not list(tmp_path.glob("*.partial"))


def test_source_audit_is_captured_after_adapter_iteration(tmp_path):
    from pretrain.pipeline.prepare import prepare_adapter

    class Source:
        task, window_samples, channels = "fixture", 256, ["C1", "Pz"]

        def __init__(self):
            self.label_evidence = {"status": "verified", "record_audit": []}

        def recordings(self):
            for subject in range(10):
                self.label_evidence["record_audit"].append({"recording": f"record-{subject}"})
                yield {
                    "volts": np.random.default_rng(subject).normal(scale=1e-6, size=(2, 1024)),
                    "sample_rate": 128,
                    "subject": str(subject),
                    "recording": f"record-{subject}",
                    "source": {"sha256": fingerprint(subject)},
                    "events": [(0, 0), (256, 1)],
                }

    metadata = prepare_adapter(Source(), tmp_path / "audit.h5")
    assert len(metadata["label_evidence"]["record_audit"]) == 10
    parallel = prepare_adapter(Source(), tmp_path / "parallel.h5", workers=2)
    assert metadata == parallel
    with h5py.File(tmp_path / "audit.h5", "r") as a, h5py.File(tmp_path / "parallel.h5", "r") as b:
        np.testing.assert_array_equal(a["X"][:], b["X"][:])
        np.testing.assert_array_equal(a["window_index"][:], b["window_index"][:])


def test_same_onset_labels_and_duplicates_have_canonical_order(tmp_path):
    events = [(0, 1), (0, 0), (0, 1), (256, 1), (256, 0)]
    left = write_dataset(tmp_path / "left.h5", events=events)
    right = write_dataset(tmp_path / "right.h5", events=list(reversed(events)), reverse=True)
    assert read_metadata(left) == read_metadata(right)
    with h5py.File(left, "r") as a, h5py.File(right, "r") as b:
        np.testing.assert_array_equal(a["window_index"][:], b["window_index"][:])
        np.testing.assert_array_equal(a["X"][:], b["X"][:])
