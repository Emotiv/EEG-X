import json

import numpy as np
import pytest

from pretrain.pipeline.data import PreparedWriter, Windows
from pretrain.pipeline.preprocessing import preprocess, profile


def prepared_fixture(root):
    writer = PreparedWriter(
        root,
        task="fixture",
        window_samples=256,
        channels=["C1", "Pz"],
        label_evidence={"status": "verified", "kind": "synthetic_fixture"},
        preprocessing=profile(),
    )
    for subject in range(10):
        signal = np.random.default_rng(subject).normal(size=(2, 1024)).astype("float32")
        writer.add(
            signal,
            subject=str(subject),
            recording=f"record-{subject}",
            source={"sha256": str(subject) * 64},
            events=[(0, 0), (256, 1), (512, 0), (768, 1)],
        )
    metadata = writer.finish()
    metadata["non_rasr_evidence"] = {
        "status": "verified",
        "upstream_rasr": False,
        "authority": "synthetic test generator",
        "sha256": "f" * 64,
    }
    from pretrain.metadata import atomic_json, fingerprint

    metadata.pop("fingerprint")
    metadata["fingerprint"] = fingerprint(metadata)
    atomic_json(root / "manifest.json", metadata)
    return root / "manifest.json"


def test_prepared_windows_preserve_disjoint_subjects_and_labels(tmp_path):
    path = prepared_fixture(tmp_path / "prepared")
    views = {split: Windows(path, split) for split in ("train", "validation", "test")}
    assert {name: len(view) for name, view in views.items()} == {
        "train": 28,
        "validation": 4,
        "test": 8,
    }
    for view in views.values():
        assert view[0]["eeg"].shape == (2, 256)
    metadata = json.loads(path.read_text())
    assert len(metadata["subjects"]) == 10


def test_prepared_signal_mutation_is_rejected(tmp_path):
    path = prepared_fixture(tmp_path / "prepared")
    record = path.parent / "record-000000.npy"
    np.save(record, np.zeros((2, 1024), dtype=np.float32))
    with pytest.raises(ValueError, match="changed"):
        for split in ("train", "validation", "test"):
            Windows(path, split)


def test_unknown_labels_do_not_become_verified(tmp_path):
    with pytest.raises(ValueError, match="Verified label"):
        PreparedWriter(
            tmp_path / "bad",
            task="fixture",
            window_samples=256,
            channels=["C1", "Pz"],
            label_evidence={},
            preprocessing=profile(),
        )


def test_duplicate_windows_are_excluded_across_distinct_recordings(tmp_path):
    writer = PreparedWriter(
        tmp_path / "duplicates",
        task="fixture",
        window_samples=256,
        channels=["C1", "Pz"],
        label_evidence={"status": "verified"},
        preprocessing=profile(),
    )
    shared = np.random.default_rng(123).normal(size=(2, 256)).astype("float32")
    for subject in range(10):
        values = np.random.default_rng(subject).normal(size=(2, 1280)).astype("float32")
        values[:, :256] = shared
        writer.add(
            values,
            subject=str(subject),
            recording=f"subject-{subject}",
            source={"sha256": "a" * 64},
            events=[(0, 0), (256, 0), (512, 1)],
        )
    metadata = writer.finish()
    duplicates = [item for item in metadata["excluded"] if item["reason"] == "duplicate_window"]
    assert len(duplicates) == 10
    assert sum(metadata["counts"].values()) == 20


def test_preprocessing_is_recording_level_and_resampled():
    rate = 256
    time = np.arange(rate * 20) / rate
    amplitude = np.where(time < 10, 1e-6, 10e-6)
    signal = amplitude * np.sin(2 * np.pi * 10 * time)
    output = preprocess(np.stack((signal, -signal)), rate, ["C1", "Pz"])
    assert output.shape == (2, 2560)
    assert output.dtype == np.float32
    assert np.isfinite(output).all()
    assert output[0, :1000].std() < output[0, -1000:].std() / 5
    np.testing.assert_allclose(output.mean(-1), 0, atol=1e-6)
    np.testing.assert_allclose(output.std(-1), 1, atol=1e-6)


def test_seed_bank_pools_official_partitions_and_preserves_signal_hashes(tmp_path):
    from pretrain.pipeline.splits import split_manifest

    source = prepared_fixture(tmp_path / "source")
    original = json.loads(source.read_text())
    assignments = []
    for seed in (241, 242, 243, 244, 245):
        manifest = split_manifest(source, tmp_path / str(seed), seed)
        metadata = json.loads(manifest.read_text())
        assignments.append(metadata["subjects"])
        assert metadata["split_seed"] == seed
        assert [r["sha256"] for r in metadata["records"]] == [
            r["sha256"] for r in original["records"]
        ]
        groups = [
            {s for s, role in metadata["subjects"].items() if role == name}
            for name in ("train", "validation", "test")
        ]
        assert not groups[0] & groups[1] and not groups[1] & groups[2] and not groups[0] & groups[2]
    assert len({json.dumps(group, sort_keys=True) for group in assignments}) == 5


def test_upstream_rasr_evidence_is_optional_but_input_contract_is_checked(tmp_path):
    from pretrain.pipeline.splits import validate_prepared_input

    manifest = prepared_fixture(tmp_path / "data")
    metadata = json.loads(manifest.read_text())
    metadata.pop("non_rasr_evidence")
    validate_prepared_input(metadata)
    metadata["preprocessing"]["clip"] = 999
    with pytest.raises(ValueError, match="preprocessing differs"):
        validate_prepared_input(metadata)
