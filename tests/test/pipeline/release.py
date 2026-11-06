"""Fixed membership must survive new training seeds and official source exclusion."""

import json

import h5py
import numpy as np
import pytest
import yaml

from pretrain.metadata import fingerprint
from pretrain.pipeline.dataset import DatasetReference, PreparedWriter, Windows, read_metadata
from pretrain.pipeline.dataset.schema import bank_fingerprint
from pretrain.pipeline.preprocessing import profile
from pretrain.pipeline.release.build import build_release
from pretrain.train import run_config


def official_fixture(folder):
    folder.mkdir()
    writer = PreparedWriter(
        folder / "tuab.h5", task="tuab", window_samples=256,
        channels=["C1", "Pz"], label_evidence={"status": "verified"},
        preprocessing=profile(),
    )
    for subject in range(10):
        writer.add(
            np.random.default_rng(subject).normal(size=(2, 512)),
            subject=str(subject), recording=f"record-{subject}",
            source={"sha256": fingerprint(subject)}, events=[(0, 0), (256, 1)],
            official_split="test" if subject >= 8 else "train",
        )
    writer.finish()
    return folder


def publish(tmp_path):
    source = official_fixture(tmp_path / "old")
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"name": "fixture", "datasets": {
        "tuab": {"rule": "official", "split_seed": 241}}}))
    release = tmp_path / "new"
    manifest = build_release(source, release, protocol)
    return source / "tuab.h5", release / "tuab.h5", manifest


def test_release_keeps_eeg_and_official_test_but_source_omits_test(tmp_path):
    old, new, manifest = publish(tmp_path)
    metadata = read_metadata(new)
    assert {s for s, role in metadata["subjects"].items() if role == "test"} == {"8", "9"}
    source = read_metadata(DatasetReference(new, 241, "tuab_source"))
    assert not source["counts"]["test"]
    assert set(source["subjects"]) == {str(i) for i in range(8)}
    assert set(source["subjects"].values()) == {"train", "validation"}
    with h5py.File(old) as a, h5py.File(new) as b:
        np.testing.assert_array_equal(a["X"][:], b["X"][:])
        np.testing.assert_array_equal(a["window_index"][:], b["window_index"][:])
        assert set(b["splits/subject_holdout"]) == {"241"}
    assert manifest["datasets"]["tuab"]["release_id"] == metadata["release_id"]
    with pytest.raises(FileExistsError):
        build_release(old.parent, new.parent, tmp_path / "protocol.json")


def test_official_test_cannot_be_added_to_source_even_with_updated_hashes(tmp_path):
    _, new, _ = publish(tmp_path)
    with h5py.File(new, "r+") as handle:
        metadata = json.loads(handle["metadata"].asstr()[()])
        group = handle["splits/tuab_source/241"]
        subjects = json.loads(group.attrs["subjects"])
        subjects["8"] = "train"
        rows = {r: group[f"{r}_idx"][:] for r in ("train", "validation", "test")}
        rows["train"] = np.sort(np.concatenate([rows["train"], [16, 17]])).astype("<u8")
        digest = bank_fingerprint(subjects, rows)
        group.attrs["subjects"] = json.dumps(subjects)
        group.attrs["fingerprint"] = digest
        del group["train_idx"]
        group.create_dataset("train_idx", data=rows["train"])
        metadata["split_banks"]["tuab_source/241"] = digest
        metadata["split_exclusions"]["tuab_source/241"].remove("8")
        metadata.pop("fingerprint")
        metadata["fingerprint"] = fingerprint(metadata)
        del handle["metadata"]
        handle.create_dataset("metadata", data=json.dumps(metadata))
    with pytest.raises(ValueError, match="Official test"):
        read_metadata(DatasetReference(new, 241, "tuab_source"))


def test_two_training_seeds_use_identical_indices_and_different_initial_weights(tmp_path):
    _, dataset, _ = publish(tmp_path)
    from pretrain.pipeline.checkpoint import set_seed
    from pretrain.models.factory import create_model
    from pretrain.pipeline.training import resolve_training_config

    memberships, weights = [], []
    for seed in (242, 243):
        config = resolve_training_config({"seed": seed, "split_seed": 241,
                                           "depth": 1, "width": 16, "heads": 8, "input_samples": 256})
        view = Windows(DatasetReference(dataset, config["split_seed"]), "train")
        memberships.append(view.rows)
        set_seed(seed)
        model = create_model(config)
        weights.append(next(model.parameters()).detach().clone())
    np.testing.assert_array_equal(*memberships)
    assert not np.array_equal(*weights)


def test_main_selects_requested_bank_before_initial_metadata_read(tmp_path, monkeypatch):
    _, dataset, _ = publish(tmp_path)
    settings = {
        "dataset": str(dataset), "output": str(tmp_path / "run"), "device": "cpu",
        "precision": "fp32", "purpose": "engineering", "seed": 242, "split_seed": 241,
        "depth": 1, "width": 16, "heads": 8, "epochs": 1, "workers": 0,
    }
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(settings))
    seen = []

    def training(config, selected, output, **kwargs):
        seen.append((config["seed"], selected.seed, read_metadata(selected)["subjects"]))
        return {"status": "complete"}

    monkeypatch.setattr("pretrain.pipeline.training.run", training)
    run_config(path)
    assert seen[0][:2] == (242, 241)
    assert seen[0][2] == read_metadata(dataset)["subjects"]


def test_main_trains_and_finetunes_two_seeds_on_one_official_split(tmp_path, monkeypatch):
    _, dataset, _ = publish(tmp_path)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    settings = {
        "dataset": str(dataset), "output": str(tmp_path / "pretrain"), "device": "cpu",
        "precision": "fp32", "purpose": "engineering", "seeds": [242, 243],
        "split_seed": 241, "depth": 1, "width": 16, "heads": 8, "objective": None,
        "epochs": 1, "warmup_epochs": 0, "workers": 0, "microbatch": 16,
        "logical_batch": 16,
    }
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(settings))
    run_config(path)
    settings.update(stage="finetune", output=str(tmp_path / "finetune"),
                    checkpoint=str(tmp_path / "pretrain/seed-{seed}/training/best.pth"))
    path.write_text(yaml.safe_dump(settings))
    run_config(path)
    memberships = []
    for stage in ("pretrain", "finetune"):
        for seed in (242, 243):
            root = tmp_path / stage / f"seed-{seed}"
            metadata = json.loads((root / "split.json").read_text())
            assert metadata["split_seed"] == 241
            memberships.append(metadata["subjects"])
            result = json.loads((root / "training/result.json").read_text())
            assert result["status"] == "complete"
            if stage == "finetune":
                assert result["selection"]["status"] == "locked_before_test"
    assert all(m == memberships[0] for m in memberships)
