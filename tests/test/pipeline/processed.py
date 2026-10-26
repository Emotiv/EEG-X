import json
import h5py

import numpy as np
import pytest

from pretrain.channel_identity import sha256_file
from pretrain.pipeline.data import Windows
from pretrain.pipeline.preprocessing import profile
from pretrain.pipeline.processed import prepare


def release_contract(tmp_path):
    values = np.random.default_rng(4).normal(size=(40, 2, 256)).astype("float32")
    artifact = tmp_path / "release.npy"
    np.save(
        artifact, {"X": values, "Y": np.tile([0, 1, 0, 1], 10), "ID": np.repeat(np.arange(10), 4)}
    )
    evidence = tmp_path / "history.json"
    evidence.write_text(
        json.dumps(
            {
                "status": "verified",
                "authority": "synthetic fixture producer",
                "upstream_rasr": False,
                "source_sha256": [sha256_file(artifact)],
            }
        )
    )
    verified = {"status": "verified", "authority": "synthetic fixture producer"}
    contract = {
        "task": "dreamer_arousal",
        "artifact": artifact.name,
        "artifact_sha256": sha256_file(artifact),
        "layout": "NCT",
        "channels": ["C1", "Pz"],
        "label_map": {"0": 0, "1": 1},
        "partitions": [{"signals": "X", "labels": "Y", "subjects": "ID"}],
        "label_evidence": verified,
        "subject_evidence": verified,
        "preprocessing_evidence": verified,
        "preprocessing": profile(),
        "source_evidence": evidence.name,
    }
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract))
    return path, values, evidence, contract


def test_processed_release_keeps_values_and_freezes_source_evidence(tmp_path):
    path, values, evidence, contract = release_contract(tmp_path)
    prepared = tmp_path / "prepared.h5"
    metadata = prepare(path, prepared)
    with h5py.File(prepared, "r") as handle:
        np.testing.assert_array_equal(handle["X"][0], values[0])
    assert metadata["non_rasr_evidence"]["sha256"] == sha256_file(evidence)
    assert metadata["release_counts"]["retained_windows"] == 40
    assert sum(len(Windows(prepared, role)) for role in ("train", "validation", "test")) == 40
    contract["preprocessing"]["clip"] = 999
    path.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="preprocessing is incompatible"):
        prepare(path, tmp_path / "invalid.h5")


def test_preparation_publishes_only_the_complete_manifest(tmp_path, monkeypatch):
    from pretrain.pipeline import processed
    from pretrain.pipeline.splits import validate_prepared_input

    path, _, _, _ = release_contract(tmp_path)
    prepared = tmp_path / "prepared.h5"
    attach = processed.attach_source_evidence
    observed = []

    def observe_during_preparation(metadata, evidence):
        observed.append((prepared).exists())
        return attach(metadata, evidence)

    monkeypatch.setattr(processed, "attach_source_evidence", observe_during_preparation)
    prepare(path, prepared)
    assert observed == [False]
    validate_prepared_input(Windows(prepared, "train").metadata)


def test_processed_release_can_preserve_values_without_upstream_history(tmp_path):
    from pretrain.pipeline.preprocessing import processed_profile

    path, values, _, contract = release_contract(tmp_path)
    artifact = tmp_path / contract["artifact"]
    values = (values * 100).astype("float64")
    np.save(
        artifact, {"X": values, "Y": np.tile([0, 1, 0, 1], 10), "ID": np.repeat(np.arange(10), 4)}
    )
    contract["artifact_sha256"] = sha256_file(artifact)
    contract["preprocessing"] = processed_profile()
    contract.pop("source_evidence")
    contract.pop("preprocessing_evidence")
    path.write_text(json.dumps(contract))
    output = tmp_path / "as-provided.h5"
    metadata = prepare(path, output)
    with h5py.File(output, "r") as handle:
        np.testing.assert_array_equal(handle["X"][0], values[0].astype("float32"))
    assert metadata["preprocessing"]["upstream_history"] == "not_verified"
    assert "non_rasr_evidence" not in metadata
