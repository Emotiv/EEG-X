"""Original row lineage is self-contained and checked before HDF5 publication."""

import json

import numpy as np
import pytest

from pretrain.metadata import fingerprint
from pretrain.pipeline.dataset import PreparedWriter, read_metadata
from pretrain.pipeline.dataset.provenance import load_provenance
from pretrain.pipeline.prepare import prepare_adapter
from pretrain.pipeline.preprocessing import profile


def publish(path, *, mutation=None, include=True):
    writer = PreparedWriter(
        path,
        task="fixture",
        window_samples=256,
        channels=["C1", "Pz"],
        label_evidence={"status": "verified", "kind": "synthetic_fixture"},
        preprocessing=profile(),
        sampling_policy={"requires_row_provenance": True, "reference_native_samples": 256},
    )
    sources, mappings = [], []
    for subject in range(10):
        source = {
            "path": f"raw-{subject}.edf",
            "sha256": fingerprint(subject),
            "native_samples": 512,
        }
        sources.append({"file": source["path"], "sha256": source["sha256"]})
        writer.add(
            np.random.default_rng(subject).normal(size=(2, 512)),
            subject=str(subject),
            recording=f"record-{subject}",
            source=source,
            events=[(0, 0), (256, 1)],
        )
        for label in (0, 1):
            start = label * 256
            mappings.append(
                {
                    "recording": f"record-{subject}",
                    "start": start,
                    "native_crop": [start, start + 256],
                    "label": label,
                    "original_rows": [{"field": "X_train", "row": subject * 2 + label}],
                }
            )
    proof = {
        "schema_version": 1,
        "task": "fixture",
        "status": "numerically_verified",
        "signal_multiset_matches": True,
        "artifact": "synthetic-reference.npy",
        "artifact_sha256": "a" * 64,
        "original_partitions": {"X_train": 20, "X_val": 0, "X_test": 0},
        "sources": sources,
        "mappings": mappings,
    }
    if mutation:
        mutation(proof)
    reference = path.with_suffix(".json")
    reference.write_text(json.dumps(proof))

    def finalize(metadata):
        if include:
            metadata["reference_provenance"] = load_provenance(reference)
        return metadata

    writer.finish(finalize=finalize)
    reference.unlink()
    return path


def test_original_array_rows_are_recoverable_without_the_proof_sidecar(tmp_path):
    path = publish(tmp_path / "lineage.h5")
    assert not path.with_suffix(".json").exists()
    proof = read_metadata(path)["reference_provenance"]
    assert proof["artifact_sha256"] == "a" * 64
    assert {old["row"] for item in proof["mappings"] for old in item["original_rows"]} == set(
        range(20)
    )


@pytest.mark.parametrize("part", ["source", "crop", "stop", "row", "row_offset"])
def test_invalid_original_row_lineage_cannot_be_published(tmp_path, part):
    def mutate(proof):
        if part == "source":
            proof["sources"][0]["sha256"] = "f" * 64
        elif part == "crop":
            proof["mappings"][0]["native_crop"][0] = 1
        elif part == "stop":
            proof["mappings"][0]["native_crop"][1] += 1_000_000_000
        elif part == "row_offset":
            for mapping in proof["mappings"]:
                for old in mapping["original_rows"]:
                    old["row"] += 1_000_000
        else:
            proof["mappings"][-1]["original_rows"] = proof["mappings"][0]["original_rows"]

    path = tmp_path / "invalid.h5"
    with pytest.raises(ValueError, match="provenance"):
        publish(path, mutation=mutate)
    assert not path.exists()
    assert not list(tmp_path.glob("*.working"))
    assert not list(tmp_path.glob("*.partial"))


def test_established_lineage_is_required_before_publication(tmp_path):
    path = tmp_path / "missing.h5"
    with pytest.raises(ValueError, match="provenance is required"):
        publish(path, include=False)
    assert not path.exists()


def test_native_command_requires_established_proof_before_loading_signals(tmp_path):
    class Source:
        task, window_samples, channels = "bciciv_2a", 256, ["C1", "Pz"]
        label_evidence = {"status": "verified"}

        def recordings(self):
            pytest.fail("Missing lineage must fail before signal work")

    with pytest.raises(ValueError, match="reference-proof"):
        prepare_adapter(Source(), tmp_path / "missing.h5")
