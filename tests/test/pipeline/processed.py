import json

import numpy as np
import pytest

from pretrain.channel_identity import sha256_file
from pretrain.pipeline.data import Windows
from pretrain.pipeline.preprocessing import profile
from pretrain.pipeline.processed import prepare


def test_processed_release_keeps_values_and_freezes_source_evidence(tmp_path):
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
    prepared = tmp_path / "prepared"
    metadata = prepare(path, prepared)
    np.testing.assert_array_equal(np.load(prepared / "record-000000.npy"), values[0])
    assert metadata["non_rasr_evidence"]["sha256"] == sha256_file(evidence)
    assert metadata["release_counts"]["retained_windows"] == 40
    assert (
        sum(
            len(Windows(prepared / "manifest.json", role))
            for role in ("train", "validation", "test")
        )
        == 40
    )
    contract["preprocessing"]["clip"] = 999
    path.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="preprocessing is incompatible"):
        prepare(path, tmp_path / "invalid")
