import json

import pytest
import torch

from pretrain.pipeline.checkpoint import contract, save
from pretrain.pipeline.data import Windows
from pretrain.pipeline.exports import export, load
from pretrain.models.factory import create_model
from tests.test.pipeline.data import prepared_fixture


def test_release_encoder_round_trip_matches_selected_weights(tmp_path, monkeypatch):
    manifest = prepared_fixture(tmp_path / "data")
    config = {"size": "d8w64", "input_samples": 256, "seed": 241}
    model = create_model(config)
    metadata = Windows(manifest, "train").metadata
    checkpoint = tmp_path / "best.pth"
    save(
        checkpoint,
        {"model": model.state_dict(), "config": config, "contract": contract(config, metadata)},
    )
    export(checkpoint, manifest, tmp_path / "release")
    restored, _ = load(tmp_path / "release")
    for name, value in model.encoder.state_dict().items():
        if torch.is_tensor(value):
            torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
    assert (tmp_path / "release" / "checksums.json").exists()
    original = manifest.read_bytes()
    changed = json.loads(original)
    changed["preprocessing"]["clip"] = 100
    manifest.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="manifest differs"):
        export(checkpoint, manifest, tmp_path / "altered-data-release")
    assert not (tmp_path / "altered-data-release").exists()
    manifest.write_bytes(original)
    from pretrain.pipeline import exports

    sha256 = exports.sha256_file
    monkeypatch.setattr(
        exports, "sha256_file", lambda path: "changed" if path.name == "uv.lock" else sha256(path)
    )
    with pytest.raises(ValueError, match="environment lock"):
        export(checkpoint, manifest, tmp_path / "altered-lock-release")
