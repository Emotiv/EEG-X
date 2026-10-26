import numpy as np
import torch

from pretrain.pipeline import readout
from pretrain.pipeline.budget import NoGpuBudget


def test_probe_stops_on_validation_and_restores_rng(tmp_path, monkeypatch):
    x = np.array([[10, 0], [12, 1], [14, 2], [16, 3]], dtype=np.float32)
    y = np.array([0, 1, 0, 1])
    validation_scores = iter(
        [
            {"loss": 0.8, "balanced_accuracy": 0.6},
            {"loss": 0.7, "balanced_accuracy": 0.8},
            {"loss": 0.6, "balanced_accuracy": 0.7},
            {"loss": 0.5, "balanced_accuracy": 0.7},
            {"loss": 0.4, "balanced_accuracy": 0.7},
        ]
    )
    monkeypatch.setattr(readout, "classification", lambda *_: next(validation_scores))
    before = torch.random.get_rng_state().clone()
    probe = readout.fit_readout(
        x,
        y,
        x + 100,
        y,
        {"seed": 241},
        tmp_path,
        NoGpuBudget(),
        settings={"epochs": 20, "patience": 3},
    )
    assert torch.equal(before, torch.random.get_rng_state())
    assert len(probe.history) == 5
    assert probe.selected_epoch == 2
    assert probe.validation == {"loss": 0.7, "balanced_accuracy": 0.8}
    np.testing.assert_allclose(probe.scaler.mean_, x.mean(axis=0))
    saved = torch.load(tmp_path / "best.pth", weights_only=True)
    assert saved["epoch"] == 2
    for name, value in probe.model.state_dict().items():
        torch.testing.assert_close(value, saved["model"][name])
