import numpy as np
import torch

from pretrain.pipeline.budget import NoGpuBudget
from pretrain.pipeline.data import Windows
from pretrain.pipeline.examples import save
from pretrain.models.factory import create_model
from tests.test.pipeline.data import prepared_fixture


def test_curve_examples_keep_fixed_validation_membership(tmp_path):
    manifest = prepared_fixture(tmp_path / "data")
    view = Windows(manifest, "validation")
    config = {"size": "d8w64", "input_samples": 256, "precision": "fp32", "seed": 241}
    model = create_model(config)
    for name in ("first", "second"):
        save(model, view, config, torch.device("cpu"), NoGpuBudget(), tmp_path / f"{name}.npz")
    left, right = np.load(tmp_path / "first.npz"), np.load(tmp_path / "second.npz")
    for key in left.files:
        np.testing.assert_array_equal(left[key], right[key])
    assert left["dataset"].item() == view.metadata["fingerprint"]
    assert left["mask"].mean() == 0.75
