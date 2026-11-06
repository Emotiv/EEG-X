"""Behavior of the optional detached HYDRA patch-target objective."""

import copy

import pytest
import torch

from pretrain.models.factory import create_model, masks
from pretrain.objectives import DEFAULT_HYDRA_TARGET, validate_objective


def candidate(**changes):
    return create_model({
        "width": 32, "depth": 1, "input_samples": 128, "dropout": 0.0,
        "seed": 241, "objective": {**DEFAULT_HYDRA_TARGET, **changes},
    })


def test_auxiliary_loss_trains_decoder_and_encoder_without_dictionary_backward():
    model = candidate()
    eeg = torch.randn(2, 3, 128)
    ids = torch.tensor([0, 1, 2])
    mask = masks(["a", "b"], ids, samples=128, seed=241)
    calls = []
    model.hydra_target.transform.register_forward_pre_hook(
        lambda *_: calls.append(torch.is_grad_enabled())
    )
    result = model(eeg, ids, mask=mask)
    auxiliary = (result["loss"] - result["mae_loss"]) / model.hydra_weight
    # Remove the raw-MSE path so its gradients cannot hide a disconnected head.
    raw = (model.patch_errors(eeg, result["pred"]) * mask).sum() / mask.sum()
    (result["loss"] - raw).backward()
    for module in (model.hydra_target.head, model.encoder.raw_projection, model.Predictor,
                   model.Decoder):
        assert sum(float(p.grad.abs().sum()) for p in module.parameters()
                   if p.grad is not None) > 0
    assert model.reconstruction_head.weight.grad.abs().sum() == 0
    assert calls and not any(calls)
    assert torch.isfinite(auxiliary)
    assert not list(model.hydra_target.transform.parameters())
    assert model.hydra is None


def test_targets_are_detached_channel_local_and_match_reference_transform():
    model = candidate()
    eeg = torch.randn(2, 3, 128, requires_grad=True)
    target = model.hydra_target.targets(eeg)
    assert not target.requires_grad
    assert target.shape == (2, 2, 3, model.hydra_target.feature_dim)
    for region in range(2):
        for channel in range(3):
            expected = model.hydra_target.transform(
                eeg[:, channel:channel + 1, region * 64:(region + 1) * 64].detach()
            )
            torch.testing.assert_close(target[:, region, channel], expected)
    permutation = torch.tensor([2, 0, 1])
    torch.testing.assert_close(model.hydra_target.targets(eeg[:, permutation]),
                               target[:, :, permutation])


def test_response_bounded_target_batches_preserve_features_and_rng():
    from pretrain.models.targets import HydraPatchTarget

    large = HydraPatchTarget(32, DEFAULT_HYDRA_TARGET)
    small = HydraPatchTarget(32, DEFAULT_HYDRA_TARGET, max_response_elements=1024)
    small.load_state_dict(large.state_dict())
    eeg = torch.randn(3, 3, 128)
    rng = torch.get_rng_state()
    torch.testing.assert_close(large.targets(eeg), small.targets(eeg), rtol=0, atol=0)
    assert torch.equal(rng, torch.get_rng_state())


def test_auxiliary_mask_excludes_fully_visible_regions():
    model = candidate()
    objective = model.hydra_target
    eeg = torch.randn(1, 2, 128)
    decoded = torch.randn(1, 8, 32, requires_grad=True)
    mask = torch.tensor([[[0, 1], [0, 1], [1, 1], [1, 1]]]).flatten(1)
    objective(eeg, decoded, mask).backward()
    gradient = decoded.grad.reshape(1, 4, 2, 32)
    assert gradient[:, :2, 0].abs().sum() == 0
    assert gradient[:, :2, 1].abs().sum() > 0
    assert gradient[:, 2:].abs().sum() > 0


def test_candidate_checkpoint_rejects_a_different_target_recipe():
    model = candidate()
    state = copy.deepcopy(model.state_dict())
    candidate().load_state_dict(state)
    with pytest.raises(ValueError, match="checkpoint contract"):
        candidate(k=8).load_state_dict(state)


def test_target_training_saves_a_usable_checkpoint_and_logs_the_actual_objective(tmp_path):
    from types import SimpleNamespace

    from pretrain.pipeline.training import run
    from tests.test.pipeline.data import prepared_fixture

    rows = []
    tracker = SimpleNamespace(
        log=lambda row: rows.append(row),
        parameters=lambda *args, **kwargs: None,
        summary=lambda *args, **kwargs: None,
        checkpoint=lambda *args, **kwargs: None,
    )
    config = {
        "width": 32, "depth": 1, "input_samples": 256, "seed": 241,
        "epochs": 1, "warmup_epochs": 0, "precision": "fp32", "workers": 0,
        "logical_batch": 12, "microbatch": 2, "purpose": "pipeline_test",
        "objective": DEFAULT_HYDRA_TARGET,
    }
    checkpoint = run(config, prepared_fixture(tmp_path / "data"), tmp_path / "run",
                     tracker=tracker)
    payload = torch.load(checkpoint, weights_only=True)
    create_model(payload["config"]).load_state_dict(payload["model"])
    assert any("train/hydra_target" in row for row in rows)
    assert not any("train/soft_dict" in row for row in rows)


@pytest.mark.parametrize("key,value", [("patch_size", 33), ("patch_size", 16),
    ("kernel_size", 8), ("kernel_size", 65), ("groups", 3), ("k", 0)])
def test_target_recipe_rejects_invalid_geometry(key, value):
    with pytest.raises(ValueError):
        validate_objective({**DEFAULT_HYDRA_TARGET, key: value})
