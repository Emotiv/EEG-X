import pytest
import torch

from pretrain.objectives import DEFAULT_HYDRA
from pretrain.models.factory import Classifier, assert_all_rms, create_model, masks


@pytest.mark.parametrize("normalization", ["none", "rmsnorm"])
def test_attention_backend_preserves_the_selected_equation(normalization):
    import copy

    from pretrain.models.attention import MultiHeadAttention

    manual = MultiHeadAttention(2, 16, dropout=0).eval()
    manual.qk_normalization = normalization
    optimized = copy.deepcopy(manual)
    optimized.use_sdpa = True
    inputs = [torch.randn(2, length, 16, requires_grad=True) for length in (3, 5, 5)]
    copies = [value.detach().clone().requires_grad_() for value in inputs]
    left, right = manual(*inputs), optimized(*copies)
    torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)
    left.square().sum().backward()
    right.square().sum().backward()
    for before, after in zip(inputs, copies):
        torch.testing.assert_close(before.grad, after.grad, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("samples", [256, 512, 1280])
@pytest.mark.parametrize("hydra", [False, True])
def test_task_window_forward_backward_and_rms_readout(samples, hydra):
    config = {"size": "d8w64", "input_samples": samples, "objective": None}
    if hydra:
        config["objective"] = {**DEFAULT_HYDRA, "groups": 2, "k": 4}
    model = create_model(config)
    eeg, ids = torch.randn(1, 2, samples), torch.tensor([0, 1])
    mask = masks(["window-a"], ids, samples=samples, seed=4)
    result = model(eeg, ids, mask=mask)
    loss = result["loss"] if isinstance(result, dict) else result[0]
    loss.backward()
    assert torch.isfinite(model.reconstruction_head.weight.grad).all()
    classifier = Classifier(model.encoder, 2)
    assert classifier(eeg, ids).shape == (1, 2)
    assert_all_rms(classifier)


def test_masks_are_batch_and_channel_order_independent():
    ids = torch.tensor([2, 0, 1])
    full = masks(["a", "b"], ids, samples=256, seed=7)
    assert torch.equal(full[:1], masks(["a"], ids, samples=256, seed=7))
    permutation = torch.tensor([2, 0, 1])
    reordered = masks(["b", "a"], ids[permutation], samples=256, seed=7)
    assert torch.equal(reordered.reshape(2, 8, 3), full.flip(0).reshape(2, 8, 3)[:, :, permutation])


def test_all_rms_rejects_layernorm():
    with pytest.raises(ValueError, match="LayerNorm"):
        assert_all_rms(torch.nn.Sequential(torch.nn.LayerNorm(8)))


def test_hybrid_tokens_cannot_read_hidden_patch_samples():
    model = create_model({"size": "d8w64", "input_samples": 256}).eval()
    eeg, ids = torch.randn(2, 2, 256), torch.tensor([0, 1])
    mask = masks(["a", "b"], ids, samples=256, seed=241)
    changed = model.encoder.patchify(eeg).clone()
    changed[mask.bool()] += 100
    changed = changed.reshape(2, 8, 2, 32).permute(0, 2, 1, 3).reshape(2, 2, 256)
    with torch.no_grad():
        left = model.encoder.embed_patches(eeg, ids)
        right = model.encoder.embed_patches(changed, ids)
    torch.testing.assert_close(left[mask == 0], right[mask == 0], rtol=0, atol=0)
    assert not torch.equal(left[mask == 1], right[mask == 1])


def test_predictor_remains_in_the_active_reconstruction_path():
    model = create_model({"size": "d8w64", "input_samples": 256})
    eeg, ids = torch.randn(2, 2, 256), torch.tensor([0, 1])
    model(eeg, ids, mask=masks(["a", "b"], ids, samples=256, seed=241))["loss"].backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.Predictor.parameters()
    )
    assert sum(float(p.grad.abs().sum()) for p in model.Predictor.parameters()) > 0
    assert model.encoder.raw_projection.weight.grad.abs().sum() > 0
    assert model.encoder.spectral_projection.weight.grad.abs().sum() > 0
    assert len(model.Decoder.TRMs) == 4


def test_frozen_encoder_supports_a_different_task_duration():
    model = create_model({"size": "d8w64", "input_samples": 1280}).eval()
    classifier = Classifier(model.encoder, 4).eval()
    with torch.no_grad():
        assert classifier(torch.randn(1, 3, 512), torch.tensor([0, 1, 2])).shape == (1, 4)


def test_eegx2_checkpoint_geometry_is_rejected():
    with pytest.raises(ValueError, match="EEGX2 checkpoints are incompatible"):
        create_model({"size": "d11", "input_samples": 256})


def test_predictions_follow_physical_channel_order():
    model = create_model({"size": "d8w64", "input_samples": 256}).eval()
    eeg, ids = torch.randn(1, 3, 256), torch.tensor([0, 1, 2])
    permutation = torch.tensor([2, 0, 1])
    mask = masks(["window"], ids, samples=256, seed=241)
    reordered_mask = mask.reshape(1, 8, 3)[:, :, permutation].flatten(1)
    with torch.no_grad():
        original = model(eeg, ids, mask=mask)
        reordered = model(eeg[:, permutation], ids[permutation], mask=reordered_mask)
    expected = original["pred"].reshape(1, 8, 3, 32)[:, :, permutation]
    torch.testing.assert_close(
        reordered["pred"].reshape(1, 8, 3, 32), expected, rtol=2e-5, atol=2e-5
    )
    torch.testing.assert_close(original["loss"], reordered["loss"], rtol=2e-5, atol=2e-5)


def test_checkpoint_rejects_changed_physical_position_buffers():
    model = create_model({"size": "d8w64", "input_samples": 256})
    state = model.state_dict()
    key = "encoder.channel_embedding.weight"
    state[key] = state[key].clone()
    state[key][0, 0] += 1
    with pytest.raises(ValueError, match="fixed buffer"):
        model.load_state_dict(state)


def test_primary_and_soft_dictionary_are_defaults():
    model = create_model({"input_samples": 256, "seed": 241})
    assert model.contract["width"] == 64 and model.contract["depth"] == 8
    assert model.contract["objective"] == {**DEFAULT_HYDRA, "seed": 241}
    assert model.hydra is not None
    assert model.contract["teacher"] == "none"
    assert not any(
        "teacher" in name or "target_encoder" in name for name, _ in model.named_modules()
    )


def test_one_explicit_model_configuration_controls_the_encoder():
    model = create_model({"depth": 2, "width": 32, "input_samples": 256, "objective": None})
    assert model.contract["depth"] == 2 and model.contract["width"] == 32
    assert len(model.encoder.contex_encoder.TRMs) == 2
    eeg, ids = torch.randn(1, 2, 256), torch.tensor([0, 1])
    assert torch.isfinite(model(eeg, ids, mask=masks(["a"], ids, samples=256, seed=241))["loss"])
    with pytest.raises(ValueError, match="geometry differs"):
        create_model({"size": "d8w64", "depth": 2, "input_samples": 256})
    with pytest.raises(ValueError, match="width divisible"):
        create_model({"width": 65, "input_samples": 256})


def test_old_model_and_objective_versions_are_rejected():
    model = create_model({"input_samples": 256})
    state = model.state_dict()
    state["_extra_state"]["name"] = "eegx_hybrid_reconstruction_v1"
    with pytest.raises(ValueError, match="checkpoint contract"):
        model.load_state_dict(state)
    with pytest.raises(ValueError, match="Unknown objective"):
        create_model(
            {"input_samples": 256, "objective": {**DEFAULT_HYDRA, "name": "raw_soft_hydra_v1"}}
        )
