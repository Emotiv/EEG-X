import copy
import os
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest
import torch
import torch.nn.functional as F

from pretrain.models.dictionary import SoftDiCT, _ieee_dictionary_math
from pretrain.objectives import DEFAULT_HYDRA, validate_objective


def reference_counts(transform, eeg):
    batch, channels, samples = eeg.shape
    views = []
    for size in transform.kernel_sizes:
        for index, dilation in enumerate(transform.dilations):
            responses = F.conv1d(
                eeg.reshape(batch * channels, 1, samples),
                getattr(transform, f"filters_{size}")[index],
                dilation=dilation,
                padding=(size - 1) * dilation // 2,
            ).reshape(batch, channels, transform.groups, transform.k, samples)
            views.append((responses / transform.temperature).softmax(dim=3).sum(-1).flatten(2))
    return torch.cat(views, dim=-1).flatten(1)


def test_soft_counts_preserve_dictionary_competition_and_temporal_sums():
    transform = SoftDiCT(DEFAULT_HYDRA)
    samples = 64
    features = transform(torch.randn(2, 3, samples)).reshape(2, 3, 10, 8, 32)
    torch.testing.assert_close(features.sum(-1), torch.full((2, 3, 10, 8), float(samples)))
    assert not list(transform.parameters())
    assert torch.isfinite(features).all()


@pytest.mark.parametrize("flat", [False, True])
def test_dictionary_loss_backpropagates_through_reconstruction(flat):
    transform = SoftDiCT({**DEFAULT_HYDRA, "groups": 2, "k": 4})
    target = torch.randn(2, 2, 64)
    prediction = torch.zeros_like(target) if flat else torch.randn_like(target)
    prediction.requires_grad_()
    loss = F.mse_loss(transform(prediction), transform(target).detach())
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad.abs().sum() > 0


def test_dictionary_seed_does_not_consume_training_rng_and_buffers_round_trip():
    rng = torch.get_rng_state()
    left = SoftDiCT({**DEFAULT_HYDRA, "seed": 241})
    assert torch.equal(rng, torch.get_rng_state())
    right = SoftDiCT({**DEFAULT_HYDRA, "seed": 242})
    assert not torch.equal(left.filters_9, right.filters_9)
    right.load_state_dict(left.state_dict())
    eeg = torch.randn(1, 2, 64)
    torch.testing.assert_close(left(eeg), right(eeg), rtol=0, atol=0)


def test_dictionary_backward_does_not_retain_full_temporal_softmax_activations():
    transform = SoftDiCT(DEFAULT_HYDRA)
    eeg = torch.randn(2, 3, 128, requires_grad=True)
    saved = []

    def pack(value):
        saved.append(value.numel())
        return value

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
        transform(eeg).square().mean().backward()
    # Keep pooled features and small inputs rather than each dictionary view's
    # complete [batch, channels, groups, kernels, time] probability tensor.
    assert max(saved) <= eeg.numel() * 32
    assert torch.isfinite(eeg.grad).all()


@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_bounded_dictionary_matches_reference_features_and_input_gradients(temperature):
    objective = {**DEFAULT_HYDRA, "groups": 2, "k": 4, "temperature": temperature}
    transform = SoftDiCT(objective, max_response_elements=2048)
    actual_input = torch.randn(3, 2, 64, requires_grad=True)
    reference_input = actual_input.detach().clone().requires_grad_()
    reference = reference_counts(transform, reference_input)
    actual = transform(actual_input)
    torch.testing.assert_close(actual, reference, rtol=2e-6, atol=2e-5)
    actual.square().mean().backward()
    reference.square().mean().backward()
    torch.testing.assert_close(actual_input.grad, reference_input.grad, rtol=2e-5, atol=2e-5)


def test_dictionary_keeps_fp32_gradients_inside_enclosing_bf16_autocast():
    torch.manual_seed(123)
    transform = SoftDiCT({**DEFAULT_HYDRA, "groups": 2, "k": 4, "temperature": 0.7})
    actual_input = torch.randn(3, 2, 64, requires_grad=True)
    reference_input = actual_input.detach().clone().requires_grad_()
    reference = reference_counts(transform, reference_input)
    reference.square().mean().backward()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = transform(actual_input)
        actual.square().mean().backward()
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, reference, rtol=2e-6, atol=2e-5)
    torch.testing.assert_close(actual_input.grad, reference_input.grad, rtol=2e-5, atol=2e-5)


def test_dictionary_precision_scopes_serialize_and_restore_the_original_policy():
    previous = torch.backends.cuda.matmul.fp32_precision
    started, entered = threading.Event(), threading.Event()

    def concurrent_scope():
        started.set()
        with _ieee_dictionary_math(torch.device("cuda")):
            entered.set()
            assert torch.backends.cuda.matmul.fp32_precision == "ieee"

    try:
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        with ThreadPoolExecutor(1) as pool:
            with _ieee_dictionary_math(torch.device("cuda")):
                future = pool.submit(concurrent_scope)
                assert started.wait(5)
                assert not entered.wait(0.05)
                assert torch.backends.cuda.matmul.fp32_precision == "ieee"
            future.result(timeout=5)
        assert entered.is_set()
        assert torch.backends.cuda.matmul.fp32_precision == "tf32"
    finally:
        torch.backends.cuda.matmul.fp32_precision = previous


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_dictionary_rejects_invalid_response_limits(limit):
    with pytest.raises(ValueError, match="response limit"):
        SoftDiCT(DEFAULT_HYDRA, max_response_elements=limit)


def test_fused_dictionary_is_explicit_and_requires_cuda():
    transform = SoftDiCT(DEFAULT_HYDRA, backend="triton")
    with pytest.raises(ValueError, match="requires CUDA"):
        transform(torch.randn(2, 3, 64))


@pytest.mark.parametrize("backend", ["unknown", [], True])
def test_dictionary_backend_rejects_invalid_values(backend):
    with pytest.raises(ValueError, match="backend"):
        SoftDiCT(DEFAULT_HYDRA, backend=backend)


@pytest.mark.parametrize("groups,k", [(3, 4), (2, 3), (16, 32)])
def test_fused_dictionary_rejects_unsupported_group_layouts(groups, k):
    with pytest.raises(ValueError, match="power-of-two"):
        SoftDiCT({**DEFAULT_HYDRA, "groups": groups, "k": k}, backend="triton")


@pytest.mark.skipif(
    os.environ.get("EEGX_RUN_CUDA_DICTIONARY") != "1",
    reason="Explicit bounded GPU dictionary check required",
)
def test_fused_gpu_dictionary_matches_reference_values_and_gradients():
    from pretrain.compute import configure_compute
    from pretrain.pipeline.budget import Budget
    from scripts.profile_training import profiling_budget

    path = os.environ.get("EEGX_BUDGET_PATH")
    if not path:
        pytest.fail("Provide the existing authoritative budget for GPU checks")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    with (
        profiling_budget(Budget, path, reserve_live=True, max_seconds=180, concurrency=4),
        configure_compute({"precision": "bf16", "deterministic": True}, "cuda:0"),
    ):
        torch.cuda.set_per_process_memory_fraction(
            8 * 2**30 / torch.cuda.get_device_properties(0).total_memory
        )
        for samples, channels, groups, k, temperature, flat in [
            (63, 2, 2, 4, 0.7, False),
            (64, 2, 2, 4, 0.7, False),
            (64, 2, 2, 4, 1.0, True),
            (256, 14, 8, 32, 1.0, False),
            (1280, 19, 8, 32, 1.0, False),
        ]:
            objective = {**DEFAULT_HYDRA, "groups": groups, "k": k, "temperature": temperature}
            transform = SoftDiCT(objective, backend="triton").cuda()
            values = (
                torch.zeros(2, channels, samples) if flat else torch.randn(2, channels, samples)
            )
            actual_input = values.cuda().requires_grad_()
            reference_input = actual_input.detach().clone().requires_grad_()
            reference = reference_counts(transform, reference_input)
            reference.square().mean().backward()
            rng_cpu, rng_gpu = torch.get_rng_state(), torch.cuda.get_rng_state()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                actual = transform(actual_input)
                actual.square().mean().backward()
            torch.testing.assert_close(actual, reference, rtol=2e-5, atol=2e-4)
            torch.testing.assert_close(
                actual_input.grad, reference_input.grad, rtol=1e-4, atol=2e-5
            )
            assert torch.equal(torch.get_rng_state(), rng_cpu)
            assert torch.equal(torch.cuda.get_rng_state(), rng_gpu)


@pytest.mark.parametrize(
    "key,value", [("kernel_sizes", [8]), ("dilations", [0]), ("dilations", [1, 1]), ("k", 1)]
)
def test_dictionary_rejects_invalid_support(key, value):
    objective = copy.deepcopy(DEFAULT_HYDRA)
    objective[key] = value
    with pytest.raises(ValueError):
        validate_objective(objective)
