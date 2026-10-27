import pytest
import torch

from pretrain.normalization import RMSNorm, parameter_free_rms_norm


@pytest.mark.parametrize("weighted", [False, True])
def test_triton_backend_uses_identical_native_cpu_fallback(weighted):
    generator = torch.Generator().manual_seed(241)
    source = torch.randn(2, 7, 8, 64, generator=generator).transpose(1, 2)
    left, right = source.detach().requires_grad_(), source.detach().clone().requires_grad_()
    if weighted:
        native, selected = RMSNorm(64), RMSNorm(64, backend="triton")
        selected.load_state_dict(native.state_dict())
        expected, actual = native(left), selected(right)
    else:
        expected = parameter_free_rms_norm(left)
        actual = parameter_free_rms_norm(right, backend="triton")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    gradient = torch.randn(actual.shape, generator=generator)
    expected.backward(gradient)
    actual.backward(gradient)
    torch.testing.assert_close(right.grad, left.grad, rtol=0, atol=0)
    if weighted:
        torch.testing.assert_close(selected.weight.grad, native.weight.grad, rtol=0, atol=0)


def test_invalid_rms_backend_is_rejected():
    with pytest.raises(ValueError, match="backend"):
        RMSNorm(64, backend={})
    with pytest.raises(ValueError, match="backend"):
        parameter_free_rms_norm(torch.ones(1, 64), backend="other")


@pytest.mark.parametrize("rows, width, weighted", [
    (1 << 25, 64, False), ((1 << 19) + 1, 64, True),
    (4, 1, False), (4, 4, True), (4, 7, False), (4, 9, True),
])
def test_unsupported_cuda_shapes_use_native_fallback_without_allocation(monkeypatch, rows, width, weighted):
    import pretrain.normalization as normalization

    class MetadataOnlyTensor:
        is_cuda = True
        dtype = torch.float32
        device = torch.device("cuda:0")

        def __init__(self, shape):
            self.shape = shape
            self.ndim = len(shape)

        def numel(self):
            return self.shape[0] * self.shape[1]

        def stride(self, axis):
            return 1 if axis in (-1, 0) else self.shape[-1]

    inputs = MetadataOnlyTensor((rows, width))
    weight = MetadataOnlyTensor((width,)) if weighted else None
    marker = object()
    monkeypatch.setattr(normalization, "float32_rms_norm", lambda *_args, **_kwargs: marker)
    assert normalization._apply_rms_norm(inputs, weight=weight, eps=1e-6, backend="triton") is marker


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("shape", [(2, 7, 192), (2, 3, 7, 64), (2, 7, 512)])
@pytest.mark.parametrize("scale", [1.0, 1e-4])
def test_rms_outputs_and_gradients_preserve_fp32_math(weighted, shape, scale):
    generator = torch.Generator().manual_seed(241)
    initial = torch.randn(shape, generator=generator) * scale
    # Include the noncontiguous layout used by attention Q/K normalization.
    if len(shape) == 4:
        initial = initial.transpose(1, 2)
    inputs = initial.detach().requires_grad_()
    reference = initial.detach().clone().requires_grad_()
    epsilon = 1e-6
    expected = reference * torch.rsqrt(reference.square().mean(-1, keepdim=True) + epsilon)
    if weighted:
        module = RMSNorm(shape[-1], eps=epsilon)
        weight = torch.randn(shape[-1], generator=generator, requires_grad=True)
        with torch.no_grad():
            module.weight.copy_(weight)
        expected = expected * weight
        actual = module(inputs)
    else:
        actual = parameter_free_rms_norm(inputs, eps=epsilon)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    gradient = torch.randn(actual.shape, generator=generator)
    actual.backward(gradient)
    expected.backward(gradient)
    torch.testing.assert_close(
        inputs.grad, reference.grad, rtol=2e-5, atol=2e-3 if scale < 1 else 2e-6
    )
    if weighted:
        torch.testing.assert_close(module.weight.grad, weight.grad, rtol=2e-5, atol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("weighted", [False, True])
def test_cuda_rms_keeps_fp32_math_under_bf16_autocast(weighted):
    torch.manual_seed(241)
    inputs = torch.randn(4, 8, 31, 64, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    inputs = inputs.detach().requires_grad_()
    reference = inputs.detach().clone().requires_grad_()
    values = reference.float()
    expected = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)
    module = RMSNorm(64).cuda() if weighted else None
    if weighted:
        weight = module.weight.detach().clone().requires_grad_()
        expected = expected * weight
    expected = expected.to(torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual = module(inputs) if weighted else parameter_free_rms_norm(inputs)
    assert actual.dtype == inputs.dtype
    torch.testing.assert_close(actual, expected, rtol=0.008, atol=0.008)
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    torch.testing.assert_close(inputs.grad, reference.grad, rtol=0.016, atol=0.016)
    if weighted:
        assert module.weight.grad.dtype == torch.float32
        torch.testing.assert_close(module.weight.grad, weight.grad, rtol=2e-5, atol=2e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("inference", [False, True])
def test_cuda_fused_norm_supports_inference_contexts(inference):
    inputs = torch.randn(2, 7, 64, device="cuda", dtype=torch.bfloat16)
    native = RMSNorm(64).cuda()
    selected = RMSNorm(64, backend="triton").cuda()
    selected.load_state_dict(native.state_dict())
    context = torch.inference_mode() if inference else torch.no_grad()
    with context, torch.autocast("cuda", dtype=torch.bfloat16):
        expected, actual = native(inputs), selected(inputs)
    assert not actual.requires_grad
    torch.testing.assert_close(actual, expected, rtol=0.008, atol=0.008)
