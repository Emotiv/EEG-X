"""Explicit CUDA math policy using the PyTorch precision API."""

from contextlib import contextmanager

import torch


DEFAULT_COMPUTE = {
    "matmul": "tf32",
    "convolution": "ieee",
    "cudnn_benchmark": False,
}


def autocast_context(device, precision):
    if precision not in {"fp32", "bf16"}:
        raise ValueError("Precision must be fp32 or bf16")
    device = torch.device(device)
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda" and precision == "bf16",
    )


def compute_policy(config):
    policy = config.get("compute", DEFAULT_COMPUTE)
    if not isinstance(policy, dict) or set(policy) != set(DEFAULT_COMPUTE):
        raise ValueError("compute requires matmul, convolution and cudnn_benchmark")
    if policy["matmul"] not in {"tf32", "ieee"} or policy["convolution"] not in {"tf32", "ieee"}:
        raise ValueError("CUDA math precision must be tf32 or ieee")
    if type(policy["cudnn_benchmark"]) is not bool:
        raise ValueError("cudnn_benchmark must be boolean")
    return dict(policy)


@contextmanager
def _configure_compute(config, device):
    """Keep math settings scoped to the run and record effective CPU fallback."""
    policy = compute_policy(config)
    device = torch.device(device)
    if device.type != "cuda":
        yield {"matmul": "ieee", "convolution": "ieee", "cudnn_benchmark": False}
        return
    if config["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This CUDA device does not support BF16 training")
    settings = (
        (torch.backends, "fp32_precision", "ieee"),
        (torch.backends.cuda.matmul, "fp32_precision", policy["matmul"]),
        (torch.backends.cudnn, "fp32_precision", "ieee"),
        (torch.backends.cudnn.conv, "fp32_precision", policy["convolution"]),
        (torch.backends.cudnn, "benchmark", policy["cudnn_benchmark"]),
    )
    previous = [(owner, attribute, getattr(owner, attribute)) for owner, attribute, _ in settings]
    try:
        for owner, attribute, value in settings:
            setattr(owner, attribute, value)
        yield policy
    finally:
        for owner, attribute, value in previous:
            setattr(owner, attribute, value)


@contextmanager
def configure_compute(config, device):
    """Scope deterministic kernels and numerical policy to this run."""
    deterministic = config.get("deterministic", True)
    if type(deterministic) is not bool:
        raise ValueError("deterministic must be boolean")
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(deterministic)
        with _configure_compute(config, device) as effective:
            yield effective
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)
