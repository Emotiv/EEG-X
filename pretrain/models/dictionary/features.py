"""Differentiable winner counts with the manuscript's convolutional dictionary."""

from contextlib import contextmanager
import threading

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


_DICTIONARY_MATH_LOCK = threading.RLock()


def _temporal_probability_counts(responses, temperature):
    if temperature != 1.0:
        responses = responses / temperature
    return responses.softmax(dim=-1).sum(dim=1).flatten(1)


@contextmanager
def _ieee_dictionary_math(device):
    """Keep dictionary forward and input gradients in the original FP32 policy."""
    with torch.autocast(device_type=device.type, enabled=False):
        if device.type != "cuda":
            yield
            return
        # CUDA precision is process-wide. Serialize dictionary scopes so nested
        # or concurrent dictionary calls restore the policy they inherited.
        with _DICTIONARY_MATH_LOCK:
            previous = torch.backends.cuda.matmul.fp32_precision
            try:
                torch.backends.cuda.matmul.fp32_precision = "ieee"
                yield
            finally:
                torch.backends.cuda.matmul.fp32_precision = previous


class _FixedDictionaryConvolution(torch.autograd.Function):
    """IEEE fixed-filter convolution with a deterministic matrix-product gradient."""

    @staticmethod
    def forward(ctx, signal, filters, dilation):
        ctx.save_for_backward(filters)
        ctx.dilation = dilation
        ctx.samples = signal.shape[-1]
        support = (filters.shape[-1] - 1) * dilation + 1
        padding = support // 2
        windows = F.pad(signal[:, 0], (padding, padding)).unfold(-1, support, 1)
        windows = windows[..., ::dilation].contiguous()
        with _ieee_dictionary_math(signal.device):
            return F.linear(windows, filters[:, 0])

    @staticmethod
    def backward(ctx, gradient):
        (filters,) = ctx.saved_tensors
        samples, dilation = ctx.samples, ctx.dilation
        width = filters.shape[-1]
        padding = (width - 1) * dilation // 2
        with _ieee_dictionary_math(gradient.device):
            windows_gradient = gradient.matmul(filters[:, 0])
        # col2im performs the overlapping-tap reduction in one native operation.
        # Keeping the compact tap axis avoids gradients for dilation holes and
        # the Python slices and kernel launch previously needed for every tap.
        signal_gradient = F.fold(
            windows_gradient.transpose(1, 2).contiguous(),
            output_size=(1, samples),
            kernel_size=(1, width),
            dilation=(1, dilation),
            padding=(0, padding),
        )
        return signal_gradient.squeeze(2), None, None


class SoftDiCT(nn.Module):
    """Keep fixed filters and temporal counts; soften only kernel assignments.

    Channels share the seeded dictionary but compete independently. Features use
    temporal sums, without raw/difference branches or response-amplitude weights.
    """

    def __init__(self, objective, *, max_response_elements=128 * 1024 * 1024, backend="torch"):
        super().__init__()
        from pretrain.objectives import validate_objective

        validate_objective(objective)
        if type(max_response_elements) is not int or max_response_elements < 1:
            raise ValueError("Dictionary response limit must be a positive integer")
        self.max_response_elements = max_response_elements
        if not isinstance(backend, str) or backend not in {"torch", "triton"}:
            raise ValueError("Dictionary backend must be torch or triton")
        self.backend = backend
        self.groups = objective["groups"]
        self.k = objective["k"]
        self.temperature = objective["temperature"]
        self.kernel_sizes = tuple(objective["kernel_sizes"])
        self.dilations = tuple(objective["dilations"])
        if backend == "triton" and (
            self.groups & (self.groups - 1) or self.k & (self.k - 1) or self.groups * self.k > 256
        ):
            raise ValueError(
                "Fused dictionary requires power-of-two groups/kernels and at most 256 filters"
            )
        generator = torch.Generator(device="cpu").manual_seed(objective["seed"])
        for size in self.kernel_sizes:
            weight = torch.randn(
                len(self.dilations), self.groups * self.k, 1, size, generator=generator
            )
            weight -= weight.mean(dim=-1, keepdim=True)
            weight /= weight.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            self.register_buffer(f"filters_{size}", weight)

    def _counts(self, signal, filters, dilation):
        """Pool one dictionary view with contiguous kernel competition."""
        samples = signal.shape[-1]
        responses = _FixedDictionaryConvolution.apply(signal, filters, dilation).reshape(
            len(signal), samples, self.groups, self.k
        )
        # A contiguous kernel axis uses the row-softmax kernel instead of the
        # strided spatial-softmax path. Each channel still competes independently.
        return _temporal_probability_counts(responses, self.temperature)

    def forward(self, eeg):
        if eeg.ndim != 3 or min(eeg.shape) < 1:
            raise ValueError("Expected nonempty EEG [batch, channels, time]")
        if self.backend == "triton" and eeg.device.type != "cuda":
            raise ValueError("Fused dictionary backend requires CUDA")
        with torch.autocast(device_type=eeg.device.type, enabled=False):
            batch, channels, samples = eeg.shape
            signal = eeg.float().reshape(batch * channels, 1, samples)
            elements_per_signal = self.groups * self.k * samples
            if self.backend == "triton":
                from .cuda import temporary_elements_per_signal

                elements_per_signal = temporary_elements_per_signal(
                    samples, self.groups, self.k, max(self.kernel_sizes)
                )
            chunk_size = max(1, self.max_response_elements // elements_per_signal)
            chunks = []
            for part in signal.split(chunk_size):
                features = []
                for size in self.kernel_sizes:
                    filters = getattr(self, f"filters_{size}")
                    for index, dilation in enumerate(self.dilations):
                        arguments = (part, filters[index], dilation)
                        if self.backend == "triton":
                            from .cuda import fused_counts

                            counts = fused_counts(
                                part,
                                filters[index],
                                dilation,
                                self.temperature,
                                self.groups,
                                self.k,
                            )
                        elif torch.is_grad_enabled() and part.requires_grad:
                            # Pool first and recompute the temporary probabilities
                            # during backward, rather than retaining every view.
                            counts = checkpoint(
                                self._counts,
                                *arguments,
                                use_reentrant=False,
                                preserve_rng_state=False,
                            )
                        else:
                            counts = self._counts(*arguments)
                        features.append(counts)
                chunks.append(torch.cat(features, dim=-1))
            return torch.cat(chunks).reshape(batch, channels, -1).flatten(1)
