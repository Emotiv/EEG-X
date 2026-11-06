from __future__ import annotations

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _module_device(module: nn.Module) -> torch.device:
    for tensor in module.parameters():
        return tensor.device
    for tensor in module.buffers():
        return tensor.device
    return torch.device("cpu")


def compile_hydra_transform(hydra_transform: nn.Module, *, enabled: bool) -> nn.Module:
    """Compile CUDA Hydra transforms while leaving CPU/debug paths eager."""

    if not enabled:
        return hydra_transform
    if _module_device(hydra_transform).type != "cuda":
        return hydra_transform
    if not hasattr(torch, "compile"):
        raise RuntimeError(
            "torch.compile is unavailable in this PyTorch build. Re-run with --no_compile_hydra_loss."
        )
    return torch.compile(hydra_transform, backend="inductor", dynamic=False)


def _seeded_generator(seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def _hydra_random_kwargs(seed: int | None) -> dict:
    generator = _seeded_generator(seed)
    return {} if generator is None else {"generator": generator}


class HydraMultivariateGPU(nn.Module):
    """Device-safe copy of the Hydra transform used by the original EEG-X Lightning path."""

    def __init__(
        self,
        input_length: int,
        num_channels: int,
        k: int = 8,
        g: int = 64,
        max_num_channels: int = 8,
        channel_sample_size: int | None = None,
        seed: int | None = None,
        kernel_size: int = 15,
        eps: float = 1e-5,
    ) -> None:
        """
        For univariate use:
        HydraMultivariateGPU(
            ...,
            num_channels=1,
            channel_sample_size=1,
        )
        """
        super().__init__()

        self.k = int(k)
        self.g = int(g)
        self.eps = float(eps)
        self.kernel_size = int(kernel_size)
        if self.kernel_size <= 1:
            raise ValueError("kernel_size must be > 1")
        if self.g < 2 or self.g % 2 != 0:
            raise ValueError("old Hydra requires g to be an even value >= 2")
        random_kwargs = _hydra_random_kwargs(seed)

        max_exponent = np.log2((input_length - 1) / (self.kernel_size - 1))
        dilations = 2 ** torch.arange(int(max_exponent) + 1)
        paddings = torch.div((self.kernel_size - 1) * dilations, 2, rounding_mode="floor").int()
        self.register_buffer("dilations", dilations)
        self.register_buffer("paddings", paddings)
        # Keep Python scalar copies so torch.compile does not graph-break on Tensor.item().
        self.dilation_values = tuple(int(value) for value in dilations.tolist())
        self.padding_values = tuple(int(value) for value in paddings.tolist())
        self.num_dilations = len(self.dilations)

        self.divisor = min(2, self.g)
        self.h = self.g // self.divisor

        weight = torch.randn(
            self.num_dilations,
            self.divisor,
            self.k * self.h,
            1,
            self.kernel_size,
            **random_kwargs,
        )
        weight = weight - weight.mean(-1, keepdims=True)
        weight = weight / weight.abs().sum(-1, keepdims=True)
        self.register_buffer("W", weight)

        if channel_sample_size is None:
            num_channels_per = int(np.clip(num_channels // 2, 2, max_num_channels))
        else:
            num_channels_per = int(channel_sample_size)
            if num_channels_per <= 0:
                raise ValueError("channel_sample_size must be > 0")
        self.channel_sample_size = num_channels_per
        channel_index = torch.randint(
            0,
            num_channels,
            (self.num_dilations, self.divisor, self.h, num_channels_per),
            **random_kwargs,
        )
        self.register_buffer("I", channel_index)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_examples = x.shape[0]

        if self.divisor <= 1:
            raise NotImplementedError("HydraMultivariateGPU only supports divisor = 1 for now")

        diff_x = torch.diff(x)
        outputs = []

        for dilation_index in range(self.num_dilations):
            dilation = self.dilation_values[dilation_index]
            padding = self.padding_values[dilation_index]

            for diff_index in range(self.divisor):
                if diff_index == 0:
                    signal = x[:, self.I[dilation_index, diff_index]].sum(2)
                else:
                    signal = diff_x[:, self.I[dilation_index, diff_index]].sum(2)

                transformed = F.conv1d(
                    signal,
                    self.W[dilation_index, diff_index],
                    dilation=dilation,
                    padding=padding,
                    groups=self.h,
                ).view(num_examples, self.h, self.k, -1)

                max_values, max_indices = transformed.max(2)
                count_max = transformed.new_zeros(num_examples, self.h, self.k)
                min_indices = transformed.argmin(2)
                count_min = transformed.new_zeros(num_examples, self.h, self.k)

                count_max.scatter_add_(-1, max_indices, max_values)
                count_min.scatter_add_(-1, min_indices, torch.ones_like(max_values))

                outputs.append(count_max)
                outputs.append(count_min)

        features = torch.cat(outputs, 1).view(num_examples, -1)
        return features.clamp_min(self.eps).sqrt()


class ChannelWiseHydraMultivariateGPU(nn.Module):
    """Channel-wise Hydra transform with one grouped convolution per EEG channel."""

    def __init__(
        self,
        input_length: int,
        num_channels: int,
        k: int = 8,
        g: int = 64,
        max_num_channels: int = 8,
        seed: int | None = None,
        kernel_size: int = 15,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()

        self.num_channels = int(num_channels)
        self.k = int(k)
        self.g = int(g)
        self.kernel_size = int(kernel_size)
        self.eps = float(eps)
        if self.kernel_size <= 1:
            raise ValueError("kernel_size must be > 1")
        if self.g < 2 or self.g % 2 != 0:
            raise ValueError("channelwise Hydra requires g to be an even value >= 2")
        random_kwargs = _hydra_random_kwargs(seed)

        max_exponent = np.log2((input_length - 1) / (self.kernel_size - 1))
        dilations = 2 ** torch.arange(int(max_exponent) + 1)
        paddings = torch.div((self.kernel_size - 1) * dilations, 2, rounding_mode="floor").int()
        self.register_buffer("dilations", dilations)
        self.register_buffer("paddings", paddings)
        self.dilation_values = tuple(int(value) for value in dilations.tolist())
        self.padding_values = tuple(int(value) for value in paddings.tolist())
        self.num_dilations = len(self.dilations)

        self.divisor = min(2, self.g)
        self.h = self.g // self.divisor

        weight = torch.randn(
            self.num_dilations,
            self.divisor,
            self.k * self.h,
            1,
            self.kernel_size,
            **random_kwargs,
        )
        weight = weight - weight.mean(-1, keepdims=True)
        weight = weight / weight.abs().sum(-1, keepdims=True)
        self.register_buffer("W", weight)

        self.output_channels = self.h * self.num_channels
        self.num_features = self.num_dilations * self.divisor * self.k * self.output_channels * 2

    def batch(self, x: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
        num_examples = x.shape[0]
        if num_examples <= batch_size:
            return self(x)

        outputs = []
        for batch in torch.arange(num_examples, device=x.device).split(batch_size):
            outputs.append(self(x[batch]))
        return torch.cat(outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_examples = x.shape[0]
        if x.shape[1] != self.num_channels:
            raise ValueError(
                f"ChannelWiseHydraMultivariateGPU expected {self.num_channels} channels, "
                f"got {x.shape[1]}"
            )
        diff_x = torch.diff(x)
        outputs = []

        for dilation_index in range(self.num_dilations):
            dilation = self.dilation_values[dilation_index]
            padding = self.padding_values[dilation_index]

            for diff_index in range(self.divisor):
                signal = x if diff_index == 0 else diff_x
                signal = signal.unsqueeze(2).expand(-1, -1, self.h, -1)
                signal = signal.reshape(num_examples, self.output_channels, -1)
                weight = self.W[dilation_index, diff_index].repeat(self.num_channels, 1, 1)

                transformed = F.conv1d(
                    signal,
                    weight,
                    dilation=dilation,
                    padding=padding,
                    groups=self.output_channels,
                ).view(num_examples, self.num_channels, self.h, self.k, -1)

                max_values, max_indices = transformed.max(3)
                count_max = transformed.new_zeros(num_examples, self.num_channels, self.h, self.k)
                count_max.scatter_add_(-1, max_indices, max_values)

                min_indices = transformed.argmin(3)
                count_min = transformed.new_zeros(num_examples, self.num_channels, self.h, self.k)
                count_min.scatter_add_(-1, min_indices, torch.ones_like(max_values))

                outputs.append(count_max)
                outputs.append(count_min)

        features = torch.cat(
            [output.reshape(num_examples, self.num_channels, -1) for output in outputs], 2
        )
        features = features.reshape(num_examples, -1)
        return features.clamp_min(self.eps).sqrt()


class SoftChannelWiseHydra(ChannelWiseHydraMultivariateGPU):
    """Smooth competition for reconstruction loss, distinct from paper Hydra.

    Positive and negative response softmaxes replace max and argmin routing.
    Both feature branches propagate gradients. Temporal means keep feature
    magnitudes independent of window length. No trainable loss normalizer is used.
    """

    def __init__(self, *args, temperature: float = 1.0, **kwargs):
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        super().__init__(*args, **kwargs)
        self.temperature = float(temperature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.num_channels:
            raise ValueError("Expected EEG with shape [batch, configured channels, time]")
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            batch, channels, _ = x.shape
            outputs = []
            for dilation_index, dilation in enumerate(self.dilation_values):
                padding = (self.kernel_size - 1) * dilation // 2
                for branch, signal in enumerate((x, x.diff())):
                    # Channels share filters, but never compete with each other.
                    responses = F.conv1d(
                        signal.reshape(batch * channels, 1, -1),
                        self.W[dilation_index, branch],
                        dilation=dilation,
                        padding=padding,
                    ).reshape(batch, channels, self.h, self.k, -1)
                    logits = responses / self.temperature
                    positive = logits.softmax(dim=3)
                    negative = (-logits).softmax(dim=3)
                    outputs.extend(
                        (
                            (positive * F.softplus(responses)).mean(-1),
                            negative.mean(-1),
                        )
                    )
            return (torch.cat(outputs, dim=2).flatten(1) + self.eps).sqrt()


def build_hydra_transform(
    variant: str,
    *,
    input_length: int,
    num_channels: int,
    k: int = 8,
    g: int = 64,
    max_num_channels: int = 8,
    seed: int | None = None,
    temperature: float = 1.0,
) -> nn.Module:
    if variant == "soft_channelwise":
        return SoftChannelWiseHydra(
            input_length=input_length,
            num_channels=num_channels,
            k=k,
            g=g,
            max_num_channels=max_num_channels,
            seed=seed,
            temperature=temperature,
        )
    if variant == "old":
        return HydraMultivariateGPU(
            input_length=input_length,
            num_channels=num_channels,
            k=k,
            g=g,
            max_num_channels=max_num_channels,
            seed=seed,
        )
    if variant == "channelwise":
        return ChannelWiseHydraMultivariateGPU(
            input_length=input_length,
            num_channels=num_channels,
            k=k,
            g=g,
            max_num_channels=max_num_channels,
            seed=seed,
        )
    raise ValueError(f"Unsupported Hydra loss variant: {variant}")
