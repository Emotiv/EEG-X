"""Fuse fixed FP32 FIR responses, kernel competition and temporal counts on CUDA."""

import torch
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl

FORWARD_TILE = 32


def temporary_elements_per_signal(samples, groups, kernels, maximum_width):
    """Bound global temporaries for the actual forward launch tile."""
    return max(triton.cdiv(samples, FORWARD_TILE) * groups * kernels,
               samples * maximum_width)


@triton.jit
def _probabilities(
    signal,
    filters,
    n,
    positions,
    channels,
    SAMPLES: tl.constexpr,
    WIDTH: tl.constexpr,
    DILATION: tl.constexpr,
    GROUPS: tl.constexpr,
    K: tl.constexpr,
    TEMPERATURE: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    padding: tl.constexpr = (WIDTH - 1) * DILATION // 2
    if GROUPS * K < 16:
        scores = tl.full((BLOCK_T, GROUPS * K), 0, tl.float32)
        for tap in tl.static_range(WIDTH):
            samples = positions + tap * DILATION - padding
            values = tl.load(
                signal + n * SAMPLES + samples,
                mask=(samples >= 0) & (samples < SAMPLES) & (positions < SAMPLES),
                other=0,
            )
            weights = tl.load(filters + channels * WIDTH + tap)
            scores += values[:, None] * weights[None, :]
    else:
        taps = tl.arange(0, max(16, triton.next_power_of_2(WIDTH)))
        samples = positions[None, :] + taps[:, None] * DILATION - padding
        windows = tl.load(
            signal + n * SAMPLES + samples,
            mask=(samples >= 0) & (samples < SAMPLES) &
                 (positions[None, :] < SAMPLES) & (taps[:, None] < WIDTH),
            other=0,
        )
        weights = tl.load(
            filters + channels[:, None] * WIDTH + taps[None, :],
            mask=taps[None, :] < WIDTH,
            other=0,
        )
        scores = tl.dot(tl.trans(windows), tl.trans(weights), input_precision="ieee")
    scores = (scores / TEMPERATURE).reshape((BLOCK_T, GROUPS, K))
    scores -= tl.max(scores, 2)[:, :, None]
    numerator = tl.exp(scores)
    probabilities = numerator / tl.sum(numerator, 2)[:, :, None]
    return tl.where((positions < SAMPLES)[:, None, None], probabilities, 0)


@triton.jit
def _forward(
    signal,
    filters,
    partial,
    SAMPLES: tl.constexpr,
    WIDTH: tl.constexpr,
    DILATION: tl.constexpr,
    GROUPS: tl.constexpr,
    K: tl.constexpr,
    TEMPERATURE: tl.constexpr,
    TILES: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    n, tile = tl.program_id(0), tl.program_id(1)
    positions = tile * BLOCK_T + tl.arange(0, BLOCK_T)
    channels = tl.arange(0, GROUPS * K)
    probabilities = _probabilities(
        signal,
        filters,
        n,
        positions,
        channels,
        SAMPLES,
        WIDTH,
        DILATION,
        GROUPS,
        K,
        TEMPERATURE,
        BLOCK_T,
    )
    counts = tl.sum(probabilities, 0).reshape((GROUPS * K,))
    tl.store(partial + (n * TILES + tile) * GROUPS * K + channels, counts)


@triton.jit
def _backward(
    signal,
    filters,
    count_gradient,
    window_gradient,
    SAMPLES: tl.constexpr,
    WIDTH: tl.constexpr,
    DILATION: tl.constexpr,
    GROUPS: tl.constexpr,
    K: tl.constexpr,
    TEMPERATURE: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    n, tile = tl.program_id(0), tl.program_id(1)
    positions = tile * BLOCK_T + tl.arange(0, BLOCK_T)
    channels = tl.arange(0, GROUPS * K)
    probabilities = _probabilities(
        signal,
        filters,
        n,
        positions,
        channels,
        SAMPLES,
        WIDTH,
        DILATION,
        GROUPS,
        K,
        TEMPERATURE,
        BLOCK_T,
    )
    upstream = tl.load(count_gradient + n * GROUPS * K + channels).reshape((GROUPS, K))
    centered = upstream[None, :, :] - tl.sum(probabilities * upstream[None, :, :], 2)[:, :, None]
    response_gradient = (probabilities * centered / TEMPERATURE).reshape((BLOCK_T, GROUPS * K))
    if GROUPS * K < 16:
        # The FP32 dot primitive requires K >= 16. Tiny banks use direct reduction.
        for tap in tl.static_range(WIDTH):
            weights = tl.load(filters + channels * WIDTH + tap)
            value = tl.sum(response_gradient * weights[None, :], 1)
            tl.store(
                window_gradient + (n * WIDTH + tap) * SAMPLES + positions,
                value,
                mask=positions < SAMPLES,
            )
    else:
        # IEEE FP32 dot avoids one cross-warp reduction for each filter tap.
        taps = tl.arange(0, triton.next_power_of_2(WIDTH))
        weights = tl.load(
            filters + channels[:, None] * WIDTH + taps[None, :],
            mask=taps[None, :] < WIDTH,
            other=0,
        )
        values = tl.dot(response_gradient, weights, input_precision="ieee")
        tl.store(
            window_gradient + (n * WIDTH + taps[:, None]) * SAMPLES + positions[None, :],
            tl.trans(values),
            mask=(taps[:, None] < WIDTH) & (positions[None, :] < SAMPLES),
        )


@triton.jit
def _fold_1d(windows, result, COUNT: tl.constexpr, SAMPLES: tl.constexpr,
             WIDTH: tl.constexpr, DILATION: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    example = offset // SAMPLES
    position = offset % SAMPLES
    value = tl.full((BLOCK,), 0, tl.float32)
    # Native col2im visits increasing source columns, hence decreasing taps.
    for reverse in tl.static_range(WIDTH):
        tap = WIDTH - 1 - reverse
        source = position + (WIDTH - 1) * DILATION // 2 - tap * DILATION
        contribution = tl.load(
            windows + (example * WIDTH + tap) * SAMPLES + source,
            mask=(offset < COUNT) & (source >= 0) & (source < SAMPLES),
            other=0,
        )
        value += contribution
    tl.store(result + offset, value, mask=offset < COUNT)


class _FusedCounts(torch.autograd.Function):
    @staticmethod
    def forward(ctx, signal, filters, dilation, temperature, groups, kernels):
        signal, filters = signal.contiguous(), filters.contiguous()
        samples = signal.shape[-1]
        tiles = triton.cdiv(samples, FORWARD_TILE)
        partial = torch.empty(
            (len(signal), tiles, groups * kernels), device=signal.device, dtype=torch.float32
        )
        with torch.cuda.device(signal.device):
            _forward[(len(signal), tiles)](
                signal,
                filters,
                partial,
                samples,
                filters.shape[-1],
                dilation,
                groups,
                kernels,
                temperature,
                tiles,
                FORWARD_TILE,
                num_warps=2,
            )
        ctx.save_for_backward(signal, filters)
        ctx.dictionary_config = (dilation, temperature, groups, kernels)
        return partial.sum(1)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        signal, filters = ctx.saved_tensors
        dilation, temperature, groups, kernels = ctx.dictionary_config
        samples, width = signal.shape[-1], filters.shape[-1]
        windows = torch.empty(
            (len(signal), width, samples), device=signal.device, dtype=torch.float32
        )
        with torch.cuda.device(signal.device):
            _backward[(len(signal), triton.cdiv(samples, 16))](
                signal,
                filters,
                gradient.contiguous(),
                windows,
                samples,
                width,
                dilation,
                groups,
                kernels,
                temperature,
                16,
                num_warps=2,
            )
        result = torch.empty_like(signal)
        with torch.cuda.device(signal.device):
            _fold_1d[(triton.cdiv(result.numel(), 256),)](
                windows, result, result.numel(), samples, width, dilation, 256, num_warps=4,
            )
        return result, None, None, None, None, None



def fused_counts(signal, filters, dilation, temperature, groups, kernels):
    """Compute counts without storing full response/probability tensors in global memory."""
    return _FusedCounts.apply(signal, filters, dilation, temperature, groups, kernels)
