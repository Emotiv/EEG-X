"""Fused FP32 RMSNorm with deterministic first-order CUDA gradients."""

import torch
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl

SUPPORTED_WIDTHS = frozenset({8, 12, 16, 32, 64, 96, 128})
MAX_ELEMENTS = 1 << 31
MAX_WEIGHT_ROWS = 1 << 19


@triton.jit
def _rms_row_offset(rows, SHAPE: tl.constexpr, STRIDES: tl.constexpr):
    remaining = rows.to(tl.int64)
    offset = tl.full((rows.shape[0],), 0, tl.int64)
    for reverse in tl.static_range(len(SHAPE)):
        offset += (remaining % SHAPE[len(SHAPE) - 1 - reverse]) * STRIDES[len(SHAPE) - 1 - reverse]
        remaining = remaining // SHAPE[len(SHAPE) - 1 - reverse]
    return offset


@triton.jit
def _rms_forward(X, W, Y, R, M: tl.constexpr, D: tl.constexpr, EPS: tl.constexpr,
             SHAPE: tl.constexpr, STRIDES: tl.constexpr, LAST: tl.constexpr,
             WSTRIDE: tl.constexpr, HAS_WEIGHT: tl.constexpr,
             BM: tl.constexpr, BD: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    columns = tl.arange(0, BD)
    source = _rms_row_offset(rows, SHAPE, STRIDES)
    x = tl.load(X + source[:, None] + columns[None, :] * LAST,
                mask=(rows[:, None] < M) & (columns[None, :] < D), other=0).to(tl.float32)
    inverse = tl.rsqrt(tl.sum(x * x, 1) / D + EPS)
    y = x * inverse[:, None]
    if HAS_WEIGHT:
        w = tl.load(W + columns * WSTRIDE, mask=columns < D, other=0).to(tl.float32)
        y = y * w[None, :]
    tl.store(Y + rows[:, None] * D + columns[None, :], y,
             mask=(rows[:, None] < M) & (columns[None, :] < D))
    tl.store(R + rows, inverse, mask=rows < M)


@triton.jit
def _rms_backward(X, W, DY, R, DX, PART, M: tl.constexpr, D: tl.constexpr,
              SHAPE: tl.constexpr, XSTRIDES: tl.constexpr, DSTRIDES: tl.constexpr,
              XLAST: tl.constexpr, DLAST: tl.constexpr, WSTRIDE: tl.constexpr,
              HAS_WEIGHT: tl.constexpr, HAS_DW: tl.constexpr,
              BM: tl.constexpr, BD: tl.constexpr):
    group = tl.program_id(0)
    rows = group * BM + tl.arange(0, BM)
    columns = tl.arange(0, BD)
    xoffset = _rms_row_offset(rows, SHAPE, XSTRIDES)
    doffset = _rms_row_offset(rows, SHAPE, DSTRIDES)
    valid = (rows[:, None] < M) & (columns[None, :] < D)
    x = tl.load(X + xoffset[:, None] + columns[None, :] * XLAST, mask=valid, other=0).to(tl.float32)
    dy = tl.load(DY + doffset[:, None] + columns[None, :] * DLAST, mask=valid, other=0).to(tl.float32)
    inverse = tl.load(R + rows, mask=rows < M, other=0)
    if HAS_WEIGHT:
        w = tl.load(W + columns * WSTRIDE, mask=columns < D, other=0).to(tl.float32)
        weighted = dy * w[None, :]
    else:
        weighted = dy
    cross = tl.sum(weighted * x, 1) / D
    dx = (weighted - x * (cross * inverse * inverse)[:, None]) * inverse[:, None]
    tl.store(DX + rows[:, None] * D + columns[None, :], dx, mask=valid)
    if HAS_DW:
        partial = tl.sum(dy * (x * inverse[:, None]), 0)
        tl.store(PART + group * D + columns, partial, mask=columns < D)


@triton.jit
def _rms_reduce_partial(PART, NEXT, COUNT: tl.constexpr, D: tl.constexpr,
                    BR: tl.constexpr, BD: tl.constexpr):
    group = tl.program_id(0)
    rows = group * BR + tl.arange(0, BR)
    columns = tl.arange(0, BD)
    values = tl.load(PART + rows[:, None] * D + columns[None, :],
                     mask=(rows[:, None] < COUNT) & (columns[None, :] < D), other=0)
    tl.store(NEXT + group * D + columns, tl.sum(values, 0), mask=columns < D)


@triton.jit
def _rms_reduce_weight(PART, DW, COUNT: tl.constexpr, D: tl.constexpr,
                   BR: tl.constexpr, BC: tl.constexpr):
    columns = tl.program_id(0) * BC + tl.arange(0, BC)
    rows = tl.arange(0, BR)
    values = tl.load(PART + rows[:, None] * D + columns[None, :],
                     mask=(rows[:, None] < COUNT) & (columns[None, :] < D), other=0)
    tl.store(DW + columns, tl.sum(values, 0), mask=columns < D)


class _FusedRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, eps):
        if inputs.device.type != "cuda" or inputs.ndim < 1 or inputs.numel() == 0:
            raise ValueError("The fused RMSNorm requires nonempty CUDA inputs")
        if inputs.numel() >= MAX_ELEMENTS or inputs.stride(-1) not in (0, 1):
            raise ValueError("The fused RMSNorm requires a safe output size and unit or broadcast final stride")
        if inputs.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("The fused RMSNorm supports floating-point model inputs")
        width = inputs.shape[-1]
        if width not in SUPPORTED_WIDTHS:
            raise ValueError("The fused RMSNorm requires a validated feature width")
        if weight is not None and (weight.device != inputs.device or weight.shape != (width,)
                                   or weight.dtype != torch.float32):
            raise ValueError("The fused RMSNorm requires an FP32 scale matching the final dimension")
        rows = inputs.numel() // width
        if weight is not None and (weight.stride(0) not in (0, 1) or rows > MAX_WEIGHT_ROWS):
            raise ValueError("The fused RMSNorm requires a supported scale layout and reduction size")
        block_rows = 128 if width <= 16 else 16
        block_width = triton.next_power_of_2(width)
        output = torch.empty(inputs.shape, device=inputs.device, dtype=inputs.dtype)
        inverse = torch.empty((rows,), device=inputs.device, dtype=torch.float32)
        pointer = weight if weight is not None else inputs
        with torch.cuda.device(inputs.device):
            _rms_forward[(triton.cdiv(rows, block_rows),)](
                inputs, pointer, output, inverse, rows, width, eps,
                tuple(inputs.shape[:-1]), tuple(inputs.stride()[:-1]), inputs.stride(-1),
                weight.stride(0) if weight is not None else 0, weight is not None,
                block_rows, block_width, num_warps=4,
            )
        ctx.save_for_backward(inputs, pointer, inverse)
        ctx.has_weight = weight is not None
        ctx.width, ctx.rows, ctx.block_rows = width, rows, block_rows
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        inputs, weight, inverse = ctx.saved_tensors
        if grad_output.stride(-1) not in (0, 1):
            grad_output = grad_output.contiguous()
        rows, width, block_rows = ctx.rows, ctx.width, ctx.block_rows
        block_width = triton.next_power_of_2(width)
        groups = triton.cdiv(rows, block_rows)
        need_weight = ctx.has_weight and ctx.needs_input_grad[1]
        dx = torch.empty(inputs.shape, device=inputs.device, dtype=inputs.dtype)
        partial = torch.empty((groups, width), device=inputs.device, dtype=torch.float32) if need_weight else dx
        with torch.cuda.device(inputs.device):
            _rms_backward[(groups,)](
                inputs, weight, grad_output, inverse, dx, partial, rows, width,
                tuple(inputs.shape[:-1]), tuple(inputs.stride()[:-1]), tuple(grad_output.stride()[:-1]),
                inputs.stride(-1), grad_output.stride(-1), weight.stride(0) if ctx.has_weight else 0,
                ctx.has_weight, need_weight, block_rows, block_width, num_warps=4,
            )
            if need_weight:
                reduced_rows = triton.cdiv(groups, 128)
                reduced = torch.empty((reduced_rows, width), device=inputs.device, dtype=torch.float32)
                dw = torch.empty(weight.shape, device=weight.device, dtype=weight.dtype)
                _rms_reduce_partial[(reduced_rows,)](partial, reduced, groups, width, 128, block_width, num_warps=4)
                _rms_reduce_weight[(triton.cdiv(width, 8),)](
                    reduced, dw, reduced_rows, width, triton.next_power_of_2(reduced_rows), 8, num_warps=4,
                )
            else:
                dw = None
        return dx, dw, None


def fused_rms_norm(inputs, *, weight=None, eps=1e-6):
    """Normalize supported model shapes with first-order derivatives."""
    return _FusedRMSNorm.apply(inputs, weight, eps)
