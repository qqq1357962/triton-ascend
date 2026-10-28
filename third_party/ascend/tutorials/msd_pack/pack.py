"""Bit-exact MSD packing with explicit full-tile and tail paths.

Input and output are contiguous [M, K] uint8 tensors, with K divisible by four.
This is a standalone layout component, not a model or quantization frontend.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _pack_msd(X, Y, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    offsets = tile * BLOCK + tl.arange(0, BLOCK)
    width: tl.constexpr = K // 4
    src = X + row * (K // 2)
    dst = Y + row * (K // 2)

    a = tl.load(src + 2 * offsets, offsets < width, 0)
    b = tl.load(src + 2 * offsets + 1, offsets < width, 0)

    hi = ((a >> 4) & 15) | ((a >> 8) & 240) | ((b << 4) & 3840) | (b & 61440)
    lo = ((a & 15) | ((a >> 4) & 240) | ((b << 8) & 3840) | ((b << 4) & 61440)) ^ 34952

    tl.store(dst + offsets, hi, offsets < width)
    tl.store(dst + width + offsets, lo, offsets < width)


def pack_msd(x: torch.Tensor, *, out: torch.Tensor | None = None, block_size: int = 1024):
    """Pack adjacent signed nibbles without changing their bit representation.

    Each input byte holds a high and low nibble. For every adjacent byte pair,
    high nibbles form one output byte in the first half of the row; low nibbles
    form one output byte in the second half, with their sign bits flipped.
    ``out`` permits allocation-free graph capture. In-place packing is unsafe.
    """
    if x.device.type != "npu" or x.dtype != torch.uint8 or x.ndim != 2 or not x.is_contiguous():
        raise ValueError("x must be a contiguous 2D uint8 NPU tensor")
    rows, k = x.shape
    if k % 4:
        raise ValueError("K must be divisible by four for paired uint16 loads")
    if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size < 16 or block_size > 1024:
        raise ValueError("block_size must be a power of two in [16, 1024]")
    if block_size & (block_size - 1):
        raise ValueError("block_size must be a power of two in [16, 1024]")
    if out is None:
        out = torch.empty_like(x)
    if out.shape != x.shape or out.dtype != x.dtype or out.device != x.device or not out.is_contiguous():
        raise ValueError("out must match x's shape, dtype and device and be contiguous")
    if rows == 0 or k == 0:
        return out
    if x.storage_offset() % 2 or out.storage_offset() % 2:
        raise ValueError("uint16 views require two-byte-aligned storage offsets")
    if x.untyped_storage().data_ptr() == out.untyped_storage().data_ptr():
        raise ValueError("input and output must not share storage")
    _pack_msd[(rows, triton.cdiv(k // 4, block_size))](
        x.view(torch.uint16), out.view(torch.uint16), k, block_size, num_warps=4
    )
    return out
