"""Single-tile BF16 matmul on Blackwell (SM100): D = A @ B.

This file is the host-side scaffolding (dtype/shape checks, from_dlpack
conversion, compile cache). Fill in the @cute.jit and @cute.kernel
bodies below as you walk through the tcgen05 tutorial:
https://gau-nernst.github.io/tcgen05/
"""
import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


# Single-tile config. Pick a shape that one `tcgen05.mma cta_group::1`
# can deliver. (M, N, K) = (128, 128, 64) is a safe starting point for BF16.
TILE_M = 128
TILE_N = 128
TILE_K = 64


@cute.jit
def mma(A: cute.Tensor, B: cute.Tensor, D: cute.Tensor):
    # TODO: tile A, B, D; build TMA atoms for A/B; allocate TMEM for the
    # accumulator; launch the inner kernel.
    pass


@cute.kernel
def kernel():
    # TODO: load A/B tile into SMEM (cp.async first, then TMA); allocate
    # TMEM; issue tcgen05.mma; wait; copy TMEM -> RMEM -> GMEM.
    pass


_compile_cache: dict = {}


def mma_interface(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute D = A @ B via the Blackwell tcgen05 MMA kernel.

    Constraints (host-side, easy to relax later):
    - BF16 inputs, FP32 output. CUDA, contiguous (row-major).
    - 2-D inputs.
    - M divisible by TILE_M, N by TILE_N, K by TILE_K.
    """
    assert a.is_cuda and a.is_contiguous(), "a must be cuda + contiguous"
    assert b.is_cuda and b.is_contiguous(), "b must be cuda + contiguous"
    assert a.dim() == 2 and b.dim() == 2, "expected 2-D tensors"
    assert a.dtype == torch.bfloat16, f"expected bf16 a, got {a.dtype}"
    assert b.dtype == torch.bfloat16, f"expected bf16 b, got {b.dtype}"

    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"K mismatch: a is {a.shape}, b is {b.shape}"
    assert M % TILE_M == 0, f"M must be divisible by {TILE_M}, got {M}"
    assert N % TILE_N == 0, f"N must be divisible by {TILE_N}, got {N}"
    assert K % TILE_K == 0, f"K must be divisible by {TILE_K}, got {K}"

    if out is None:
        out = torch.empty((M, N), device=a.device, dtype=torch.float32)
    else:
        assert out.shape == (M, N), f"out must be {(M, N)}, got {tuple(out.shape)}"
        assert out.dtype == torch.float32, "out must be fp32"
        assert out.is_cuda and out.is_contiguous(), "out must be cuda + contiguous"

    a_ = from_dlpack(a, assumed_align=16)
    b_ = from_dlpack(b, assumed_align=16)
    d_ = from_dlpack(out, assumed_align=16)

    key = (a.dtype, a.shape, b.shape)
    if key not in _compile_cache:
        _compile_cache[key] = cute.compile(mma, a_, b_, d_)
    _compile_cache[key](a_, b_, d_)

    return out
