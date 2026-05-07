"""Single-tile BF16 matmul on Blackwell (SM100): D = A @ B.

This file is the host-side scaffolding (dtype/shape checks, from_dlpack
conversion, compile cache). Fill in the @cute.jit and @cute.kernel
bodies below as you walk through the tcgen05 tutorial:
https://gau-nernst.github.io/tcgen05/

Layouts:
  A: (M, K) row-major  -> K is contiguous (K-major).
  B: (K, N) column-major -> K is contiguous (K-major).
  D: (M, N) row-major BF16.

Both operands are K-major in storage; tcgen05 BF16 MMA atoms expect
the contraction dim contiguous in SMEM.
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
    # TMEM; issue tcgen05.mma; wait; copy TMEM -> RMEM -> BF16 -> GMEM.
    pass


_compile_cache: dict = {}


def mma_interface(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute D = A @ B via the Blackwell tcgen05 MMA kernel.

    Layout / dtype contract:
    - A: (M, K) BF16, row-major (K contiguous).
    - B: (K, N) BF16, column-major (K contiguous; stride (1, K)).
    - D: (M, N) BF16, row-major. FP32 accumulator lives in TMEM and is
      cast down to BF16 before the GMEM store.
    - M % TILE_M == 0, N % TILE_N == 0, K % TILE_K == 0.
    """
    assert a.is_cuda and b.is_cuda, "a, b must be cuda"
    assert a.dim() == 2 and b.dim() == 2, "expected 2-D tensors"
    assert a.dtype == torch.bfloat16, f"expected bf16 a, got {a.dtype}"
    assert b.dtype == torch.bfloat16, f"expected bf16 b, got {b.dtype}"

    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"K mismatch: a is {a.shape}, b is {b.shape}"
    assert M % TILE_M == 0, f"M must be divisible by {TILE_M}, got {M}"
    assert N % TILE_N == 0, f"N must be divisible by {TILE_N}, got {N}"
    assert K % TILE_K == 0, f"K must be divisible by {TILE_K}, got {K}"

    # A: row-major (M, K) — strides (K, 1).
    assert a.stride() == (K, 1), f"a must be row-major, got stride {a.stride()}"
    # B: column-major (K, N) — strides (1, K).
    assert b.stride() == (1, K), f"b must be column-major, got stride {b.stride()}"

    if out is None:
        out = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    else:
        assert out.shape == (M, N), f"out must be {(M, N)}, got {tuple(out.shape)}"
        assert out.dtype == torch.bfloat16, "out must be bf16"
        assert out.is_cuda and out.is_contiguous(), "out must be cuda + contiguous"

    a_ = from_dlpack(a, assumed_align=16)
    b_ = from_dlpack(b, assumed_align=16)
    d_ = from_dlpack(out, assumed_align=16)

    key = (a.dtype, a.shape, b.shape)
    if key not in _compile_cache:
        _compile_cache[key] = cute.compile(mma, a_, b_, d_)
    _compile_cache[key](a_, b_, d_)

    return out
