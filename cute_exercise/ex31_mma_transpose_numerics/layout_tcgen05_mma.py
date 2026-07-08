"""Layout-parameterized wrapper around the custom ex21 tcgen05 MMA kernel."""

from __future__ import annotations

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import sm100 as sm100_utils
from cutlass.utils.layout import LayoutEnum

from cute_exercise.ex21_tcgen05_mma.kernel import (
    TILE_K,
    TILE_M,
    TILE_N,
    kernel,
)
from cute_exercise.ex31_mma_transpose_numerics.transpose_analysis import (
    major_mode_name,
    validate_operand_layout,
)


_compile_cache: dict = {}


def _operand_major_mode(layout: str) -> tcgen05.OperandMajorMode:
    name = major_mode_name(layout)
    if name == "K":
        return tcgen05.OperandMajorMode.K
    if name == "MN":
        return tcgen05.OperandMajorMode.MN
    raise ValueError(f"unknown major mode {name!r}")


@cute.jit
def mma_layout(
    A: cute.Tensor,
    B: cute.Tensor,
    D: cute.Tensor,
    a_major_mode: cutlass.Constexpr,
    b_major_mode: cutlass.Constexpr,
):
    tile_m = TILE_M
    tile_n = TILE_N
    tile_k = TILE_K
    tiler = (tile_m, tile_n, tile_k)

    M, N = D.shape
    block = (128, 1, 1)
    grid = (cute.ceil_div(M, tile_m), cute.ceil_div(N, tile_n), 1)

    tiled_mma = sm100_utils.make_trivial_tiled_mma(
        ab_dtype=cutlass.BFloat16,
        a_leading_mode=a_major_mode,
        b_leading_mode=b_major_mode,
        acc_dtype=cutlass.Float32,
        cta_group=tcgen05.CtaGroup.ONE,
        mma_tiler_mn=(TILE_M, TILE_N),
    )

    smem_layout_a = sm100_utils.make_smem_layout_a(
        tiled_mma, tiler, cutlass.BFloat16, num_stages=1
    )
    smem_layout_b = sm100_utils.make_smem_layout_b(
        tiled_mma, tiler, cutlass.BFloat16, num_stages=1
    )
    smem_layout_d = sm100_utils.make_smem_layout_epi(
        cutlass.BFloat16,
        LayoutEnum.ROW_MAJOR,
        (tile_m, tile_n),
        1,
    )

    tma_load_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
    tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
        tma_load_op,
        A,
        cute.select(smem_layout_a, mode=[0, 1, 2]),
        tiler,
        tiled_mma,
        cluster_shape_vmnk=(1, 1, 1, 1),
    )
    tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
        tma_load_op,
        B,
        cute.select(smem_layout_b, mode=[0, 1, 2]),
        tiler,
        tiled_mma,
        cluster_shape_vmnk=(1, 1, 1, 1),
    )

    tma_store_op = cpasync.CopyBulkTensorTileS2GOp()
    tma_atom_d, tma_tensor_d = cpasync.make_tiled_tma_atom(
        tma_store_op,
        D,
        cute.select(smem_layout_d, mode=[0, 1]),
        (tile_m, tile_n),
    )

    kernel(
        tma_tensor_a,
        tma_tensor_b,
        tma_tensor_d,
        tma_atom_a,
        tma_atom_b,
        tma_atom_d,
        tiled_mma,
        smem_layout_a,
        smem_layout_b,
        smem_layout_d,
        tiler,
    ).launch(grid=grid, block=block)


def layout_mma_interface(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    layout_a: str,
    layout_b: str,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute BF16 ``A @ B.T`` with explicit tcgen05 A/B operand-major modes."""
    assert a.is_cuda and b.is_cuda, "a, b must be cuda"
    assert a.dim() == 2 and b.dim() == 2, "expected 2-D tensors"
    assert a.dtype == torch.bfloat16, f"expected bf16 a, got {a.dtype}"
    assert b.dtype == torch.bfloat16, f"expected bf16 b, got {b.dtype}"
    validate_operand_layout(a, layout_a)
    validate_operand_layout(b, layout_b)

    m, k = a.shape
    n, k2 = b.shape
    assert k == k2, f"K mismatch: a is {a.shape}, b is {b.shape}"
    assert m % TILE_M == 0, f"M must be divisible by {TILE_M}, got {m}"
    assert n % TILE_N == 0, f"N must be divisible by {TILE_N}, got {n}"
    assert k % TILE_K == 0, f"K must be divisible by {TILE_K}, got {k}"

    if out is None:
        out = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
    else:
        assert out.shape == (m, n), f"out must be {(m, n)}, got {tuple(out.shape)}"
        assert out.dtype == torch.bfloat16, "out must be bf16"
        assert out.is_cuda and out.is_contiguous(), "out must be cuda + contiguous"

    a_major_mode = _operand_major_mode(layout_a)
    b_major_mode = _operand_major_mode(layout_b)
    a_ = from_dlpack(a, assumed_align=16)
    b_ = from_dlpack(b, assumed_align=16)
    d_ = from_dlpack(out, assumed_align=16)

    key = (
        a.dtype,
        tuple(a.shape),
        tuple(a.stride()),
        tuple(b.shape),
        tuple(b.stride()),
        layout_a,
        layout_b,
    )
    if key not in _compile_cache:
        _compile_cache[key] = cute.compile(
            mma_layout,
            a_,
            b_,
            d_,
            a_major_mode,
            b_major_mode,
        )
    _compile_cache[key](a_, b_, d_, a_major_mode, b_major_mode)
    return out
