"""Out-of-place 2D transpose: B[i, j] = A[j, i] for an M x N row-major tensor.

This file is the host-side scaffolding (dtype/shape checks, from_dlpack
conversion, compile cache). Fill in the @cute.jit and @cute.kernel
bodies below.
"""
import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import SmemAllocator
from cutlass.cute.nvgpu import cpasync, tcgen05


@cute.jit
def transpose(A: cute.Tensor, B: cute.Tensor):
    tiler = (32, 32)

    # gA = cute.zipped_divide(A, tiler)
    # gB = cute.zipped_divide(B, tiler)

    # S = cute.make_ordered_layout(shape=gA.shape[1], order=(1, 0))
    # gB = cute.composition(gB, (None, S))

    smem_layout = cute.make_layout((32, 32), stride=(32, 1))
    tma_tiler = (32, 32)

    tma_load_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
    tma_atom_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
        tma_load_op,
        A,
        smem_layout,
        tma_tiler
    )

    tma_store_op = cpasync.CopyBulkTensorTileS2GOp()
    tma_atom_b, tma_tensor_b = cpasync.make_tiled_tma_atom(
        tma_store_op,
        B,
        smem_layout,
        tma_tiler,
    )

    kernel(tma_tensor_a, tma_tensor_b, tma_atom_a, tma_atom_b, smem_layout).launch(
        grid=(
            cute.ceil_div(cute.size(A.shape[0]), tma_tiler[0]),
            cute.ceil_div(cute.size(A.shape[1]), tma_tiler[1]),
            1,
        ),
        block=(32, 1, 1),
    )


@cute.kernel
def kernel(tma_tensor_a, tma_tensor_b, tma_atom_a, tma_atom_b, smem_layout):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    smem = SmemAllocator()
    sA = smem.allocate_tensor(cutlass.Float32, smem_layout)

    # blk_coord = ((None, None), bidx)
    # blkA = gA[blk_coord]
    # blkB = gB[blk_coord]

    blk_coord = (bidx, bidy)
    gA_tile = cute.local_tile(tma_tensor_a, (32, 32), blk_coord)
    gB_tile = cute.local_tile(tma_tensor_b, (32, 32), blk_coord)

    tAsA, tAgA = cpasync.tma_partition(
        tma_atom_a,
        0,
        cute.make_layout(1),
        cute.group_modes(sA, 0, 2),
        cute.group_modes(gA_tile, 0, 2),
    )

    tBsB, tBgB = cpasync.tma_partition(
        tma_atom_b,
        0,
        cute.make_layout(1),
        cute.group_modes(sA, 0, 2),
        cute.group_modes(gB_tile, 0, 2),
    )

    with cute.arch.elect_one():
        cute.copy(
            tma_atom_a,
            tAgA,
            tAsA,
        )

    cute.arch.cp_async_bulk_commit_group()
    cute.arch.cp_async_bulk_wait_group(0, read=True)
    cute.arch.sync_threads()

    with cute.arch.elect_one():
        cute.copy(
            tma_atom_b, tBsB, tBgB,
        )

    cute.arch.cp_async_bulk_commit_group()
    cute.arch.cp_async_bulk_wait_group(0, read=False)
    # cute.arch.sync_threads()

    # for i in cutlass.range_constexpr(32):
    #     sA[i, tidx] = blkA[i, tidx]

    # cute.arch.sync_threads()

    # for i in cutlass.range_constexpr(32):
    #     blkB[i, tidx] = sA[tidx, i]


_compile_cache: dict = {}


def transpose_interface(
    a: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """Compute B = A.T out-of-place via the cute transpose kernel.

    Constraints (host-side, easy to relax later):
    - fp32, CUDA, contiguous (row-major).
    - 2-D input.
    - Both dimensions divisible by 32 (matches the 32x32 tile).
    """
    assert a.is_cuda and a.is_contiguous(), "a must be cuda + contiguous"
    assert a.dim() == 2, f"expected 2-D tensor, got {a.dim()}-D"
    assert a.dtype == torch.float32, f"expected fp32, got {a.dtype}"
    M, N = a.shape
    assert M % 32 == 0 and N % 32 == 0, f"dims must be divisible by 32, got {M}x{N}"

    if out is None:
        out = torch.empty((N, M), device=a.device, dtype=a.dtype)
    else:
        assert out.shape == (N, M), f"out must be {(N, M)}, got {tuple(out.shape)}"
        assert out.dtype == a.dtype, "out dtype mismatch"
        assert out.is_cuda and out.is_contiguous(), "out must be cuda + contiguous"

    a_ = from_dlpack(a, assumed_align=16)
    b_ = from_dlpack(out, assumed_align=16)

    # Cache compiles per (dtype, shape). Shape matters because the
    # kernel will likely bake tile counts as constants.
    key = (a.dtype, a.shape)
    if key not in _compile_cache:
        _compile_cache[key] = cute.compile(transpose, a_, b_)
    _compile_cache[key](a_, b_)

    return out
