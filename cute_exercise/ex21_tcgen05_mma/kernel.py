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
from cutlass.utils import SmemAllocator, TmemAllocator
from cutlass.utils import sm100 as sm100_utils

from cutlass.utils.layout import LayoutEnum
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.pipeline as pipeline


# Single-tile config. Pick a shape that one `tcgen05.mma cta_group::1`
# can deliver. (M, N, K) = (128, 128, 64) is a safe starting point for BF16.
TILE_M = 128
TILE_N = 128
TILE_K = 64


@cute.jit
def mma(A: cute.Tensor, B: cute.Tensor, D: cute.Tensor):
    # TODO: tile A, B, D; build TMA atoms for A/B; allocate TMEM for the
    # accumulator; launch the inner kernel.
    tile_m = 128
    tile_n = 128
    tile_k = 64
    tiler = (tile_m, tile_n, tile_k)

    M, N = D.shape

    block = (128, 1, 1)
    grid = (cute.ceil_div(M, tile_m), cute.ceil_div(N, tile_n), 1)


    # alloc tmem
    tiled_mma = sm100_utils.make_trivial_tiled_mma(
        ab_dtype=cutlass.BFloat16,
        a_leading_mode=tcgen05.OperandMajorMode.K,
        b_leading_mode=tcgen05.OperandMajorMode.K,
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
    epi_tile_mn = (tile_m, 32)
    smem_layout_d = sm100_utils.make_smem_layout_epi(
        cutlass.BFloat16,
        LayoutEnum.ROW_MAJOR,
        epi_tile_mn,
        1,                    # epi_stage
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
        epi_tile_mn,
    )

    kernel(tma_tensor_a, tma_tensor_b, tma_tensor_d, tma_atom_a, tma_atom_b, tma_atom_d, tiled_mma, smem_layout_a, smem_layout_b, smem_layout_d, tiler, epi_tile_mn,).launch(
        grid=grid,
        block=block,
    )

@cute.kernel
def kernel(tma_tensor_a, tma_tensor_b, tma_tensor_d, tma_atom_a, tma_atom_b, tma_atom_d, tiled_mma, smem_layout_a, smem_layout_b, smem_layout_d, tiler: cutlass.Constexpr, epi_tile_mn: cutlass.Constexpr):
    # TODO: load A/B tile into SMEM (cp.async first, then TMA); allocate
    # TMEM; issue tcgen05.mma; wait; copy TMEM -> RMEM -> BF16 -> GMEM.
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    TILE_M, TILE_N, TILE_K = tiler

    smem = SmemAllocator()
    sA_tile = smem.allocate_tensor(element_type=cutlass.BFloat16, layout=smem_layout_a.outer, swizzle=smem_layout_a.inner)
    sB_tile = smem.allocate_tensor(element_type=cutlass.BFloat16, layout=smem_layout_b.outer, swizzle=smem_layout_b.inner)

    sD_tile = smem.allocate_tensor(cutlass.BFloat16, smem_layout_d.outer, swizzle=smem_layout_d.inner)

    sA_tile = sA_tile[None, None, None, 0]
    sB_tile = sB_tile[None, None, None, 0]

    tmem_holding_buf = smem.allocate(cutlass.Int32, byte_alignment=4)
    tmem_alloc_barrier = pipeline.NamedBarrier(
        barrier_id=1,
        num_threads=128,
    )

    mbar_load_a = smem.allocate_tensor(cutlass.Int64, cute.make_layout(1), byte_alignment=8)
    mbar_load_b = smem.allocate_tensor(cutlass.Int64, cute.make_layout(1), byte_alignment=8)
    mbar_mma = smem.allocate_tensor(cutlass.Int64, cute.make_layout(1), byte_alignment=8)
    if tidx == 0:
        cute.arch.mbarrier_init(mbar_load_a.iterator, 1)
        cute.arch.mbarrier_init(mbar_load_b.iterator, 1)
        cute.arch.mbarrier_init(mbar_mma.iterator, 1)

    gA_tile = cute.local_tile(tma_tensor_a, tiler, (bidx, bidy, 0), proj=(1, None, 1))
    gB_tile = cute.local_tile(tma_tensor_b, tiler, (bidx, bidy, 0), proj=(None, 1, 1))

    thr_mma = tiled_mma.get_slice(0)
    tCgA = thr_mma.partition_A(gA_tile)
    tCgB = thr_mma.partition_B(gB_tile)

    gD_tile = cute.local_tile(tma_tensor_d, tiler, (bidx, bidy, 0), proj=(1, 1, None))

    tAsA, tAgA = cpasync.tma_partition(
        tma_atom_a,
        0,
        cute.make_layout(1),
        cute.group_modes(sA_tile, 0, 3),
        cute.group_modes(tCgA, 0, 3),
    )
    tBsB, tBgB = cpasync.tma_partition(
        tma_atom_b,
        0,
        cute.make_layout(1),
        cute.group_modes(sB_tile, 0, 3),
        cute.group_modes(tCgB, 0, 3),
    )
    gD_epi = cute.flat_divide(gD_tile, epi_tile_mn)
    sD_for_tma = sD_tile[None, None, 0]
    tDsD, tDgD = cpasync.tma_partition(
        tma_atom_d,
        0,
        cute.make_layout(1),
        cute.group_modes(sD_for_tma, 0, 2),
        cute.group_modes(gD_epi, 0, 2),
    )


    # fence
    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()

    if tidx == 0:
        cute.arch.mbarrier_arrive_and_expect_tx(
            mbar_load_a.iterator, cute.size_in_bytes(cutlass.BFloat16, smem_layout_a),
        )
        cute.arch.mbarrier_arrive_and_expect_tx(
            mbar_load_b.iterator, cute.size_in_bytes(cutlass.BFloat16, smem_layout_b),
        )

    if cute.arch.warp_idx() == 0:
        cute.copy(
            tma_atom_a,
            tAgA,
            tAsA,
            tma_bar_ptr=mbar_load_a.iterator,
        )
        cute.copy(
            tma_atom_b,
            tBgB,
            tBsB,
            tma_bar_ptr=mbar_load_b.iterator,
        )

    cute.arch.mbarrier_wait(mbar_load_a.iterator, phase=0)
    cute.arch.mbarrier_wait(mbar_load_b.iterator, phase=0)


    tmem = TmemAllocator(
        tmem_holding_buf,
        barrier_for_retrieve=tmem_alloc_barrier,
        allocator_warp_id=0,
    )
    tmem.allocate(128)
    tmem.relinquish_alloc_permit()
    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)

    acc_shape = tiled_mma.partition_shape_C((TILE_M, TILE_N))
    tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)
    tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

    thr_mma = tiled_mma.get_slice(0)
    tCrA = thr_mma.make_fragment_A(sA_tile)
    tCrB = thr_mma.make_fragment_B(sB_tile)

    # todo: make this loop over k
    cute.gemm(
        tiled_mma,
        tCtAcc,  # D (acc out)
        tCrA,    # A
        tCrB,    # B
        tCtAcc,  # C (acc in)
    )

    if cute.arch.warp_idx() == 0:
        with cute.arch.elect_one():
            cute.nvgpu.tcgen05.commit(mbar_mma.iterator)

    cute.arch.mbarrier_wait(mbar_mma.iterator, phase=0)

    copy_atom_t2r = sm100_utils.get_tmem_load_op(
        cta_tile_shape=(TILE_M, TILE_N, TILE_K),
        layout_d=LayoutEnum.ROW_MAJOR,
        elem_ty_d=cutlass.BFloat16,
        elem_ty_acc=cutlass.Float32,
        epi_tile=epi_tile_mn,
        use_2cta_instrs=False,
    )
    tCtAcc_epi = cute.flat_divide(
        tCtAcc[((None, None), 0, 0)], epi_tile_mn
    )
    tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc_epi[None, None, 0, 0])
    thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)

    tTR_tAcc = thr_copy_t2r.partition_S(tCtAcc_epi)
    tTR_cAcc = thr_copy_t2r.partition_D(
        cute.make_identity_tensor(tCtAcc_epi.shape)
    )
    tTR_rAcc = cute.make_rmem_tensor(
        tTR_cAcc[None, None, None, 0, 0].shape, cutlass.Float32
    )
    tTR_rAcc_bf16 = cute.make_rmem_tensor(tTR_rAcc.shape, cutlass.BFloat16)

    copy_atom_r2s = sm100_utils.get_smem_store_op(
        layout_d=LayoutEnum.ROW_MAJOR,
        elem_ty_d=cutlass.BFloat16,
        elem_ty_acc=cutlass.Float32,
        tiled_tmem_load=tiled_copy_t2r,
    )
    tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
    thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
    tRS_sD = thr_copy_r2s.partition_D(sD_tile)
    tRS_rD = tiled_copy_r2s.retile(tTR_rAcc_bf16)

    tTR_tAcc_i = tTR_tAcc[None, None, None, 0, 0]
    cute.copy(tiled_copy_t2r, tTR_tAcc_i, tTR_rAcc)
    tTR_rAcc_bf16.store(tTR_rAcc.load().to(cutlass.BFloat16))
    cute.copy(tiled_copy_r2s, tRS_rD, tRS_sD[None, None, None, 0])

    cute.arch.fence_proxy("async.shared", space="cta")
    cute.arch.sync_threads()
    if cute.arch.warp_idx() == 0:
        cute.copy(
            tma_atom_d,
            tDsD,
            tDgD[None, 0, 0],
        )
        cute.arch.cp_async_bulk_commit_group()
    cute.arch.cp_async_bulk_wait_group(0)
    cute.arch.sync_threads()

    if cute.arch.warp_idx() == 0:
        tmem.free(tmem_ptr, 128)

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
