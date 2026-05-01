"""``ElementwiseAdd`` as a class, for the "how bad is a for loop" exercise.

Same TV layout for both variants; they differ only in how the per-thread
fragment is consumed:

  variant='vectorized' — fragment-level ``.load()`` / assign to ``[None]``
                         (compiler emits 128-bit vector ld/st)
  variant='for_loop'   — explicit Python ``for`` over fragment elements

Tile sizing follows examples/python/CuTeDSL/notebooks/elementwise_add.ipynb:
4x64 threads per block, 16-byte coalesced ld/st per thread per row, 16 rows.
"""

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from cute_exercise.base import CuteDSLKernel


VARIANTS = ("vectorized", "for_loop", "vec_ld_scalar_add", "scalar_ld_vec_add")


class ElementwiseAdd(CuteDSLKernel):
    def __init__(
        self,
        variant: str = "vectorized",
        coalesced_ldst_bytes: int = 16,
        thr_layout_shape: tuple = (4, 64),
        val_layout_rows: int = 16,
    ):
        assert variant in VARIANTS, f"variant must be one of {VARIANTS}"
        self.variant = variant
        self.coalesced_ldst_bytes = coalesced_ldst_bytes
        self.thr_layout_shape = thr_layout_shape
        self.val_layout_rows = val_layout_rows

    @cute.jit
    def __call__(self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
        # Inputs are row-major (M, N) tensors with layout (M, N):(N, 1).
        dtype = mA.element_type

        # Thread layout: split the block's 256 threads over a 2-D tile.
        # With the default shape=(4, 64), order=(1, 0), mode 1 is stride-1:
        #   (4, 64):(64, 1)
        # so adjacent thread ids advance along columns before rows.
        thr_layout = cute.make_ordered_layout(self.thr_layout_shape, order=(1, 0))

        # Value layout: each thread owns a small 2-D fragment. Build it in
        # bytes first so the inner dimension is exactly the requested
        # coalesced load/store width. With defaults:
        #   (16, 16):(16, 1) bytes
        # meaning 16 rows and 16 contiguous bytes per row.
        val_layout = cute.make_ordered_layout(
            (self.val_layout_rows, self.coalesced_ldst_bytes), order=(1, 0)
        )
        # Convert that byte layout into an element layout for this dtype.
        # Examples with the default 16-byte inner dimension:
        #   fp16 -> (16, 8):(8, 1) elements
        #   fp32 -> (16, 4):(4, 1) elements
        val_layout = cute.recast_layout(dtype.width, 8, val_layout)

        # Combine thread ownership and per-thread values into a TV layout.
        # tiler_mn is the full block tile shape in elements:
        #   (thr_rows * val_rows, thr_cols * val_cols)
        # For fp16 defaults this is (4*16, 64*8) = (64, 512).
        # tv_layout maps (thread_id, value_id) to a coordinate inside that
        # tile, arranged so warp lanes touch consecutive 16-byte chunks.
        tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

        # Tile each global tensor. zipped_divide returns a tensor whose first
        # mode indexes inside one tile and whose second mode enumerates tiles:
        #   ((tile_M, tile_N), (num_tiles_M, num_tiles_N))
        # For fp16 defaults:
        #   ((64, 512), (M/64, N/512)) : ((N, 1), (64*N, 512))
        gA = cute.zipped_divide(mA, tiler_mn)
        gB = cute.zipped_divide(mB, tiler_mn)
        gC = cute.zipped_divide(mC, tiler_mn)

        # Remap the tile-grid mode so linear block ids visit N-tiles before
        # M-tiles. For row-major tensors this keeps consecutive blocks near
        # each other in memory. select(..., mode=[1, 0]) swaps the grid shape
        # before make_ordered_layout makes that swapped mode stride-1.
        # For fp16 defaults, remap_block is:
        #   (N/512, M/64):(M/64, 1)
        remap_block = cute.make_ordered_layout(
            cute.select(gA.shape[1], mode=[1, 0]), order=(1, 0)
        )
        # composition(A, B) builds the view A(B(x)): B defines the new
        # coordinate space, then A maps those coordinates to memory. In the
        # per-mode form below, None leaves the tile-internal mode unchanged
        # while remap_block relabels only the tile-grid mode.
        #
        # For fp16 defaults, the outer tile-grid mode changes from:
        #   (M/64, N/512):(64*N, 512)
        # to:
        #   (N/512, M/64):(512, 64*N)
        # so consecutive blockIdx values walk across N-tiles in a row before
        # moving to the next M-tile.
        gA = cute.composition(gA, (None, remap_block))
        gB = cute.composition(gB, (None, remap_block))
        gC = cute.composition(gC, (None, remap_block))

        # Launch one CUDA block per output tile. The block size is the number
        # of threads represented by the T mode of tv_layout.
        self.kernel(gA, gB, gC, tv_layout).launch(
            grid=[cute.size(gC, mode=[1]), 1, 1],
            block=[cute.size(tv_layout, mode=[0]), 1, 1],
        )

    @cute.kernel
    def kernel(
        self,
        gA: cute.Tensor,
        gB: cute.Tensor,
        gC: cute.Tensor,
        tv_layout: cute.Layout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        # Select this block's full tile. The leading (None, None) keeps all
        # coordinates inside the tile; bidx selects one tile-grid coordinate.
        # blkA/blkB/blkC are now per-block (tile_M, tile_N) views.
        blk_coord = ((None, None), bidx)
        blkA = gA[blk_coord]
        blkB = gB[blk_coord]
        blkC = gC[blk_coord]

        # Re-layout the tile as (Thread, Value). After composition, mode 0
        # selects a thread and mode 1 selects a value from that thread's
        # fragment. With fp16 defaults, each thread has 16*8 values.
        tidfrgA = cute.composition(blkA, tv_layout)
        tidfrgB = cute.composition(blkB, tv_layout)
        tidfrgC = cute.composition(blkC, tv_layout)

        # Slice the T mode at this thread id and keep the full V mode. The
        # resulting thr* tensors are this thread's fragment views; for fp16
        # defaults, each is a (16, 8) fragment.
        thr_coord = (tidx, None)
        # this is the value layout for the thread (v0..v7) -> actual values
        thrA = tidfrgA[thr_coord]
        thrB = tidfrgB[thr_coord]
        thrC = tidfrgC[thr_coord]

        if cutlass.const_expr(self.variant == "vectorized"):
            # Load and store whole fragments. For fp16 defaults, each 8-value
            # row is 16 bytes, so the compiler can use a 128-bit vector
            # memory instruction per row of the fragment.
            thrC[None] = thrA.load() + thrB.load()
        elif cutlass.const_expr(self.variant == "vec_ld_scalar_add"):
            rA = thrA.load()
            rB = thrB.load()
            rC = cute.make_fragment_like(thrC)
            for i in cutlass.range_constexpr(cute.size(rC)):
                rC[i] = rA[i] + rB[i]
            thrC[None] = rC.load()
        elif cutlass.const_expr(self.variant == "scalar_ld_vec_add"):
            rA = cute.make_fragment_like(thrA)
            rB = cute.make_fragment_like(thrB)
            for i in cutlass.range_constexpr(cute.size(thrA)):
                rA[i] = thrA[i]
                rB[i] = thrB[i]
            thrC[None] = rA.load() + rB.load()
        else:
            # Scalar indexing consumes the same fragment shape one element at
            # a time. That preserves the TV layout but loses the 16-byte vector
            # memory operation, giving many more load/store instructions.
            for i in cutlass.range_constexpr(cute.size(thrA)):
                thrC[i] = thrA[i] + thrB[i]


_compile_cache: dict = {}


def elementwise_add_interface(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor | None = None,
    variant: str = "vectorized",
) -> torch.Tensor:
    """Torch-facing wrapper: compute ``out = a + b`` via ``ElementwiseAdd``.

    Caches compiled kernels per (dtype, variant). Returns ``out``.
    """
    assert variant in VARIANTS, f"variant must be one of {VARIANTS}"
    assert a.shape == b.shape
    assert a.dtype == b.dtype
    assert a.is_cuda and b.is_cuda
    assert a.is_contiguous() and b.is_contiguous()

    if out is None:
        out = torch.empty_like(a)
    else:
        assert out.shape == a.shape
        assert out.dtype == a.dtype
        assert out.is_cuda and out.is_contiguous()

    a_ = from_dlpack(a, assumed_align=16)
    b_ = from_dlpack(b, assumed_align=16)
    c_ = from_dlpack(out, assumed_align=16)

    key = (a.dtype, variant)
    if key not in _compile_cache:
        op = ElementwiseAdd(variant=variant)
        _compile_cache[key] = cute.compile(op, a_, b_, c_)
    _compile_cache[key](a_, b_, c_)

    return out
