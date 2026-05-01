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
        # mA, mB, mC: row-major (M, N) tensors with layout (M, N):(N, 1).
        dtype = mA.element_type

        # Thread layout — how the block's threads tile a 2-D patch.
        # shape=(4, 64), order=(1, 0) ⇒ mode 1 has the smallest stride,
        # so the layout is (4, 64):(64, 1): 4 rows × 64 threads-per-row,
        # threads contiguous along the inner (column) dim. 256 threads/block.
        thr_layout = cute.make_ordered_layout(self.thr_layout_shape, order=(1, 0))

        # Value layout — what each thread owns inside one tile, expressed in
        # BYTES first so the inner dim is exactly the 16 B coalesced ld/st
        # width. shape=(16, 16):(16, 1) bytes ⇒ each thread owns 16 rows ×
        # 16 contiguous bytes per row.
        val_layout = cute.make_ordered_layout(
            (self.val_layout_rows, self.coalesced_ldst_bytes), order=(1, 0)
        )
        # recast_layout(new_bits, old_bits, layout): reinterpret the same
        # memory as elements of new_bits instead of old_bits (here 8 = byte).
        # fp16: (16, 16):(16, 1) bytes → (16, 8):(8, 1) elts; fp32 → (16, 4):(4, 1).
        val_layout = cute.recast_layout(dtype.width, 8, val_layout)

        # Build the (Thread, Value) layout used to partition each tile.
        # tiler_mn   = per-block tile shape in elements
        #              = (thr_rows * val_rows, thr_cols * val_cols)
        #              = (4*16, 64*8) = (64, 512) for fp16.
        # tv_layout  : (thread_id, val_id) → coord inside tiler_mn,
        #              arranged so the 32 threads of a warp hit 32
        #              consecutive 16 B chunks (one 128 B sector × 4).
        tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

        # Tile each global tensor by tiler_mn. zipped_divide produces a
        # tensor of shape ((tile_M, tile_N), (M/tile_M, N/tile_N)):
        # mode 0 indexes inside one tile, mode 1 enumerates the tile grid.
        # gA layout: ((64, 512), (M/64, N/512)) : ((N, 1), (64*N, 512))  [fp16].
        gA = cute.zipped_divide(mA, tiler_mn)
        gB = cute.zipped_divide(mB, tiler_mn)
        gC = cute.zipped_divide(mC, tiler_mn)

        # Reorder the tile-grid axis so blockIdx walks columns-first across
        # the grid (matches row-major memory ⇒ better L2 locality across
        # consecutive blocks). select(..., mode=[1, 0]) pulls the grid
        # shape (M/tile_M, N/tile_N) and the ordered layout makes the
        # column dim stride-1 in block-id space.
        # remap_block layout: (N/512, M/64) : (M/64, 1)  [fp16].
        remap_block = cute.make_ordered_layout(
            cute.select(gA.shape[1], mode=[1, 0]), order=(1, 0)
        )
        # composition(A, B) = "A ∘ B": for any coord x in B's domain,
        #   (A ∘ B)(x) = A(B(x)). B reinterprets/relabels the input space,
        #   then A maps that to the original memory offset. Shape comes
        #   from B; strides come from A. Use cases: axis swap (here),
        #   tile swizzles, sub-tiling, TV partitioning.
        # Per-mode form: composition(gA, (None, remap_block)) means
        #   "leave mode 0 alone (identity), apply remap_block to mode 1".
        # Net effect here (transpose-only case):
        #   outer was (M/64, N/512):(64*N, 512)
        #   outer now (N/512, M/64):(512,  64*N)   — axes swapped.
        # gA full layout: ((64, 512), (N/512, M/64)) : ((N, 1), (512, 64*N))
        # → consecutive blockIdx walks N-tiles within a row, then M-tiles.
        gA = cute.composition(gA, (None, remap_block))
        gB = cute.composition(gB, (None, remap_block))
        gC = cute.composition(gC, (None, remap_block))

        # 1 block per tile, threads per block = size of the T mode of tv.
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

        # Pick this block's tile out of the zipped-divided tensor.
        # gA shape is ((tile_M, tile_N), (#tiles_M, #tiles_N)); the
        # inner (None, None) keeps the full tile, bidx selects the tile.
        # Result: blkA has the (tile_M, tile_N) per-block view, e.g.
        # (64, 512) for fp16.
        blk_coord = ((None, None), bidx)
        blkA = gA[blk_coord]
        blkB = gB[blk_coord]
        blkC = gC[blk_coord]

        # Re-layout the tile from (tile_M, tile_N) → (Thread, Value) using
        # tv_layout. After composition, mode 0 is thread_id (size 256) and
        # mode 1 is val_id (size 16*8=128 fp16 elements per thread).
        tidfrgA = cute.composition(blkA, tv_layout)
        tidfrgB = cute.composition(blkB, tv_layout)
        tidfrgC = cute.composition(blkC, tv_layout)

        # Slice the T mode at this thread's tidx, keep the full V mode.
        # thrA is this thread's fragment view: a (16, 8) sub-tensor for
        # fp16 (16 rows, 8 contiguous fp16 elements per row), with strides
        # inherited from the tile's row-major layout.
        thr_coord = (tidx, None)
        # this is the value layout of that thread
        thrA = tidfrgA[thr_coord]
        thrB = tidfrgB[thr_coord]
        thrC = tidfrgC[thr_coord]

        if cutlass.const_expr(self.variant == "vectorized"):
            # .load() reads the whole fragment as one register tile; the
            # inner row of 8 fp16 (=16 B) collapses into a single 128-bit
            # ld.global.v4.b32. 32 lanes of a warp issue together → 4
            # consecutive 128 B sectors per instruction (fully coalesced),
            # × 16 row-instructions to drain the fragment.
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
            # Per-element indexing forces the compiler to emit a separate
            # ld/st per scalar. Same TV layout, same coalescing pattern
            # at any single instant, but the wide 128-bit vector op is
            # lost — 8× more instructions per row.
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
