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
        dtype = mA.element_type

        thr_layout = cute.make_ordered_layout(self.thr_layout_shape, order=(1, 0))
        val_layout = cute.make_ordered_layout(
            (self.val_layout_rows, self.coalesced_ldst_bytes), order=(1, 0)
        )
        val_layout = cute.recast_layout(dtype.width, 8, val_layout)
        tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

        gA = cute.zipped_divide(mA, tiler_mn)
        gB = cute.zipped_divide(mB, tiler_mn)
        gC = cute.zipped_divide(mC, tiler_mn)

        remap_block = cute.make_ordered_layout(
            cute.select(gA.shape[1], mode=[1, 0]), order=(1, 0)
        )
        gA = cute.composition(gA, (None, remap_block))
        gB = cute.composition(gB, (None, remap_block))
        gC = cute.composition(gC, (None, remap_block))

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

        blk_coord = ((None, None), bidx)
        blkA = gA[blk_coord]
        blkB = gB[blk_coord]
        blkC = gC[blk_coord]

        tidfrgA = cute.composition(blkA, tv_layout)
        tidfrgB = cute.composition(blkB, tv_layout)
        tidfrgC = cute.composition(blkC, tv_layout)

        thr_coord = (tidx, None)
        thrA = tidfrgA[thr_coord]
        thrB = tidfrgB[thr_coord]
        thrC = tidfrgC[thr_coord]

        if cutlass.const_expr(self.variant == "vectorized"):
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
