"""Grid-wide barrier ("grid.sync") fusion exercise.

Operation: global RMS-normalize a 1-D tensor.

    out[i] = x[i] * rsqrt(mean(x**2) + eps)        mean over ALL N elements

This genuinely needs a two-pass structure with a grid-wide dependency:
every CTA must finish contributing to the global sum-of-squares before
*any* CTA can normalize. We compare:

  variant="two_launch"  — V0: two separate (non-cooperative) kernel
                          launches, reduce then normalize. The obvious
                          correct baseline; pays a launch gap + reads x
                          from DRAM twice.
  variant="grid_sync"   — V1: ONE persistent cooperative kernel. Phase 1
                          reduces, then a hand-rolled grid-wide barrier
                          (arrival-counter + spin, the same primitives as
                          cooperative_groups::this_grid().sync()), then
                          phase 2 normalizes. No second launch.

CuTe DSL exposes the *launch* side (`cooperative=True` on `.launch`) but
NOT a device-side grid barrier, so we build it from inline PTX
(`red.release.gpu` to arrive, `ld.global.acquire.gpu` to spin), exactly
like flash-attention's `flash_attn/cute/barrier.py`.
"""

import torch

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

from cute_exercise.base import CuteDSLKernel


VARIANTS = ("two_launch", "grid_sync")

WARP_SIZE = 32


# --------------------------------------------------------------------------
# Grid-barrier primitive: acquire-load of a GMEM flag (adapted from
# flash_attn/cute/barrier.py). atomic_add(..., sem="release") handles the
# arrive side; this handles the spin side.
# --------------------------------------------------------------------------
@dsl_user_op
def ld_acquire_gpu(ptr: cute.Pointer, *, loc=None, ip=None) -> Int32:
    ptr_i64 = ptr.toint(loc=loc, ip=ip).ir_value()
    state = llvm.inline_asm(
        T.i32(),
        [ptr_i64],
        "ld.global.acquire.gpu.b32 $0, [$1];",
        "=r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(state)


@cute.jit
def warp_reduce_add(val: Float32) -> Float32:
    # Butterfly all-reduce within a warp.
    for i in cutlass.range_constexpr(5):  # log2(32)
        val = val + cute.arch.shuffle_sync_bfly(val, offset=1 << i)
    return val


@cute.jit
def block_partial_sumsq(mX: cute.Tensor, N: Int32, sumsq_ptr: cute.Pointer) -> None:
    """Grid-stride accumulate x**2, warp-reduce, one atomic per warp into
    the global sum-of-squares accumulator."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    gdim, _, _ = cute.arch.grid_dim()

    stride = gdim * bdim
    i = bidx * bdim + tidx
    partial = Float32(0.0)
    while i < N:
        v = mX[i].to(Float32)
        partial = partial + v * v
        i += stride

    partial = warp_reduce_add(partial)
    if cute.arch.lane_idx() == 0:
        cute.arch.atomic_add(sumsq_ptr.llvm_ptr, partial, sem="relaxed", scope="gpu")


@cute.jit
def normalize(mX: cute.Tensor, mOut: cute.Tensor, N: Int32, inv: Float32) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    gdim, _, _ = cute.arch.grid_dim()

    stride = gdim * bdim
    i = bidx * bdim + tidx
    while i < N:
        mOut[i] = (mX[i].to(Float32) * inv).to(mOut.element_type)
        i += stride


class GridSyncRMSNorm(CuteDSLKernel):
    def __init__(self, variant: str = "grid_sync", threads: int = 256, eps: float = 1e-6):
        assert variant in VARIANTS, f"variant must be one of {VARIANTS}"
        self.variant = variant
        self.threads = threads
        self.eps = eps

    # ---- V1: single cooperative kernel with a grid barrier ----------------
    @cute.kernel
    def fused_kernel(
        self,
        mX: cute.Tensor,
        mOut: cute.Tensor,
        mSumsq: cute.Tensor,   # (1,) f32, zeroed by host
        mBar: cute.Tensor,     # (1,) i32, zeroed by host
        N: Int32,
        eps: Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        gdim, _, _ = cute.arch.grid_dim()
        sumsq_ptr = mSumsq.iterator
        bar_ptr = mBar.iterator

        # ---- Phase 1: partial reduction into the global accumulator ----
        block_partial_sumsq(mX, N, sumsq_ptr)

        # ---- Grid-wide barrier -----------------------------------------
        # All warps in this block have done their atomic; make those GMEM
        # writes visible grid-wide before we announce arrival.
        cute.arch.sync_threads()
        cute.arch.fence_acq_rel_gpu()
        if tidx == 0:
            # arrive (release) then spin until every CTA has arrived
            cute.arch.atomic_add(bar_ptr.llvm_ptr, Int32(1), sem="release", scope="gpu")
            while ld_acquire_gpu(bar_ptr) != gdim:
                pass
        cute.arch.sync_threads()

        # ---- Phase 2: normalize using the now-global statistic ----------
        sumsq = mSumsq[0]
        inv = cute.math.rsqrt(sumsq / Float32(N) + eps, fastmath=True)
        normalize(mX, mOut, N, inv)

    # ---- V0: two plain (non-cooperative) kernels --------------------------
    @cute.kernel
    def reduce_kernel(self, mX: cute.Tensor, mSumsq: cute.Tensor, N: Int32):
        block_partial_sumsq(mX, N, mSumsq.iterator)

    @cute.kernel
    def normalize_kernel(
        self, mX: cute.Tensor, mOut: cute.Tensor, mSumsq: cute.Tensor, N: Int32, eps: Float32
    ):
        sumsq = mSumsq[0]
        inv = cute.math.rsqrt(sumsq / Float32(N) + eps, fastmath=True)
        normalize(mX, mOut, N, inv)

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mOut: cute.Tensor,
        mSumsq: cute.Tensor,
        mBar: cute.Tensor,
        num_ctas: cutlass.Constexpr[int],
    ):
        N = Int32(cute.size(mX))
        eps = Float32(self.eps)
        block = [self.threads, 1, 1]

        if cutlass.const_expr(self.variant == "grid_sync"):
            self.fused_kernel(mX, mOut, mSumsq, mBar, N, eps).launch(
                grid=[num_ctas, 1, 1], block=block, cooperative=True
            )
        else:
            # V0: persistent-shaped but two ordinary launches on the stream.
            self.reduce_kernel(mX, mSumsq, N).launch(grid=[num_ctas, 1, 1], block=block)
            self.normalize_kernel(mX, mOut, mSumsq, N, eps).launch(
                grid=[num_ctas, 1, 1], block=block
            )


_compile_cache: dict = {}


def _num_persistent_ctas(cta_per_sm: int = 1) -> int:
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return props.multi_processor_count * cta_per_sm


def ref_rmsnorm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Reference: out = x * rsqrt(mean(x**2) + eps), mean over all elements."""
    x = x.reshape(-1).float()
    return (x * torch.rsqrt((x * x).mean() + eps)).to(x.dtype)


def rmsnorm_interface(
    x: torch.Tensor,
    out: torch.Tensor | None = None,
    variant: str = "grid_sync",
    eps: float = 1e-6,
    cta_per_sm: int = 1,
    threads: int = 256,
) -> torch.Tensor:
    """out = x * rsqrt(mean(x**2) + eps), mean over all elements."""
    assert variant in VARIANTS
    assert x.is_cuda and x.is_contiguous()
    x = x.reshape(-1)
    if out is None:
        out = torch.empty_like(x)
    else:
        out = out.reshape(-1)
        assert out.shape == x.shape and out.dtype == x.dtype

    num_ctas = _num_persistent_ctas(cta_per_sm)

    sumsq = torch.zeros(1, device=x.device, dtype=torch.float32)
    barrier = torch.zeros(1, device=x.device, dtype=torch.int32)

    x_ = from_dlpack(x, assumed_align=16)
    out_ = from_dlpack(out, assumed_align=16)
    sumsq_ = from_dlpack(sumsq)
    bar_ = from_dlpack(barrier)

    # x.numel() is part of the key: from_dlpack bakes the tensor shape
    # statically, so a kernel compiled for one N can't be reused for another.
    key = (x.dtype, variant, x.numel(), num_ctas, threads, eps)
    if key not in _compile_cache:
        op = GridSyncRMSNorm(variant=variant, threads=threads, eps=eps)
        _compile_cache[key] = cute.compile(op, x_, out_, sumsq_, bar_, num_ctas)
    _compile_cache[key](x_, out_, sumsq_, bar_)

    return out
