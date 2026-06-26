"""Global atomic-add contention microbenchmark.

The hot variant makes every thread repeatedly atomic-add to one global
counter. The striped variant keeps the same number of global atomics but
spreads them over many counters to reduce single-line serialization.
"""

import torch

import cutlass
import cutlass.cute as cute
from cutlass import Int32
from cutlass.cute.runtime import from_dlpack

from cute_exercise.base import CuteDSLKernel


VARIANTS = ("hot", "striped")


class GlobalAtomicAdd(CuteDSLKernel):
    def __init__(self, variant: str = "hot", threads: int = 256, iters: int = 1024):
        assert variant in VARIANTS, f"variant must be one of {VARIANTS}"
        self.variant = variant
        self.threads = threads
        self.iters = iters

    @cute.kernel
    def atomic_add_kernel(self, mCounters: cute.Tensor, counters: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        bdim, _, _ = cute.arch.block_dim()

        offset = Int32(0)
        if cutlass.const_expr(self.variant == "striped"):
            offset = (bidx * bdim + tidx) % counters

        ptr = mCounters.iterator + offset

        i = Int32(0)
        while i < self.iters:
            cute.arch.atomic_add(ptr.llvm_ptr, Int32(1), sem="relaxed", scope="gpu")
            i += 1

    @cute.jit
    def __call__(
        self,
        mCounters: cute.Tensor,
        counters: Int32,
        num_ctas: cutlass.Constexpr[int],
    ):
        self.atomic_add_kernel(mCounters, counters).launch(
            grid=[num_ctas, 1, 1],
            block=[self.threads, 1, 1],
        )


_compile_cache: dict = {}


def atomic_add_interface(
    *,
    num_ctas: int,
    threads: int = 256,
    iters: int = 1024,
    counters: int = 1,
    variant: str = "hot",
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the selected atomic-add microbenchmark and return counters."""
    assert variant in VARIANTS
    assert counters >= 1
    if variant == "hot":
        counters = 1

    if out is None:
        out = torch.zeros(counters, device="cuda", dtype=torch.int32)
    else:
        assert out.is_cuda and out.is_contiguous() and out.dtype == torch.int32
        assert out.numel() == counters
        out.zero_()

    out_ = from_dlpack(out)
    key = (variant, num_ctas, threads, iters, counters)
    if key not in _compile_cache:
        op = GlobalAtomicAdd(variant=variant, threads=threads, iters=iters)
        _compile_cache[key] = cute.compile(op, out_, Int32(counters), num_ctas)
    _compile_cache[key](out_, Int32(counters))
    return out
