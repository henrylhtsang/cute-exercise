"""Benchmark the ElementwiseAdd variants against a torch.add baseline."""

import torch

from cute_exercise.benchmark import bandwidth_gbps, benchmark
from cute_exercise.for_loop.kernel import VARIANTS, elementwise_add_interface


def run_variants(M=16384, N=8192, dtype=torch.float16):
    a = torch.randn(M, N, device="cuda", dtype=dtype)
    b = torch.randn(M, N, device="cuda", dtype=dtype)
    c = torch.zeros(M, N, device="cuda", dtype=dtype)

    total_bytes = (a.numel() + b.numel() + c.numel()) * a.element_size()

    results = {}

    torch.add(a, b, out=c)
    torch.testing.assert_close(c, a + b)
    us = benchmark(lambda: torch.add(a, b, out=c))
    results["torch.add"] = (us, bandwidth_gbps(total_bytes, us))

    for variant in VARIANTS:
        elementwise_add_interface(a, b, out=c, variant=variant)
        torch.testing.assert_close(c, a + b)
        us = benchmark(lambda v=variant: elementwise_add_interface(a, b, out=c, variant=v))
        results[variant] = (us, bandwidth_gbps(total_bytes, us))

    return results


if __name__ == "__main__":
    for dtype_name, dtype in [("fp16", torch.float16), ("fp32", torch.float32)]:
        print(f"\n=== {dtype_name} (M=16384, N=8192) ===")
        for name, (us, gbps) in run_variants(dtype=dtype).items():
            print(f"  {name:12s}: {us:8.3f} us   {gbps:8.1f} GB/s")
