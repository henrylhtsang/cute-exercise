"""Benchmark V0 (two launches) vs V1 (fused, grid barrier) global RMS-norm.

Both variants move the same DRAM bytes (read x twice + write out once),
so for large N they should land at the same effective bandwidth — the
fused kernel's win is the launch gap, which only matters when N is small.
This is the whole point of the exercise: a grid barrier removes the
launch overhead, but NOT the second DRAM read of x (that needs the V4
"keep the slab resident" trick).
"""

import torch

from cute_exercise.benchmark import bandwidth_gbps, benchmark
from cute_exercise.ex22_grid_sync.kernel import VARIANTS, ref_rmsnorm, rmsnorm_interface

# Note on CUDA graphs: the cooperative (grid_sync) kernel cannot be naively
# wrapped in a torch CUDA-graph capture — the capture hangs (the cooperative
# launch + grid barrier does not behave under stream capture). And capturing
# the two_launch variant gives bogus constant device times, because the DSL
# launches the kernel on a stream torch's capture isn't watching. So we time
# in eager mode. That's fine for the launch-gap question: both variants are a
# single Python->DSL dispatch and differ only in how many kernels the device
# runs (2 vs 1), so the eager delta isolates the inter-kernel launch gap.


def run(N, dtype=torch.float32):
    x = torch.randn(N, device="cuda", dtype=dtype)
    out = torch.empty_like(x)
    # DRAM traffic: read x in phase 1 + read x in phase 2 + write out.
    total_bytes = 3 * x.numel() * x.element_size()
    ref = ref_rmsnorm(x)

    results = {}
    for variant in VARIANTS:
        rmsnorm_interface(x, out=out, variant=variant)
        torch.testing.assert_close(out, ref, atol=2e-3, rtol=2e-3)
        us = benchmark(lambda v=variant: rmsnorm_interface(x, out=out, variant=v))
        results[variant] = (us, bandwidth_gbps(total_bytes, us))
    return results


if __name__ == "__main__":
    sizes = [64 * 1024, 256 * 1024, 1024 * 1024, 16 * 1024 * 1024, 64 * 1024 * 1024, 256 * 1024 * 1024]

    print("##### Eager: two_launch (V0) vs grid_sync (V1) #####")
    print("(both move identical DRAM bytes: read x twice + write out)\n")
    for N in sizes:
        label = f"{N // 1024}Ki" if N < 1024 * 1024 else f"{N // (1024 * 1024)}Mi"
        print(f"=== N = {label} (fp32) ===")
        res = run(N)
        for name, (us, gbps) in res.items():
            print(f"  {name:11s}: {us:8.3f} us   {gbps:8.1f} GB/s")
        tl, gs = res["two_launch"][0], res["grid_sync"][0]
        print(f"  fused saves {tl - gs:+.3f} us  ({100*(tl-gs)/tl:+.1f}%)\n")
