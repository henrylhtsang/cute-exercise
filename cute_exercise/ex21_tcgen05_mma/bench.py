"""Benchmark the cute tcgen05 MMA kernel vs torch.matmul.

Reports median ms and effective TFLOP/s (2*M*N*K FLOPs per call).
"""
import torch

from cute_exercise.ex21_tcgen05_mma.kernel import (
    TILE_K,
    TILE_M,
    TILE_N,
    mma_interface,
)
from cute_exercise.ex21_tcgen05_mma.test_mma import make_inputs


def bench(fn, iters: int = 50, warmup: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


def main():
    torch.manual_seed(0)

    # Single-tile shapes plus a couple of multi-tile ones (tests scale once
    # the kernel handles a grid).
    shapes = [
        (TILE_M, TILE_N, TILE_K),
        (TILE_M, TILE_N, 4 * TILE_K),
        (4 * TILE_M, 4 * TILE_N, 4 * TILE_K),
        (8 * TILE_M, 8 * TILE_N, 8 * TILE_K),
    ]

    print(f"{'shape (MxNxK)':>18} {'kernel':>10} {'ms':>10} {'TFLOP/s':>10} {'vs torch':>10}")
    for M, N, K in shapes:
        a, b = make_inputs(M, N, K)
        d = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)

        # Reference: torch.matmul on the same operands.
        torch.cuda.synchronize()
        torch_ms = bench(lambda: torch.matmul(a, b, out=d))

        try:
            mma_interface(a, b, out=d)
            torch.cuda.synchronize()
            expected = (a.float() @ b.float()).to(torch.bfloat16)
            torch.testing.assert_close(d, expected, rtol=1e-2, atol=1e-2)
            cute_ms = bench(lambda: mma_interface(a, b, out=d))
            ok = True
        except Exception as ex:
            cute_ms = float("nan")
            ok = False
            err = str(ex).splitlines()[0]

        flops = 2 * M * N * K

        def tflops(ms):
            return flops / (ms * 1e-3) / 1e12

        shape_str = f"{M}x{N}x{K}"
        print(f"{shape_str:>18} {'torch':>10} {torch_ms:>10.4f} "
              f"{tflops(torch_ms):>10.2f} {'1.00x':>10}")
        if ok:
            print(f"{'':>18} {'cute':>10} {cute_ms:>10.4f} "
                  f"{tflops(cute_ms):>10.2f} {torch_ms / cute_ms:>9.2f}x")
        else:
            print(f"{'':>18} {'cute':>10}    FAILED: {err[:60]}")


if __name__ == "__main__":
    main()
