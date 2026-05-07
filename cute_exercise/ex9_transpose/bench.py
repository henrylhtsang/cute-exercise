"""Benchmark the cute transpose kernel vs torch's a.T.contiguous().

Bytes moved per call = 2 * M * N * sizeof(fp32) (one read, one write per
element). Reports median ms and effective HBM bandwidth.
"""
import torch

from cute_exercise.ex9_transpose.kernel import transpose_interface


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

    shapes = [
        (4096, 4096),   # square
        (8192, 1024),   # tall
        (1024, 8192),   # wide
        (4096, 8192),   # rectangular
    ]

    print(f"{'shape':>14} {'kernel':>10} {'ms':>10} {'GB/s':>10} {'vs torch':>10}")
    for M, N in shapes:
        a = torch.randn(M, N, device="cuda", dtype=torch.float32)
        b = torch.empty(N, M, device="cuda", dtype=torch.float32)
        b_ref = torch.empty(N, M, device="cuda", dtype=torch.float32)

        # Reference: copy a.T into preallocated row-major buffer.
        torch.cuda.synchronize()
        torch_ms = bench(lambda: b_ref.copy_(a.T))

        # Cute kernel.
        try:
            transpose_interface(a, out=b)
            torch.cuda.synchronize()
            torch.testing.assert_close(b, a.T.contiguous())
            cute_ms = bench(lambda: transpose_interface(a, out=b))
            ok = True
        except Exception as ex:
            cute_ms = float("nan")
            ok = False
            err = str(ex).splitlines()[0]

        bytes_moved = 2 * M * N * 4

        def gbs(ms):
            return bytes_moved / (ms * 1e-3) / 1e9

        shape_str = f"{M}x{N}"
        print(f"{shape_str:>14} {'torch':>10} {torch_ms:>10.4f} "
              f"{gbs(torch_ms):>10.2f} {'1.00x':>10}")
        if ok:
            print(f"{'':>14} {'cute':>10} {cute_ms:>10.4f} "
                  f"{gbs(cute_ms):>10.2f} {cute_ms / torch_ms:>9.2f}x")
        else:
            print(f"{'':>14} {'cute':>10}    FAILED: {err[:60]}")


if __name__ == "__main__":
    main()
