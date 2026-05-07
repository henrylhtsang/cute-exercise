"""Single-tile BF16 MMA test. Compares D = A @ B against torch.matmul."""
import torch

from cute_exercise.ex21_tcgen05_mma.kernel import (
    TILE_K,
    TILE_M,
    TILE_N,
    mma_interface,
)


def main():
    torch.manual_seed(0)
    M, N, K = TILE_M, TILE_N, TILE_K
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    d = torch.empty(M, N, device="cuda", dtype=torch.float32)

    mma_interface(a, b, out=d)
    torch.cuda.synchronize()

    expected = (a.float() @ b.float())
    print(f"a ({M}x{K}):\n{a}")
    print(f"b ({K}x{N}):\n{b}")
    print(f"d ({M}x{N}):\n{d}")
    print(f"expected:\n{expected}")

    # BF16 inputs accumulated in FP32 — tolerance scales with K.
    torch.testing.assert_close(d, expected, rtol=1e-2, atol=1e-2)
    print("PASS")


if __name__ == "__main__":
    main()
