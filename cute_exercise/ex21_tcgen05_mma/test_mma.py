"""Single-tile BF16 MMA test. Compares D = A @ B against torch.matmul.

Layouts:
  A: (M, K) BF16 row-major.
  B: (K, N) BF16 column-major.
  D: (M, N) BF16 row-major.
"""
import torch

from cute_exercise.ex21_tcgen05_mma.kernel import (
    TILE_K,
    TILE_M,
    TILE_N,
    mma_interface,
)


def make_inputs(M: int, N: int, K: int):
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    # Column-major (K, N): allocate (N, K) row-major, transpose to a (K, N)
    # view with stride (1, K).
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16).T
    assert b.shape == (K, N) and b.stride() == (1, K)
    return a, b


def main():
    torch.manual_seed(0)
    M, N, K = TILE_M, TILE_N, TILE_K
    a, b = make_inputs(M, N, K)
    d = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)

    mma_interface(a, b, out=d)
    torch.cuda.synchronize()

    expected = (a.float() @ b.float()).to(torch.bfloat16)
    print(f"a ({M}x{K}):\n{a}")
    print(f"b ({K}x{N}):\n{b}")
    print(f"d ({M}x{N}):\n{d}")
    print(f"expected:\n{expected}")

    # BF16 inputs, FP32 accumulator, BF16 output — tolerance scales with K.
    torch.testing.assert_close(d, expected, rtol=1e-2, atol=1e-2)
    print("PASS")


if __name__ == "__main__":
    main()
