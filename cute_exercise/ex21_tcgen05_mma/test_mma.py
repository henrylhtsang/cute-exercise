"""Single-tile BF16 MMA test. Compares D = A @ B against torch.matmul.

Layouts:
  A: (M, K) BF16 row-major (K-major).
  B: (N, K) BF16 row-major (K-major) — CuTe MMA-B canonical form.
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
    # CuTe MMA-B canonical form: shape (N, K), K-major (stride (K, 1)).
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    assert b.shape == (N, K) and b.stride() == (K, 1)
    return a, b


def main():
    torch.manual_seed(0)
    M, N, K = TILE_M, TILE_N, TILE_K
    a, b = make_inputs(M, N, K)
    d = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)

    mma_interface(a, b, out=d)
    torch.cuda.synchronize()

    # b is (N, K); reference matmul needs (K, N), so transpose for the ref.
    expected = (a.float() @ b.float().T).to(torch.bfloat16)
    print(f"a ({M}x{K}):\n{a}")
    print(f"b ({N}x{K}):\n{b}")
    print(f"d ({M}x{N}):\n{d}")
    print(f"expected:\n{expected}")

    # BF16 inputs, FP32 accumulator, BF16 output — tolerance scales with K.
    torch.testing.assert_close(d, expected, rtol=1e-2, atol=1e-2)
    print("PASS")


if __name__ == "__main__":
    main()
