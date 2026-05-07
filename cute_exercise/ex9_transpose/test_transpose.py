"""Small fp32 transpose test. M and N must be divisible by 4."""
import torch

from cute_exercise.ex9_transpose.kernel import transpose_interface


def main():
    torch.manual_seed(0)
    M, N = 64, 64
    a = torch.randn(M, N, device="cuda", dtype=torch.float32)
    b = torch.empty(N, M, device="cuda", dtype=torch.float32)

    transpose_interface(a, out=b)
    torch.cuda.synchronize()

    expected = a.T.contiguous()
    print(f"a ({M}x{N}):\n{a}")
    print(f"b ({N}x{M}):\n{b}")
    print(f"expected:\n{expected}")

    torch.testing.assert_close(b, expected)
    print("PASS")


if __name__ == "__main__":
    main()
