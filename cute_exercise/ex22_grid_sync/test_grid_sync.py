import pytest
import torch

from cute_exercise.ex22_grid_sync.kernel import VARIANTS, ref_rmsnorm, rmsnorm_interface


@pytest.mark.parametrize("dtype", [torch.float32], ids=["fp32"])
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("N", [256 * 1024, 4 * 1024 * 1024 + 17], ids=["256Ki", "4Mi+17"])
def test_rmsnorm(variant, dtype, N):
    torch.manual_seed(0)
    x = torch.randn(N, device="cuda", dtype=dtype)
    out = rmsnorm_interface(x, variant=variant, eps=1e-6)
    ref = ref_rmsnorm(x, eps=1e-6)
    torch.testing.assert_close(out, ref, atol=2e-3, rtol=2e-3)


if __name__ == "__main__":
    torch.manual_seed(0)
    for variant in VARIANTS:
        for N in [256 * 1024, 4 * 1024 * 1024 + 17, 64 * 1024 * 1024]:
            x = torch.randn(N, device="cuda", dtype=torch.float32)
            out = rmsnorm_interface(x, variant=variant)
            ref = ref_rmsnorm(x)
            err = (out - ref).abs().max().item()
            ok = torch.allclose(out, ref, atol=2e-3, rtol=2e-3)
            print(f"{variant:11s} N={N:>10d}  max_err={err:.2e}  {'OK' if ok else 'FAIL'}")
