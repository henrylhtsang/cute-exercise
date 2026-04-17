import pytest
import torch

from cute_exercise.for_loop.kernel import VARIANTS, elementwise_add_interface


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"]
)
@pytest.mark.parametrize("variant", ["torch.add", *VARIANTS])
def test_elementwise_add(variant, dtype):
    torch.manual_seed(0)
    M, N = 1024, 2048
    a = torch.randn(M, N, device="cuda", dtype=dtype)
    b = torch.randn(M, N, device="cuda", dtype=dtype)
    c = torch.zeros(M, N, device="cuda", dtype=dtype)

    if variant == "torch.add":
        torch.add(a, b, out=c)
    else:
        elementwise_add_interface(a, b, out=c, variant=variant)

    torch.testing.assert_close(c, a + b)
