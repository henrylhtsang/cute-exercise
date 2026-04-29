"""Correctness: PTX-shipped kernel matches DSL JIT and torch.add."""

import pytest
import torch

from cute_exercise.ex1_for_loop.kernel import elementwise_add_interface
from cute_exercise.ex5_ship_pure_ptx.ptx_runner import elementwise_add_ptx


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_ptx_matches_dsl_and_torch(dtype):
    M, N = 16384, 8192
    torch.manual_seed(0)
    a = torch.randn(M, N, dtype=dtype, device="cuda")
    b = torch.randn(M, N, dtype=dtype, device="cuda")

    out_torch = a + b
    out_dsl = elementwise_add_interface(a, b)
    out_ptx = elementwise_add_ptx(a, b)

    assert torch.equal(out_dsl, out_ptx), (
        f"DSL vs PTX mismatch; max diff = "
        f"{(out_dsl.float() - out_ptx.float()).abs().max().item()}"
    )
    assert torch.equal(out_torch, out_ptx)
