import pytest
import torch

from cute_exercise.ex23_global_atomic_add.kernel import VARIANTS, atomic_add_interface


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("variant", VARIANTS)
def test_atomic_add_sum_matches_expected(variant):
    num_ctas = 8
    threads = 128
    iters = 16
    counters = 256 if variant == "striped" else 1

    out = atomic_add_interface(
        num_ctas=num_ctas,
        threads=threads,
        iters=iters,
        counters=counters,
        variant=variant,
    )

    expected = num_ctas * threads * iters
    assert int(out.sum().item()) == expected
