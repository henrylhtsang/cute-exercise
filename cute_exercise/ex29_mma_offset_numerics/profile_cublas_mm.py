"""Profile one BF16 torch.mm case for the ex29 offset experiment.

Use with Nsight Compute, for example:

    ncu --profile-from-start off python -m \
      cute_exercise.ex29_mma_offset_numerics.profile_cublas_mm --case baseline
"""

from __future__ import annotations

import argparse

import torch

from cute_exercise.ex29_mma_offset_numerics.analysis import (
    compare_outputs,
    deterministic_algorithms,
    make_inputs,
    make_padded_inputs,
    matmul_precision,
)


def _profiler_start() -> None:
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()


def _profiler_stop() -> None:
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("baseline", "padded"), required=True)
    parser.add_argument("--m0", type=int, default=127)
    parser.add_argument("--n0", type=int, default=127)
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", choices=("default", "cublas", "cublaslt"), default="cublas")
    parser.add_argument("--allow-bf16-reduced", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    torch.backends.cuda.preferred_blas_library(args.backend)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = args.allow_bf16_reduced

    a, b = make_inputs(
        dtype_name="bf16",
        device="cuda",
        m=args.m,
        n=args.n,
        k=args.k,
        seed=args.seed,
    )
    a_pad, b_pad = make_padded_inputs(a, b, m0=args.m0, n0=args.n0)

    with deterministic_algorithms(True), matmul_precision("bf16"):
        baseline = torch.mm(a, b)
        if args.case == "baseline":
            for _ in range(5):
                out = torch.mm(a, b)
            _profiler_start()
            out = torch.mm(a, b)
            _profiler_stop()
        else:
            for _ in range(5):
                out = torch.mm(a_pad, b_pad)[args.m0 : args.m0 + args.m, args.n0 : args.n0 + args.n]
            _profiler_start()
            out = torch.mm(a_pad, b_pad)[args.m0 : args.m0 + args.m, args.n0 : args.n0 + args.n]
            _profiler_stop()
            out = out.contiguous()

    torch.cuda.synchronize()
    comparison = compare_outputs(out.contiguous(), baseline)
    print(
        {
            "case": args.case,
            "backend": str(torch.backends.cuda.preferred_blas_library()),
            "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "m0": args.m0,
            "n0": args.n0,
            "out_shape": tuple(out.shape),
            "same_bits_vs_baseline": comparison.same_bits,
            "max_abs_err": comparison.max_abs_err,
            "max_ulp": comparison.max_ulp,
        }
    )


if __name__ == "__main__":
    main()
