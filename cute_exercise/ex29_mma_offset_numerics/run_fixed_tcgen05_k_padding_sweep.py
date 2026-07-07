"""Sweep K-padding where A and B nonzero blocks are deliberately misaligned."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from cute_exercise.ex29_mma_offset_numerics.analysis import (
    DEFAULT_OFFSETS,
    compare_outputs,
)
from cute_exercise.ex29_mma_offset_numerics.fixed_tcgen05_mma import (
    TILE_K,
    TILE_M,
    TILE_N,
    fixed_gemm_interface,
    fixed_mma_interface,
)
from cute_exercise.ex29_mma_offset_numerics.run_fixed_tcgen05_offset_sweep import round_up


def parse_offsets(text: str) -> list[int]:
    return [int(value.strip()) for value in text.split(",") if value.strip()]


def summarize_k_results(rows: list[dict]) -> dict:
    bad = [row for row in rows if not row["same_bits"]]
    return {
        "cases": len(rows),
        "mismatches": len(bad),
        "max_abs_err": max((row["max_abs_err"] for row in rows), default=0),
        "max_ulp": max((row["max_ulp"] for row in rows), default=0),
        "mismatch_offsets": [row["k0"] for row in bad],
        "zero_outputs": sum(1 for row in rows if row["same_as_zero"]),
    }


def run_k_padding_sweep(*, seed: int, offsets: list[int]) -> dict:
    torch.manual_seed(seed)
    a = torch.randn(TILE_M, TILE_K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(TILE_N, TILE_K, device="cuda", dtype=torch.bfloat16)

    baseline = fixed_mma_interface(a, b)
    torch.cuda.synchronize()
    torch_baseline = torch.mm(a, b.T)
    baseline_cmp = compare_outputs(baseline, torch_baseline)
    if not baseline_cmp.same_bits:
        raise RuntimeError(f"fixed kernel baseline mismatch: {baseline_cmp}")

    rows = []
    zero_ref = torch.zeros_like(baseline)
    for k0 in dict.fromkeys(int(v) for v in offsets):
        logical_k = k0 + TILE_K
        physical_k = round_up(logical_k, TILE_K)
        a_pad = torch.zeros((TILE_M, physical_k), device="cuda", dtype=torch.bfloat16)
        b_pad = torch.zeros((TILE_N, physical_k), device="cuda", dtype=torch.bfloat16)

        # Math convention:
        #   A_pad = [zero, A]
        #   B_pad.T = [B.T; zero]
        # In canonical B storage (N, K), B_pad is [B, zero] along K.
        a_pad[:, k0 : k0 + TILE_K] = a
        b_pad[:, :TILE_K] = b

        actual = fixed_gemm_interface(a_pad, b_pad)
        torch.cuda.synchronize()

        cmp = compare_outputs(actual, baseline)
        zero_cmp = compare_outputs(actual, zero_ref)
        rows.append(
            {
                "dtype": "bf16",
                "accumulator": "f32",
                "mode": "fixed_cute_k_misaligned_padding",
                "k0": k0,
                "logical_k": logical_k,
                "physical_k": physical_k,
                "same_bits": cmp.same_bits,
                "max_abs_err": cmp.max_abs_err,
                "max_ulp": cmp.max_ulp,
                "same_as_zero": zero_cmp.same_bits,
                "max_abs_err_vs_zero": zero_cmp.max_abs_err,
                "max_ulp_vs_zero": zero_cmp.max_ulp,
                "suspected_cause": (
                    "control"
                    if k0 == 0
                    else "A and B nonzero K ranges do not overlap"
                ),
            }
        )

    return {
        "kernel": "fixed_tcgen05_mma",
        "shape": {"m": TILE_M, "n": TILE_N, "k": TILE_K},
        "dtype": "bf16",
        "accumulator": "f32",
        "seed": seed,
        "baseline_comparison_to_torch_mm_bf16": asdict(baseline_cmp),
        "summary": summarize_k_results(rows),
        "results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--offsets", type=parse_offsets, default=list(DEFAULT_OFFSETS))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = run_k_padding_sweep(seed=args.seed, offsets=args.offsets)

    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is None:
        print(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
