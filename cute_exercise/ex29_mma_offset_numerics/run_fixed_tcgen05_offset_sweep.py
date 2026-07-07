"""Sweep storage/output offsets with the fixed 128x128x128 CuTe kernel."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from cute_exercise.ex29_mma_offset_numerics.analysis import (
    DEFAULT_OFFSETS,
    compare_outputs,
    make_offset_pairs,
    summarize_results,
)
from cute_exercise.ex29_mma_offset_numerics.fixed_tcgen05_mma import (
    TILE_K,
    TILE_M,
    TILE_N,
    fixed_gemm_interface,
    fixed_mma_interface,
)


def parse_offsets(text: str) -> list[int]:
    return [int(value.strip()) for value in text.split(",") if value.strip()]


def round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def run_fixed_offset_sweep(*, seed: int, offsets: list[int]) -> dict:
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
    for m0, n0 in make_offset_pairs(offsets):
        logical_m = m0 + TILE_M
        logical_n = n0 + TILE_N
        physical_n = round_up(logical_n, 8)
        a_pad = torch.zeros((logical_m, TILE_K), device="cuda", dtype=torch.bfloat16)
        b_pad = torch.zeros((physical_n, TILE_K), device="cuda", dtype=torch.bfloat16)
        a_pad[m0 : m0 + TILE_M, :] = a
        b_pad[n0 : n0 + TILE_N, :] = b

        d_pad = fixed_gemm_interface(a_pad, b_pad)
        torch.cuda.synchronize()
        d_slice = d_pad[m0 : m0 + TILE_M, n0 : n0 + TILE_N].contiguous()

        cmp = compare_outputs(d_slice, baseline)
        rows.append(
            {
                "dtype": "bf16",
                "accumulator": "f32",
                "mode": "fixed_cute_padded",
                "m0": m0,
                "n0": n0,
                "same_bits": cmp.same_bits,
                "max_abs_err": cmp.max_abs_err,
                "max_ulp": cmp.max_ulp,
                "logical_padded_m": logical_m,
                "logical_padded_n": logical_n,
                "physical_padded_n": physical_n,
                "suspected_cause": (
                    "same output bits"
                    if cmp.same_bits
                    else "full padded fixed CuTe GEMM changed output slice"
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
        "summary": summarize_results(type("Row", (), row)() for row in rows),
        "results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--offsets", type=parse_offsets, default=list(DEFAULT_OFFSETS))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = run_fixed_offset_sweep(seed=args.seed, offsets=args.offsets)

    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is None:
        print(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
