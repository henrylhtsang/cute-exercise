"""Run the experimental fixed 128x128x128 tcgen05 MMA baseline check."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from cute_exercise.ex29_mma_offset_numerics.analysis import compare_outputs
from cute_exercise.ex29_mma_offset_numerics.fixed_tcgen05_mma import (
    TILE_K,
    TILE_M,
    TILE_N,
    fixed_mma_interface,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    a = torch.randn(TILE_M, TILE_K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(TILE_N, TILE_K, device="cuda", dtype=torch.bfloat16)
    out = fixed_mma_interface(a, b)
    torch.cuda.synchronize()

    expected = torch.mm(a, b.T)
    comparison = compare_outputs(out, expected)
    payload = {
        "kernel": "fixed_tcgen05_mma",
        "shape": {"m": TILE_M, "n": TILE_N, "k": TILE_K},
        "dtype": "bf16",
        "accumulator": "f32",
        "seed": args.seed,
        "comparison": asdict(comparison),
        "correct": comparison.same_bits or comparison.max_abs_err <= 1e-2,
        "note": "Fixed 128x128x128 CuTe tile compared against BF16 torch.mm.",
    }

    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is None:
        print(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
