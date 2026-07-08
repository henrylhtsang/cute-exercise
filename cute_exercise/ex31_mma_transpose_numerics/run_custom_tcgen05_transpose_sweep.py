"""Sweep transpose-equivalent BF16 products with a custom CuTe tcgen05 kernel."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from cute_exercise.ex21_tcgen05_mma.kernel import (
    TILE_K,
    TILE_M,
    TILE_N,
)
from cute_exercise.ex29_mma_offset_numerics.analysis import compare_outputs
from cute_exercise.ex31_mma_transpose_numerics.layout_tcgen05_mma import (
    layout_mma_interface,
)
from cute_exercise.ex31_mma_transpose_numerics.transpose_analysis import (
    TransposeCaseResult,
    make_layout_pairs,
    major_mode_name,
    source_with_layout,
    summarize_transpose_results,
)


def _run_kernel_pair(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    layout_a: str,
    layout_b: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    direct = layout_mma_interface(a, b, layout_a=layout_a, layout_b=layout_b)
    swapped_t = layout_mma_interface(
        b,
        a,
        layout_a=layout_b,
        layout_b=layout_a,
    ).T.contiguous()
    torch.cuda.synchronize()
    return direct, swapped_t


def run_custom_transpose_sweep(*, seed: int) -> dict:
    torch.manual_seed(seed)
    base_a = torch.randn(TILE_M, TILE_K, device="cuda", dtype=torch.bfloat16)
    base_b = torch.randn(TILE_N, TILE_K, device="cuda", dtype=torch.bfloat16)

    baseline, baseline_swapped_t = _run_kernel_pair(
        base_a,
        base_b,
        layout_a="row",
        layout_b="row",
    )
    baseline_transpose_cmp = compare_outputs(baseline, baseline_swapped_t)

    rows: list[TransposeCaseResult] = []
    for layout_a, layout_b in make_layout_pairs():
        source_a = source_with_layout(base_a, layout_a)
        source_b = source_with_layout(base_b, layout_b)

        direct, swapped_t = _run_kernel_pair(
            source_a,
            source_b,
            layout_a=layout_a,
            layout_b=layout_b,
        )
        transpose_cmp = compare_outputs(direct, swapped_t)
        baseline_cmp = compare_outputs(direct, baseline)
        rows.append(
            TransposeCaseResult(
                layout_a=layout_a,
                layout_b=layout_b,
                same_bits=transpose_cmp.same_bits,
                max_abs_err=transpose_cmp.max_abs_err,
                max_ulp=transpose_cmp.max_ulp,
                same_bits_vs_row_baseline=baseline_cmp.same_bits,
                max_abs_err_vs_row_baseline=baseline_cmp.max_abs_err,
                max_ulp_vs_row_baseline=baseline_cmp.max_ulp,
                copied_a=False,
                copied_b=False,
                mma_major_mode_a=major_mode_name(layout_a),
                mma_major_mode_b=major_mode_name(layout_b),
                source_stride_a=tuple(int(value) for value in source_a.stride()),
                source_stride_b=tuple(int(value) for value in source_b.stride()),
                kernel_stride_a=tuple(int(value) for value in source_a.stride()),
                kernel_stride_b=tuple(int(value) for value in source_b.stride()),
                suspected_cause=(
                    "same output bits"
                    if transpose_cmp.same_bits
                    else "swapping A/B MMA operand roles changed output bits"
                ),
            )
        )

    return {
        "kernel": "custom_tcgen05_mma_ex21",
        "shape": {"m": TILE_M, "n": TILE_N, "k": TILE_K},
        "dtype": "bf16",
        "accumulator": "f32",
        "seed": seed,
        "question": "Compare A @ B.T with (B @ A.T).T for row/column source layouts.",
        "baseline_row_row_transpose_comparison": asdict(baseline_transpose_cmp),
        "summary": summarize_transpose_results(rows),
        "results": [asdict(row) for row in rows],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = run_custom_transpose_sweep(seed=args.seed)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is None:
        print(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
