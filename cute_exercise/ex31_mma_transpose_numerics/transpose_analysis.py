"""Helpers for fixed-MMA transpose numerics experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


LAYOUTS = ("row", "column")


@dataclass(frozen=True)
class TransposeCaseResult:
    layout_a: str
    layout_b: str
    same_bits: bool
    max_abs_err: float
    max_ulp: int
    same_bits_vs_row_baseline: bool
    max_abs_err_vs_row_baseline: float
    max_ulp_vs_row_baseline: int
    copied_a: bool
    copied_b: bool
    mma_major_mode_a: str
    mma_major_mode_b: str
    source_stride_a: tuple[int, int]
    source_stride_b: tuple[int, int]
    kernel_stride_a: tuple[int, int]
    kernel_stride_b: tuple[int, int]
    suspected_cause: str


def make_layout_pairs() -> list[tuple[str, str]]:
    return [(layout_a, layout_b) for layout_a in LAYOUTS for layout_b in LAYOUTS]


def source_with_layout(tensor: torch.Tensor, layout: str) -> torch.Tensor:
    """Return a same-valued 2-D tensor with row- or column-major physical layout."""
    if tensor.dim() != 2:
        raise ValueError("expected a 2-D tensor")
    if layout == "row":
        return tensor.contiguous()
    if layout == "column":
        return tensor.t().contiguous().t()
    raise ValueError(f"unknown layout {layout!r}")


def major_mode_name(layout: str) -> str:
    if layout == "row":
        return "K"
    if layout == "column":
        return "MN"
    raise ValueError(f"unknown layout {layout!r}")


def expected_operand_stride(tensor: torch.Tensor, layout: str) -> tuple[int, int]:
    if tensor.dim() != 2:
        raise ValueError("expected a 2-D tensor")
    rows, cols = tensor.shape
    if layout == "row":
        return (cols, 1)
    if layout == "column":
        return (1, rows)
    raise ValueError(f"unknown layout {layout!r}")


def validate_operand_layout(tensor: torch.Tensor, layout: str) -> None:
    """Validate that a tensor's physical layout matches the requested MMA mode."""
    if tensor.dim() != 2:
        raise ValueError("expected a 2-D tensor")
    expected = expected_operand_stride(tensor, layout)
    actual = tuple(int(value) for value in tensor.stride())
    if actual != expected:
        raise ValueError(
            f"expected {layout} layout with stride {expected}, got stride {actual}"
        )


def suspected_cause(result) -> str:
    if result.same_bits:
        return "same output bits"
    return "swapping A/B MMA operand roles changed output bits"


def summarize_transpose_results(results: Iterable) -> dict:
    rows = list(results)
    bad = [row for row in rows if not row.same_bits]
    return {
        "cases": len(rows),
        "mismatches": len(bad),
        "max_abs_err": max((row.max_abs_err for row in bad), default=0.0),
        "max_ulp": max((row.max_ulp for row in bad), default=0),
        "mismatch_layout_pairs": [(row.layout_a, row.layout_b) for row in bad],
    }
