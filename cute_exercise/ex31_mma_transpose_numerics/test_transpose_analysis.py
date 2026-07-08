import torch

from cute_exercise.ex31_mma_transpose_numerics.transpose_analysis import (
    LAYOUTS,
    expected_operand_stride,
    make_layout_pairs,
    major_mode_name,
    source_with_layout,
    summarize_transpose_results,
    validate_operand_layout,
)


def test_make_layout_pairs_covers_row_column_cross_product():
    assert make_layout_pairs() == [
        ("row", "row"),
        ("row", "column"),
        ("column", "row"),
        ("column", "column"),
    ]


def test_source_with_layout_preserves_values_but_changes_strides():
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    row = source_with_layout(x, "row")
    column = source_with_layout(x, "column")

    assert LAYOUTS == ("row", "column")
    assert torch.equal(row, x)
    assert torch.equal(column, x)
    assert row.stride(-1) == 1
    assert column.stride(-1) != 1


def test_layout_major_names_match_tcgen05_operand_modes():
    assert major_mode_name("row") == "K"
    assert major_mode_name("column") == "MN"


def test_validate_operand_layout_accepts_row_and_column_strides():
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    row = source_with_layout(x, "row")
    column = source_with_layout(x, "column")

    assert expected_operand_stride(row, "row") == (4, 1)
    assert expected_operand_stride(column, "column") == (1, 3)
    validate_operand_layout(row, "row")
    validate_operand_layout(column, "column")

    try:
        validate_operand_layout(row, "column")
    except ValueError as exc:
        assert "expected column layout" in str(exc)
    else:
        raise AssertionError("row-major tensor should not validate as column layout")


def test_summarize_transpose_results_counts_mismatches_by_layout_pair():
    class Row:
        def __init__(self, layout_a, layout_b, same_bits, max_abs_err, max_ulp):
            self.layout_a = layout_a
            self.layout_b = layout_b
            self.same_bits = same_bits
            self.max_abs_err = max_abs_err
            self.max_ulp = max_ulp

    summary = summarize_transpose_results(
        [
            Row("row", "row", True, 0.0, 0),
            Row("row", "column", False, 0.5, 3),
            Row("column", "row", False, 0.25, 2),
        ]
    )

    assert summary == {
        "cases": 3,
        "mismatches": 2,
        "max_abs_err": 0.5,
        "max_ulp": 3,
        "mismatch_layout_pairs": [("row", "column"), ("column", "row")],
    }
