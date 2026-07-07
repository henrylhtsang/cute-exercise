import torch

import cute_exercise.ex29_mma_offset_numerics.analysis as analysis
from cute_exercise.ex29_mma_offset_numerics.analysis import (
    compare_outputs,
    make_offset_pairs,
    make_padded_inputs,
    max_ulp_distance,
    summarize_results,
)


def test_make_offset_pairs_includes_control_axes_and_mixed_cases():
    pairs = make_offset_pairs([0, 1, 16])

    assert pairs[0] == (0, 0)
    assert (1, 0) in pairs
    assert (0, 16) in pairs
    assert (16, 1) in pairs
    assert len(pairs) == len(set(pairs))


def test_make_padded_inputs_places_operands_at_requested_offsets():
    a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    b = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    a_pad, b_pad = make_padded_inputs(a, b, m0=2, n0=3)

    assert a_pad.shape == (4, 3)
    assert b_pad.shape == (3, 7)
    assert torch.equal(a_pad[:2], torch.zeros(2, 3))
    assert torch.equal(a_pad[2:], a)
    assert torch.equal(b_pad[:, :3], torch.zeros(3, 3))
    assert torch.equal(b_pad[:, 3:], b)


def test_max_ulp_distance_counts_adjacent_float32_values():
    base = torch.tensor([1.0], dtype=torch.float32)
    adjacent = torch.nextafter(base, torch.tensor([2.0], dtype=torch.float32))

    assert max_ulp_distance(base, base) == 0
    assert max_ulp_distance(base, adjacent) == 1


def test_compare_outputs_reports_bitwise_match_and_mismatch():
    expected = torch.tensor([1.0, 2.0], dtype=torch.float32)
    same = expected.clone()
    different = torch.tensor([1.0, 2.25], dtype=torch.float32)

    same_report = compare_outputs(same, expected)
    different_report = compare_outputs(different, expected)

    assert same_report.same_bits
    assert same_report.max_abs_err == 0.0
    assert same_report.max_ulp == 0
    assert not different_report.same_bits
    assert different_report.max_abs_err == 0.25
    assert different_report.max_ulp > 0


def test_mm_uses_torch_mm(monkeypatch):
    calls = []

    def fake_mm(a, b):
        calls.append((a, b))
        return torch.ones(a.shape[0], b.shape[1], dtype=a.dtype)

    monkeypatch.setattr(analysis.torch, "mm", fake_mm)

    a = torch.zeros(2, 3)
    b = torch.zeros(3, 4)

    out = analysis.mm(a, b)

    assert out.shape == (2, 4)
    assert calls == [(a, b)]


def test_deterministic_context_enables_and_restores_previous_state():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)

    with analysis.deterministic_algorithms(True):
        assert torch.are_deterministic_algorithms_enabled()

    assert not torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(previous)


def test_summarize_results_counts_mismatches_and_offsets():
    rows = [
        analysis.CaseResult("bf16", "f32", "view", 0, 0, True, 0.0, 0, "same"),
        analysis.CaseResult("bf16", "f32", "view", 1, 7, False, 0.5, 3, "offset"),
        analysis.CaseResult("bf16", "f32", "view", 2, 7, False, 0.25, 2, "offset"),
    ]

    summary = summarize_results(rows)

    assert summary["cases"] == 3
    assert summary["mismatches"] == 2
    assert summary["max_abs_err"] == 0.5
    assert summary["max_ulp"] == 3
    assert summary["mismatch_offsets"] == [(1, 7), (2, 7)]


def test_run_sweep_reuses_one_input_pair(monkeypatch):
    calls = []

    def fake_make_inputs(**kwargs):
        calls.append(kwargs)
        return torch.ones(2, 3), torch.ones(3, 2)

    monkeypatch.setattr(analysis, "make_inputs", fake_make_inputs)

    results = analysis.run_sweep(
        dtype_name="bf16",
        mode="view",
        offsets=[0, 1],
        m=2,
        n=2,
        k=3,
        device="cpu",
    )

    assert len(calls) == 1
    assert len(results) == 4
