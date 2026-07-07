"""Offset-sweep helpers for MMA/GEMM bitwise numerics experiments."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch


DEFAULT_OFFSETS = (0, 1, 2, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128)


@dataclass(frozen=True)
class Comparison:
    same_bits: bool
    max_abs_err: float
    max_ulp: int


@dataclass(frozen=True)
class CaseResult:
    dtype: str
    accumulator: str
    mode: str
    m0: int
    n0: int
    same_bits: bool
    max_abs_err: float
    max_ulp: int
    suspected_cause: str


def make_offset_pairs(offsets: Iterable[int]) -> list[tuple[int, int]]:
    """Build a compact sweep containing control, axis-only, and mixed offsets."""
    values = list(dict.fromkeys(int(v) for v in offsets))
    pairs: list[tuple[int, int]] = []

    def add(pair: tuple[int, int]) -> None:
        if pair not in pairs:
            pairs.append(pair)

    add((0, 0))
    for value in values:
        add((value, 0))
        add((0, value))
    for m0 in values:
        for n0 in values:
            add((m0, n0))
    return pairs


def make_padded_inputs(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    m0: int,
    n0: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Embed A and B into the padded matrices from the exercise statement."""
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError("expected 2-D matrices")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"K mismatch: {tuple(a.shape)} x {tuple(b.shape)}")
    if m0 < 0 or n0 < 0:
        raise ValueError("offsets must be non-negative")

    m, k = a.shape
    _, n = b.shape
    a_pad = torch.zeros((m0 + m, k), device=a.device, dtype=a.dtype)
    b_pad = torch.zeros((k, n0 + n), device=b.device, dtype=b.dtype)
    a_pad[m0 : m0 + m, :] = a
    b_pad[:, n0 : n0 + n] = b
    return a_pad, b_pad


def _ordered_float_bits(x: torch.Tensor) -> torch.Tensor:
    """Return integer keys whose distance is ULP distance for finite floats."""
    x = x.detach().contiguous().cpu()
    if x.dtype == torch.float32:
        bits = torch.bitwise_and(x.view(torch.int32).to(torch.int64), 0xFFFFFFFF)
        sign = 0x80000000
    elif x.dtype in (torch.float16, torch.bfloat16):
        bits = torch.bitwise_and(x.view(torch.int16).to(torch.int32), 0xFFFF)
        sign = 0x8000
    else:
        raise TypeError(f"ULP distance is not implemented for {x.dtype}")
    return torch.where(torch.bitwise_and(bits, sign) != 0, sign - bits, bits + sign)


def max_ulp_distance(actual: torch.Tensor, expected: torch.Tensor) -> int:
    if actual.shape != expected.shape:
        raise ValueError(f"shape mismatch: {tuple(actual.shape)} vs {tuple(expected.shape)}")
    if actual.dtype != expected.dtype:
        raise TypeError(f"dtype mismatch: {actual.dtype} vs {expected.dtype}")
    if actual.numel() == 0:
        return 0
    a_bits = _ordered_float_bits(actual)
    e_bits = _ordered_float_bits(expected)
    return int((a_bits - e_bits).abs().max().item())


def compare_outputs(actual: torch.Tensor, expected: torch.Tensor) -> Comparison:
    if actual.shape != expected.shape:
        raise ValueError(f"shape mismatch: {tuple(actual.shape)} vs {tuple(expected.shape)}")
    if actual.dtype != expected.dtype:
        raise TypeError(f"dtype mismatch: {actual.dtype} vs {expected.dtype}")
    diff = actual.float() - expected.float()
    return Comparison(
        same_bits=bool(torch.equal(actual, expected)),
        max_abs_err=float(diff.abs().max().item()) if diff.numel() else 0.0,
        max_ulp=max_ulp_distance(actual, expected),
    )


def mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Run GEMM through torch.mm, not Tensor.__matmul__."""
    return torch.mm(a, b)


@contextlib.contextmanager
def matmul_precision(dtype_name: str):
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = dtype_name == "tf32"
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


@contextlib.contextmanager
def deterministic_algorithms(enabled: bool):
    old = torch.are_deterministic_algorithms_enabled()
    if enabled:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        torch.use_deterministic_algorithms(enabled)
        yield
    finally:
        torch.use_deterministic_algorithms(old)


def torch_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "f16":
        return torch.float16
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "tf32":
        return torch.float32
    raise ValueError(f"unknown dtype {dtype_name!r}")


def make_inputs(
    *,
    dtype_name: str,
    device: str,
    m: int,
    n: int,
    k: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    dtype = torch_dtype(dtype_name)
    a = torch.randn((m, k), generator=generator, device=device, dtype=torch.float32)
    b = torch.randn((k, n), generator=generator, device=device, dtype=torch.float32)
    return a.to(dtype), b.to(dtype)


def suspected_cause(mode: str, m0: int, n0: int, comparison: Comparison) -> str:
    if comparison.same_bits:
        return "same output bits"
    if mode == "view":
        changed = []
        if m0:
            changed.append("A storage offset")
        if n0:
            changed.append("B storage offset/stride")
        return ", ".join(changed) or "control mismatch"
    return "larger GEMM shape may select different tiles/reduction path"


def run_case(
    *,
    dtype_name: str,
    mode: str,
    m0: int,
    n0: int,
    m: int,
    n: int,
    k: int,
    seed: int,
    device: str,
    deterministic: bool,
) -> CaseResult:
    a, b = make_inputs(dtype_name=dtype_name, device=device, m=m, n=n, k=k, seed=seed)
    return run_case_from_inputs(
        a,
        b,
        dtype_name=dtype_name,
        mode=mode,
        m0=m0,
        n0=n0,
        deterministic=deterministic,
    )


def run_case_from_inputs(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    dtype_name: str,
    mode: str,
    m0: int,
    n0: int,
    deterministic: bool,
) -> CaseResult:
    m, k = a.shape
    _, n = b.shape
    a_pad, b_pad = make_padded_inputs(a, b, m0=m0, n0=n0)

    with deterministic_algorithms(deterministic), matmul_precision(dtype_name):
        baseline = mm(a, b)
        if mode == "view":
            actual = mm(a_pad[m0 : m0 + m, :], b_pad[:, n0 : n0 + n])
        elif mode == "padded":
            c_pad = mm(a_pad, b_pad)
            actual = c_pad[m0 : m0 + m, n0 : n0 + n].contiguous()
        else:
            raise ValueError(f"unknown mode {mode!r}")

    comparison = compare_outputs(actual, baseline)
    return CaseResult(
        dtype=dtype_name,
        accumulator="f32",
        mode=mode,
        m0=m0,
        n0=n0,
        same_bits=comparison.same_bits,
        max_abs_err=comparison.max_abs_err,
        max_ulp=comparison.max_ulp,
        suspected_cause=suspected_cause(mode, m0, n0, comparison),
    )


def run_sweep(
    *,
    dtype_name: str,
    mode: str,
    offsets: Iterable[int],
    m: int = 128,
    n: int = 128,
    k: int = 128,
    seed: int = 0,
    device: str = "cuda",
    deterministic: bool = True,
) -> list[CaseResult]:
    a, b = make_inputs(dtype_name=dtype_name, device=device, m=m, n=n, k=k, seed=seed)
    return [
        run_case_from_inputs(
            a,
            b,
            dtype_name=dtype_name,
            mode=mode,
            m0=m0,
            n0=n0,
            deterministic=deterministic,
        )
        for m0, n0 in make_offset_pairs(offsets)
    ]


def summarize_results(results: Iterable[CaseResult]) -> dict:
    rows = list(results)
    bad = [row for row in rows if not row.same_bits]
    return {
        "cases": len(rows),
        "mismatches": len(bad),
        "max_abs_err": max((row.max_abs_err for row in bad), default=0.0),
        "max_ulp": max((row.max_ulp for row in bad), default=0),
        "mismatch_offsets": [(row.m0, row.n0) for row in bad],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("f16", "bf16", "tf32"), default="tf32")
    parser.add_argument("--mode", choices=("view", "padded"), default="padded")
    parser.add_argument("--offsets", default=",".join(str(v) for v in DEFAULT_OFFSETS))
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable torch deterministic algorithms during GEMM",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    offsets = [int(value) for value in args.offsets.split(",") if value]
    results = run_sweep(
        dtype_name=args.dtype,
        mode=args.mode,
        offsets=offsets,
        m=args.m,
        n=args.n,
        k=args.k,
        seed=args.seed,
        device=args.device,
        deterministic=args.deterministic,
    )
    payload = {
        "dtype": args.dtype,
        "mode": args.mode,
        "shape": {"m": args.m, "n": args.n, "k": args.k},
        "seed": args.seed,
        "deterministic": args.deterministic,
        "matmul": "torch.mm",
        "summary": summarize_results(results),
        "results": [asdict(result) for result in results],
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is None:
        print(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
