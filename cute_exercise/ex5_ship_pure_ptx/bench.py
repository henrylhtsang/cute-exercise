"""Compare DSL JIT vs pre-compiled PTX for the ElementwiseAdd kernel.

Two things to measure:

1. Cold-start latency: how long does it take from "import + first call"
   to "kernel result is ready"? DSL pays cute.compile (seconds). PTX pays
   only file read + cuModuleLoadData (driver re-translates PTX -> SASS).

2. Steady-state throughput: same kernel, same SASS, should be identical.

Each variant runs in a fresh subprocess so the cold-start measurement is
honest (no shared module / context state with the others).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import torch

from cute_exercise.benchmark import bandwidth_gbps, benchmark
from cute_exercise.ex5_ship_pure_ptx._common import DTYPES


_HERE = Path(__file__).parent


_COLDSTART_TEMPLATE = r"""
import json, time, sys
import torch

# t0: process is up; we measure from "first DSL/PTX import + compile + first
# kernel result on host" — that's what an end user actually sees.
M, N = {M}, {N}
dtype_name = "{dtype}"
from cute_exercise.ex5_ship_pure_ptx._common import DTYPES
dtype = DTYPES[dtype_name]
torch.manual_seed(0)

# Allocate inputs outside the timing window so we measure
# compile/load + first launch, not allocator warm-up.
a = torch.randn(M, N, dtype=dtype, device="cuda")
b = torch.randn(M, N, dtype=dtype, device="cuda")
c = torch.empty_like(a)

# Force CUDA context init outside the timing window.
torch.cuda.synchronize()

t0 = time.perf_counter()
mode = "{mode}"
if mode == "dsl":
    from cute_exercise.ex1_for_loop.kernel import elementwise_add_interface
    elementwise_add_interface(a, b, out=c)
elif mode == "ptx":
    from cute_exercise.ex5_ship_pure_ptx.ptx_runner import elementwise_add_ptx
    elementwise_add_ptx(a, b, out=c)
elif mode == "torch":
    torch.add(a, b, out=c)
else:
    raise ValueError(mode)
torch.cuda.synchronize()
t1 = time.perf_counter()

ok = bool(torch.equal(c, a + b))
print(json.dumps({{"mode": mode, "us": (t1 - t0) * 1e6, "ok": ok}}))
"""


def _coldstart_us(mode: str, M: int, N: int, dtype_name: str) -> float:
    src = _COLDSTART_TEMPLATE.format(M=M, N=N, dtype=dtype_name, mode=mode)
    out = subprocess.run(
        [sys.executable, "-c", src],
        check=True, capture_output=True, text=True,
    )
    last = out.stdout.strip().splitlines()[-1]
    rec = json.loads(last)
    if not rec["ok"]:
        raise RuntimeError(f"cold-start {mode} produced wrong result")
    return rec["us"]


def _steady_state(M: int, N: int, dtype: torch.dtype) -> dict[str, tuple[float, float]]:
    """In-process steady-state: same kernel, same SASS — DSL ≈ PTX ≈ torch."""
    from cute_exercise.ex1_for_loop.kernel import elementwise_add_interface
    from cute_exercise.ex5_ship_pure_ptx.ptx_runner import elementwise_add_ptx

    a = torch.randn(M, N, dtype=dtype, device="cuda")
    b = torch.randn(M, N, dtype=dtype, device="cuda")
    c = torch.empty_like(a)
    total_bytes = (a.numel() + b.numel() + c.numel()) * a.element_size()

    results: dict[str, tuple[float, float]] = {}
    fns = {
        "torch.add": lambda: torch.add(a, b, out=c),
        "dsl": lambda: elementwise_add_interface(a, b, out=c),
        "ptx": lambda: elementwise_add_ptx(a, b, out=c),
    }
    for name, fn in fns.items():
        fn()
        torch.cuda.synchronize()
        torch.testing.assert_close(c, a + b)
        us = benchmark(fn)
        results[name] = (us, bandwidth_gbps(total_bytes, us))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", nargs=2, type=int, default=[16384, 8192], metavar=("M", "N"))
    parser.add_argument("--dtype", choices=["fp16", "fp32"], nargs="+", default=["fp16", "fp32"])
    parser.add_argument("--skip-coldstart", action="store_true")
    parser.add_argument("--coldstart-trials", type=int, default=3)
    args = parser.parse_args(argv)

    M, N = args.shape

    print(textwrap.dedent(f"""
        Shape: M={M}, N={N}
        ====================================
    """).strip())

    for dtype_name in args.dtype:
        dtype = DTYPES[dtype_name]
        print(f"\n--- {dtype_name} ---")

        if not args.skip_coldstart:
            print("\nCold-start (process boot -> first result, ms):")
            for mode in ("torch", "dsl", "ptx"):
                trials_us = [_coldstart_us(mode, M, N, dtype_name) for _ in range(args.coldstart_trials)]
                trials_ms = [u / 1000 for u in trials_us]
                med = sorted(trials_ms)[len(trials_ms) // 2]
                print(f"  {mode:5s}: median {med:8.1f} ms  (trials: {[f'{x:.0f}' for x in trials_ms]})")

        print("\nSteady-state (same kernel, in-process, us / GB/s):")
        for name, (us, gbps) in _steady_state(M, N, dtype).items():
            print(f"  {name:10s}: {us:8.3f} us   {gbps:8.1f} GB/s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
