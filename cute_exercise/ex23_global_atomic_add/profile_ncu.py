"""Run Nsight Compute on the global atomic-add microbenchmark.

Invoked without --target, this script profiles both cases. Invoked with
--target, it runs exactly one measured kernel after compile/warmup.
"""

import argparse
import os
import subprocess
import sys

import torch

from cute_exercise.ex23_global_atomic_add.kernel import VARIANTS, atomic_add_interface


NCU = os.environ.get("NCU", "/usr/local/cuda-13.1/bin/ncu")


def default_ctas() -> int:
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return props.multi_processor_count * 4


def run_target(args: argparse.Namespace) -> None:
    counters = 1 if args.target == "hot" else args.counters
    out = torch.empty(counters, device="cuda", dtype=torch.int32)

    atomic_add_interface(
        variant=args.target,
        num_ctas=args.num_ctas,
        threads=args.threads,
        iters=args.iters,
        counters=args.counters,
        out=out,
    )
    torch.cuda.synchronize()

    atomic_add_interface(
        variant=args.target,
        num_ctas=args.num_ctas,
        threads=args.threads,
        iters=args.iters,
        counters=args.counters,
        out=out,
    )
    torch.cuda.synchronize()
    expected = args.num_ctas * args.threads * args.iters
    got = int(out.sum().item())
    if got != expected:
        raise RuntimeError(f"{args.target}: expected sum {expected}, got {got}")


def run_ncu(args: argparse.Namespace, variant: str) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    output = os.path.join(args.output_dir, f"{variant}.ncu-rep")
    cmd = [
        NCU,
        "--target-processes",
        "all",
        "--kernel-name",
        "regex:.*atomic_add.*",
        "--launch-count",
        "1",
        "--force-overwrite",
        "--set",
        "full",
        "--export",
        output,
        sys.executable,
        "-m",
        "cute_exercise.ex23_global_atomic_add.profile_ncu",
        "--target",
        variant,
        "--num-ctas",
        str(args.num_ctas),
        "--threads",
        str(args.threads),
        "--iters",
        str(args.iters),
        "--counters",
        str(args.counters),
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=VARIANTS)
    parser.add_argument("--num-ctas", type=int, default=0)
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--iters", type=int, default=2048)
    parser.add_argument("--counters", type=int, default=8192)
    parser.add_argument("--output-dir", default=os.path.join(os.path.dirname(__file__), "ncu"))
    args = parser.parse_args()

    args.num_ctas = args.num_ctas or default_ctas()
    if args.target:
        run_target(args)
        return

    for variant in VARIANTS:
        run_ncu(args, variant)


if __name__ == "__main__":
    main()
