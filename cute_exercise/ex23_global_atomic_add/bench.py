"""Benchmark global atomic-add contention."""

import argparse

import torch

from cute_exercise.benchmark import benchmark
from cute_exercise.ex23_global_atomic_add.kernel import VARIANTS, atomic_add_interface


def default_ctas() -> int:
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return props.multi_processor_count * 4


def run_case(
    variant: str,
    *,
    num_ctas: int,
    threads: int,
    iters: int,
    counters: int,
) -> tuple[float, float, int]:
    out = torch.empty(1 if variant == "hot" else counters, device="cuda", dtype=torch.int32)
    atomic_add_interface(
        variant=variant,
        num_ctas=num_ctas,
        threads=threads,
        iters=iters,
        counters=counters,
        out=out,
    )
    torch.cuda.synchronize()

    expected = num_ctas * threads * iters
    got = int(out.sum().item())
    if got != expected:
        raise RuntimeError(f"{variant}: expected sum {expected}, got {got}")

    us = benchmark(
        lambda: atomic_add_interface(
            variant=variant,
            num_ctas=num_ctas,
            threads=threads,
            iters=iters,
            counters=counters,
            out=out,
        ),
        warmup=3,
        iters=20,
        runs=3,
    )
    atomics = float(expected)
    return us, atomics / us, got


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS + ("all",), default="all")
    parser.add_argument("--num-ctas", type=int, default=0)
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--iters", type=int, default=2048)
    parser.add_argument("--counters", type=int, default=8192)
    args = parser.parse_args()

    num_ctas = args.num_ctas or default_ctas()
    variants = VARIANTS if args.variant == "all" else (args.variant,)

    print(
        f"num_ctas={num_ctas} threads={args.threads} "
        f"iters={args.iters} striped_counters={args.counters}"
    )
    for variant in variants:
        us, atomic_per_us, got = run_case(
            variant,
            num_ctas=num_ctas,
            threads=args.threads,
            iters=args.iters,
            counters=args.counters,
        )
        print(
            f"{variant:7s}: {us:9.3f} us  "
            f"{atomic_per_us:10.1f} atomic-adds/us  sum={got}"
        )


if __name__ == "__main__":
    main()
