"""Benchmark all best-vector-add configs for the requested B200 shapes."""

from __future__ import annotations

import argparse
import statistics

import torch

from cute_exercise.benchmark import bandwidth_gbps, benchmark as eager_benchmark
from cute_exercise.ex29_best_vector_add.kernel import (
    B200_SHAPES,
    CONFIGS,
    clear_jit_autotune_cache,
    jit_autotune_config,
    vector_add_interface,
)


def cuda_graph_benchmark(fn, warmup: int = 5, iters: int = 50, runs: int = 3) -> float:
    """Time CUDA graph replay of ``fn`` and return microseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()

    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()

    run_medians = []
    for _ in range(runs):
        events = [
            (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            for _ in range(iters)
        ]
        for start, end in events:
            start.record()
            graph.replay()
            end.record()
        torch.cuda.synchronize()
        run_medians.append(statistics.median(s.elapsed_time(e) * 1000 for s, e in events))
    return statistics.median(run_medians)


def run_one(
    size: int,
    warmup: int,
    iters: int,
    runs: int,
    use_graphs: bool,
    include_jit_autotune: bool,
) -> list[tuple[str, float, float]]:
    a = torch.randn(size, size, device="cuda", dtype=torch.float16)
    b = torch.randn(size, size, device="cuda", dtype=torch.float16)
    out = torch.empty_like(a)
    expected = a + b
    total_bytes = (a.numel() + b.numel() + out.numel()) * a.element_size()
    time_fn = cuda_graph_benchmark if use_graphs else eager_benchmark

    results = []
    torch.add(a, b, out=out)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    us = time_fn(lambda: torch.add(a, b, out=out), warmup=warmup, iters=iters, runs=runs)
    results.append(("torch.add", us, bandwidth_gbps(total_bytes, us)))

    if include_jit_autotune:
        clear_jit_autotune_cache()
        vector_add_interface(a, b, out=out, config_name="jit_autotune")
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        tuned_cfg = jit_autotune_config(size)
        us = time_fn(
            lambda: vector_add_interface(a, b, out=out, config_name="jit_autotune"),
            warmup=warmup,
            iters=iters,
            runs=runs,
        )
        results.append((f"jit_autotune:{tuned_cfg.name}", us, bandwidth_gbps(total_bytes, us)))

    for cfg in CONFIGS:
        try:
            vector_add_interface(a, b, out=out, config=cfg)
        except ValueError as exc:
            results.append((f"{cfg.name} (skip: {exc})", float("inf"), 0.0))
            continue
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        us = time_fn(
            lambda cfg=cfg: vector_add_interface(a, b, out=out, config=cfg),
            warmup=warmup,
            iters=iters,
            runs=runs,
        )
        results.append((cfg.name, us, bandwidth_gbps(total_bytes, us)))

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, choices=B200_SHAPES, action="append")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--emit-dispatch", action="store_true")
    parser.add_argument(
        "--eager",
        action="store_true",
        help="Include Python/CUDA launch overhead instead of timing CUDA graph replay.",
    )
    parser.add_argument(
        "--jit-autotune",
        action="store_true",
        help="Include first-use JIT autotuned dispatch over a small candidate set for each size.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    sizes = tuple(args.size) if args.size else B200_SHAPES
    winners = {}
    for size in sizes:
        mode = "eager launches" if args.eager else "CUDA graph replay"
        print(f"\n=== N={size} ({size * size:,} fp16 elements, {mode}) ===")
        results = run_one(
            size,
            args.warmup,
            args.iters,
            args.runs,
            use_graphs=not args.eager,
            include_jit_autotune=args.jit_autotune,
        )
        for name, us, gbps in sorted(results, key=lambda x: x[1]):
            print(f"{name:24s} {us:9.3f} us  {gbps:9.1f} GB/s")
        non_torch = [row for row in results if row[0] != "torch.add"]
        winner_name = min(non_torch, key=lambda x: x[1])[0]
        if winner_name.startswith("jit_autotune:"):
            winner_name = winner_name.split(":", 1)[1]
        winners[size] = winner_name

    if args.emit_dispatch:
        print("\nDEFAULT_DISPATCH = {")
        for size in sizes:
            print(f'    {size}: "{winners[size]}",')
        print("}")


if __name__ == "__main__":
    main()
