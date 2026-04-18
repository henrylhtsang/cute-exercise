"""Shared benchmarking helpers."""

import statistics

import torch


def benchmark(fn, warmup: int = 5, iters: int = 50, runs: int = 3) -> float:
    """Time ``fn()`` and return microseconds.

    Each of ``runs`` runs collects ``iters`` per-call timings and takes their
    median. Returns the median of those run medians (median-of-medians).
    """
    for _ in range(warmup):
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
            fn()
            end.record()
        torch.cuda.synchronize()
        times_us = [s.elapsed_time(e) * 1000 for s, e in events]
        run_medians.append(statistics.median(times_us))
    return statistics.median(run_medians)


def bandwidth_gbps(total_bytes: int, us: float) -> float:
    return total_bytes / (us * 1000)
