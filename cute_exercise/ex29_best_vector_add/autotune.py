"""Focused CUDA-graph autotuning for VectorAdd configs."""

from __future__ import annotations

import argparse
import itertools
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch

from cute_exercise.benchmark import bandwidth_gbps
from cute_exercise.ex29_best_vector_add.bench import cuda_graph_benchmark
from cute_exercise.ex29_best_vector_add.kernel import (
    B200_SHAPES,
    VectorAddConfig,
    _device_sm_count,
    vector_add_interface,
)


@dataclass(frozen=True)
class AutotuneMeasurement:
    us: float
    config: VectorAddConfig


@dataclass(frozen=True)
class OptionAnalysis:
    keep: set[object]
    drop: set[object]
    top_counts: dict[int, Counter]


def _valid(cfg: VectorAddConfig) -> bool:
    words_per_thread = cfg.elems_per_thread // 2
    return (
        cfg.elems_per_thread % 2 == 0
        and cfg.vector_words in (1, 4, 8)
        and cfg.cache_policy in ("default", "cs", "noalloc", "ca", "cg")
        and cfg.store_policy in ("default", "cs", "noalloc", "wt", "wb", "cg")
        and words_per_thread >= cfg.vector_words
        and words_per_thread % cfg.vector_words == 0
        and (words_per_thread // cfg.vector_words) % cfg.unroll == 0
        and (cfg.unroll != 4 or cfg.vector_words == 1)
        and cfg.threads <= 1024
        and cfg.assumed_align in (16, 32, 64, 128)
        and cfg.op_order in ("interleaved", "loads_first", "bundled_scalar")
        and (cfg.op_order == "interleaved" or (cfg.vector_words == 1 and cfg.unroll == 4))
    )


def _dedupe(configs: list[VectorAddConfig]) -> list[VectorAddConfig]:
    out = []
    seen = set()
    for cfg in configs:
        key = (
            cfg.variant,
            cfg.threads,
            cfg.cta_per_sm,
            cfg.elems_per_thread,
            cfg.vector_words,
            cfg.cache_policy,
            cfg.store_policy,
            cfg.math_op,
            cfg.unroll,
            cfg.schedule,
            cfg.tiles_per_cta,
            cfg.assume_full_tiles,
            cfg.assumed_align,
            cfg.op_order,
        )
        if key not in seen and _valid(cfg):
            seen.add(key)
            out.append(cfg)
    return out


def generate_autotune_configs() -> list[VectorAddConfig]:
    """Return the focused grid used to compete with ``torch.add``.

    The grid is intentionally centered on variants that were close in early
    runs: 128-512 thread PTX v8/v4 paths, several persistent CTA counts, and
    the one-tile/fixed-tile scheduling alternatives requested in the exercise.
    A smaller DSL grid stays in the search because DSL vectorization has been
    surprisingly competitive on this memory-bound kernel.
    """
    configs: list[VectorAddConfig] = []

    ptx_threads = (128, 256, 512)
    ptx_elems = (8, 16, 32, 64)
    ptx_vectors = (1, 4, 8)
    policy_pairs = (
        ("default", "default"),
        ("cs", "cs"),
        ("noalloc", "noalloc"),
        ("noalloc", "default"),
        ("default", "wt"),
        ("ca", "wb"),
        ("cg", "cg"),
        ("cs", "wt"),
    )
    math_ops = ("add", "fma", "mix")
    unrolls = (1, 2, 4)
    cta_per_sms = (1, 2, 3, 4, 6, 8)
    assume_full_tile_options = (False, True)
    fixed_tiles = (2, 4, 8)
    alignments = (16, 32, 64, 128)

    def op_orders(vec: int, unroll: int) -> tuple[str, ...]:
        if vec == 1 and unroll == 4:
            return ("interleaved", "loads_first", "bundled_scalar")
        return ("interleaved",)

    for threads, elems, vec, policy, math_op, unroll, cta_per_sm, assume_full_tiles, align in itertools.product(
        ptx_threads,
        ptx_elems,
        ptx_vectors,
        policy_pairs,
        math_ops,
        unrolls,
        cta_per_sms,
        assume_full_tile_options,
        alignments,
    ):
        cache, store = policy
        for op_order in op_orders(vec, unroll):
            configs.append(
                VectorAddConfig(
                    f"auto_p_t{threads}_c{cta_per_sm}_e{elems}_v{vec}_{cache}_st{store}_{math_op}_u{unroll}"
                    f"_a{align}{'_full' if assume_full_tiles else ''}"
                    f"{'' if op_order == 'interleaved' else '_' + op_order}",
                    "ptx",
                    threads,
                    cta_per_sm,
                    elems,
                    vec,
                    cache_policy=cache,
                    store_policy=store,
                    math_op=math_op,
                    unroll=unroll,
                    schedule="persistent",
                    assume_full_tiles=assume_full_tiles,
                    assumed_align=align,
                    op_order=op_order,
                )
            )

    for threads, elems, vec, policy, math_op, unroll, align in itertools.product(
        ptx_threads,
        ptx_elems,
        ptx_vectors,
        policy_pairs,
        math_ops,
        unrolls,
        alignments,
    ):
        cache, store = policy
        for op_order in op_orders(vec, unroll):
            configs.append(
                VectorAddConfig(
                    f"auto_one_t{threads}_e{elems}_v{vec}_{cache}_st{store}_{math_op}_u{unroll}_a{align}"
                    f"{'' if op_order == 'interleaved' else '_' + op_order}",
                    "ptx",
                    threads,
                    4,
                    elems,
                    vec,
                    cache_policy=cache,
                    store_policy=store,
                    math_op=math_op,
                    unroll=unroll,
                    schedule="one_tile",
                    assumed_align=align,
                    op_order=op_order,
                )
            )
            for tiles_per_cta in fixed_tiles:
                configs.append(
                    VectorAddConfig(
                        f"auto_fix{tiles_per_cta}_t{threads}_e{elems}_v{vec}_{cache}_st{store}_{math_op}_u{unroll}_a{align}"
                        f"{'' if op_order == 'interleaved' else '_' + op_order}",
                        "ptx",
                        threads,
                        4,
                        elems,
                        vec,
                        cache_policy=cache,
                        store_policy=store,
                        math_op=math_op,
                        unroll=unroll,
                        schedule="fixed_tiles",
                        tiles_per_cta=tiles_per_cta,
                        assumed_align=align,
                        op_order=op_order,
                    )
                )

    for threads, elems, cta_per_sm, assume_full_tiles in itertools.product(
        (128, 256, 512), (8, 16, 32, 64), cta_per_sms, assume_full_tile_options
    ):
        configs.append(
            VectorAddConfig(
                f"auto_dsl_p_t{threads}_c{cta_per_sm}_e{elems}"
                f"{'_full' if assume_full_tiles else ''}",
                "dsl",
                threads,
                cta_per_sm,
                elems,
                4,
                schedule="persistent",
                assume_full_tiles=assume_full_tiles,
                assumed_align=16,
            )
        )

    for threads, elems in itertools.product((128, 256, 512), (8, 16, 32, 64)):
        configs.append(
            VectorAddConfig(
                f"auto_dsl_one_t{threads}_e{elems}",
                "dsl",
                threads,
                4,
                elems,
                4,
                schedule="one_tile",
                assumed_align=16,
            )
        )
        for tiles_per_cta in fixed_tiles:
            configs.append(
                VectorAddConfig(
                    f"auto_dsl_fix{tiles_per_cta}_t{threads}_e{elems}",
                    "dsl",
                    threads,
                    4,
                    elems,
                    4,
                    schedule="fixed_tiles",
                    tiles_per_cta=tiles_per_cta,
                    assumed_align=16,
                )
            )

    return _dedupe(configs)


def _option_value(cfg: VectorAddConfig, field: str) -> object:
    if field == "variant":
        return cfg.variant
    if field == "threads":
        return cfg.threads
    if field == "cta_per_sm":
        return cfg.cta_per_sm
    if field == "elems_per_thread":
        return cfg.elems_per_thread
    if field == "vector_words":
        return cfg.vector_words
    if field == "cache_policy":
        return cfg.cache_policy
    if field == "store_policy":
        return cfg.store_policy
    if field == "math_op":
        return cfg.math_op
    if field == "unroll":
        return cfg.unroll
    if field == "schedule":
        return cfg.schedule
    if field == "tiles_per_cta":
        return cfg.tiles_per_cta
    if field == "assume_full_tiles":
        return cfg.assume_full_tiles
    if field == "assumed_align":
        return cfg.assumed_align
    if field == "op_order":
        return cfg.op_order
    raise ValueError(f"unknown field {field}")


def analyze_top_fraction(
    measurements_by_size: dict[int, list[AutotuneMeasurement]],
    top_fraction: float = 0.10,
) -> dict[str, OptionAnalysis]:
    fields = (
        "variant",
        "threads",
        "cta_per_sm",
        "elems_per_thread",
        "vector_words",
        "cache_policy",
        "store_policy",
        "math_op",
        "unroll",
        "schedule",
        "tiles_per_cta",
        "assume_full_tiles",
        "assumed_align",
        "op_order",
    )
    all_values = {
        field: {
            _option_value(measurement.config, field)
            for measurements in measurements_by_size.values()
            for measurement in measurements
        }
        for field in fields
    }
    top_values_by_field: dict[str, dict[int, Counter]] = {field: {} for field in fields}

    for size, measurements in measurements_by_size.items():
        ranked = sorted(measurements, key=lambda row: row.us)
        top_n = max(1, int(len(ranked) * top_fraction + 0.999999))
        top = ranked[:top_n]
        for field in fields:
            top_values_by_field[field][size] = Counter(_option_value(row.config, field) for row in top)

    analysis = {}
    for field in fields:
        if not measurements_by_size:
            keep = set()
        else:
            per_size_values = [set(counter) for counter in top_values_by_field[field].values()]
            keep = set.intersection(*per_size_values) if per_size_values else set()
        analysis[field] = OptionAnalysis(
            keep=keep,
            drop=all_values[field] - keep,
            top_counts=top_values_by_field[field],
        )
    return analysis


def _format_config(cfg: VectorAddConfig) -> str:
    args = [
        repr(cfg.name),
        repr(cfg.variant),
        str(cfg.threads),
        str(cfg.cta_per_sm),
        str(cfg.elems_per_thread),
        str(cfg.vector_words),
    ]
    kwargs = []
    if cfg.cache_policy != "default":
        kwargs.append(f"cache_policy={cfg.cache_policy!r}")
    if cfg.store_policy != "default":
        kwargs.append(f"store_policy={cfg.store_policy!r}")
    if cfg.math_op != "add":
        kwargs.append(f"math_op={cfg.math_op!r}")
    if cfg.unroll != 1:
        kwargs.append(f"unroll={cfg.unroll}")
    if cfg.schedule != "persistent":
        kwargs.append(f"schedule={cfg.schedule!r}")
    if cfg.tiles_per_cta != 1:
        kwargs.append(f"tiles_per_cta={cfg.tiles_per_cta}")
    if cfg.assume_full_tiles:
        kwargs.append("assume_full_tiles=True")
    if cfg.assumed_align != 16:
        kwargs.append(f"assumed_align={cfg.assumed_align}")
    if cfg.op_order != "interleaved":
        kwargs.append(f"op_order={cfg.op_order!r}")
    return f"VectorAddConfig({', '.join(args + kwargs)})"


def run_autotune(
    sizes: tuple[int, ...],
    warmup: int,
    iters: int,
    runs: int,
    top_k: int,
    limit: int | None,
) -> str:
    configs = generate_autotune_configs()
    if limit is not None:
        configs = configs[:limit]

    lines = [
        "# VectorAdd CUDA graph autotune",
        "",
        f"configs: {len(configs)}",
        f"device: {torch.cuda.get_device_name()}",
        f"SM count: {_device_sm_count()}",
        f"warmup={warmup} iters={iters} runs={runs}",
        "",
    ]
    winners: dict[int, VectorAddConfig] = {}
    measurements_by_size: dict[int, list[AutotuneMeasurement]] = {}

    for size in sizes:
        torch.manual_seed(123)
        a = torch.randn(size, size, device="cuda", dtype=torch.float16)
        b = torch.randn(size, size, device="cuda", dtype=torch.float16)
        out = torch.empty_like(a)
        expected = a + b
        total_bytes = (a.numel() + b.numel() + out.numel()) * a.element_size()

        torch.add(a, b, out=out)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        torch_us = cuda_graph_benchmark(
            lambda: torch.add(a, b, out=out), warmup=warmup, iters=iters, runs=runs
        )

        lines.append(f"## N={size}")
        lines.append(f"torch.add {torch_us:.3f} us {bandwidth_gbps(total_bytes, torch_us):.1f} GB/s")

        measurements: list[AutotuneMeasurement] = []
        start = time.time()
        for idx, cfg in enumerate(configs, 1):
            try:
                vector_add_interface(a, b, out=out, config=cfg)
                torch.testing.assert_close(out, expected, rtol=0, atol=0)
                us = cuda_graph_benchmark(
                    lambda cfg=cfg: vector_add_interface(a, b, out=out, config=cfg),
                    warmup=warmup,
                    iters=iters,
                    runs=runs,
                )
            except Exception as exc:
                lines.append(f"FAIL {cfg.name}: {type(exc).__name__}: {exc}")
                continue

            measurements.append(AutotuneMeasurement(us, cfg))
            best = sorted(measurements, key=lambda row: row.us)[:top_k]
            if idx % 100 == 0:
                current = best[0]
                print(
                    f"N={size} {idx}/{len(configs)} elapsed={time.time() - start:.1f}s "
                    f"best={current.us:.3f}us {current.config.name}",
                    flush=True,
                )

        measurements_by_size[size] = measurements
        best = sorted(measurements, key=lambda row: row.us)[:top_k]
        winners[size] = best[0].config
        for measurement in best:
            us = measurement.us
            cfg = measurement.config
            delta = (us / torch_us - 1.0) * 100.0
            lines.append(
                f"{us:.3f} us {bandwidth_gbps(total_bytes, us):.1f} GB/s "
                f"{delta:+.2f}% vs torch.add {cfg.name} "
                f"schedule={cfg.schedule} tiles_per_cta={cfg.tiles_per_cta} cta_per_sm={cfg.cta_per_sm} "
                f"load={cfg.cache_policy} store={cfg.store_policy} align={cfg.assumed_align} "
                f"order={cfg.op_order}"
            )
            lines.append(f"    {_format_config(cfg)}")
        lines.append("")

    analysis = analyze_top_fraction(measurements_by_size, top_fraction=0.10)
    lines.append("## Top 10% Option Analysis")
    lines.append("Values listed under keep appeared in the top 10% for every measured size.")
    for field, result in analysis.items():
        keep = ", ".join(str(value) for value in sorted(result.keep, key=str)) or "<none>"
        drop = ", ".join(str(value) for value in sorted(result.drop, key=str)) or "<none>"
        lines.append(f"{field}: keep {{{keep}}}; drop {{{drop}}}")
        for size in sizes:
            counts = ", ".join(
                f"{value}:{count}" for value, count in sorted(result.top_counts[size].items(), key=lambda row: str(row[0]))
            )
            lines.append(f"    N={size}: {counts}")
    lines.append("")

    lines.append("DEFAULT_DISPATCH = {")
    for size in sizes:
        lines.append(f'    {size}: "{winners[size].name}",')
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, choices=B200_SHAPES, action="append")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    sizes = tuple(args.size) if args.size else B200_SHAPES
    report = run_autotune(sizes, args.warmup, args.iters, args.runs, args.top_k, args.limit)
    print(report)
    if args.output is not None:
        args.output.write_text(report + "\n")


if __name__ == "__main__":
    main()
