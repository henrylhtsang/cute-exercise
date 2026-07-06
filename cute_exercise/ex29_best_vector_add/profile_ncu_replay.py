"""Replay one VectorAdd or torch.add CUDA graph under Nsight Compute."""

from __future__ import annotations

import argparse

import torch

from cute_exercise.ex29_best_vector_add.kernel import (
    B200_SHAPES,
    CONFIG_BY_NAME,
    dispatch_config,
    _device_sm_count,
    vector_add_interface,
)


def _capture_graph(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()
    return graph


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("torch", "vector"), required=True)
    parser.add_argument("--size", type=int, choices=B200_SHAPES, default=16384)
    parser.add_argument("--config", choices=tuple(CONFIG_BY_NAME), default=None)
    parser.add_argument("--replays", type=int, default=3)
    args = parser.parse_args()

    torch.manual_seed(123)
    a = torch.randn(args.size, args.size, device="cuda", dtype=torch.float16)
    b = torch.randn(args.size, args.size, device="cuda", dtype=torch.float16)
    out = torch.empty_like(a)
    expected = a + b

    if args.mode == "torch":
        label = f"torch_add_N{args.size}"

        def fn():
            torch.add(a, b, out=out)

    else:
        cfg = CONFIG_BY_NAME[args.config] if args.config else dispatch_config(args.size)
        label = f"vector_add_{cfg.name}_N{args.size}"

        def fn():
            vector_add_interface(a, b, out=out, config=cfg)

    fn()
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    graph = _capture_graph(fn)

    print(f"device={torch.cuda.get_device_name()} sm_count={_device_sm_count()}")
    print(f"profile_label={label}")
    torch.cuda.cudart().cudaProfilerStart()
    for i in range(args.replays):
        torch.cuda.nvtx.range_push(f"{label}_replay_{i}")
        graph.replay()
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    torch.testing.assert_close(out, expected, rtol=0, atol=0)


if __name__ == "__main__":
    main()
