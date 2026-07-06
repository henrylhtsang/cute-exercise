"""Dump PTX and SASS for selected best-vector-add configs."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

os.environ["CUTE_DSL_KEEP_PTX"] = "1"
os.environ["CUTE_DSL_KEEP_CUBIN"] = "1"

import torch

import cutlass.torch as cutlass_torch
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from cute_exercise.ex29_best_vector_add.kernel import (
    B200_SHAPES,
    CONFIG_BY_NAME,
    VectorAdd,
    _num_ctas,
    _tile_count,
    _validate_launch_config,
)


ROOT = Path(__file__).resolve().parent
NVDISASM = os.environ.get("NVDISASM", "/usr/local/cuda/bin/nvdisasm")


def dump(size: int, config_name: str) -> Path:
    cfg = CONFIG_BY_NAME[config_name]
    a = torch.randn(size, size, device="cuda", dtype=torch.float16).reshape(-1)
    b = torch.randn(size, size, device="cuda", dtype=torch.float16).reshape(-1)
    c = torch.empty_like(a)
    if a.data_ptr() % cfg.assumed_align or b.data_ptr() % cfg.assumed_align or c.data_ptr() % cfg.assumed_align:
        raise ValueError(f"{config_name} requires {cfg.assumed_align}-byte aligned tensors")
    a_ = from_dlpack(a, assumed_align=cfg.assumed_align)
    b_ = from_dlpack(b, assumed_align=cfg.assumed_align)
    c_ = from_dlpack(c, assumed_align=cfg.assumed_align)
    tile_count = _tile_count(a.numel(), cfg)
    num_ctas = _num_ctas(tile_count, cfg)
    assume_full_tiles = 1 if cfg.assume_full_tiles else 0
    _validate_launch_config(tile_count, num_ctas, cfg)
    stream = cutlass_torch.current_stream()

    compiled = cute.compile[(cute.KeepPTX, cute.KeepCUBIN)](
        VectorAdd(cfg, num_ctas, tile_count, assume_full_tiles), a_, b_, c_, stream
    )

    out_dir = ROOT / "dump"
    out_dir.mkdir(exist_ok=True)
    stem = out_dir / f"{config_name}_{size}"
    (stem.with_suffix(".ptx")).write_text(compiled.__ptx__)
    stem.with_suffix(".cubin").write_bytes(compiled.__cubin__)
    with stem.with_suffix(".sass").open("w") as f:
        subprocess.run([NVDISASM, str(stem.with_suffix(".cubin"))], check=True, stdout=f)
    ptx = stem.with_suffix(".ptx").read_text()
    sass = stem.with_suffix(".sass").read_text()
    stem.with_suffix(".txt").write_text(
        f"# VectorAdd assembly dump\n\n"
        f"config: {config_name}\n"
        f"size: {size}x{size}\n"
        f"raw PTX: {stem.with_suffix('.ptx')}\n"
        f"raw SASS: {stem.with_suffix('.sass')}\n\n"
        f"## PTX\n\n"
        f"```ptx\n{ptx}\n```\n\n"
        f"## SASS\n\n"
        f"```text\n{sass}\n```\n"
    )
    print(f"dumped {stem}.{{ptx,cubin,sass,txt}}")
    return stem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, choices=B200_SHAPES, default=2048)
    parser.add_argument("--config", choices=CONFIG_BY_NAME, default="ptx_h2_v4_fma_mix")
    args = parser.parse_args()
    dump(args.size, args.config)


if __name__ == "__main__":
    main()
