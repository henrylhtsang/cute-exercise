"""Compile ex1's ``ElementwiseAdd`` once via CuTe DSL and dump the PTX
+ a sidecar manifest describing how to launch it.

The DSL source stays the source of truth: re-run this script whenever the
DSL kernel or its tile geometry changes, and the shipped ``.ptx`` /
``.manifest.json`` are refreshed.

Output (in ``ex5_ship_pure_ptx/artifacts/``):

  elementwise_add_<dtype>_<M>x<N>_sm<arch>.ptx
  elementwise_add_<dtype>_<M>x<N>_sm<arch>.manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

# ``cute.compile[(KeepPTX,)]`` writes a side-effect ``.ptx`` to
# ``CUTE_DSL_DUMP_DIR`` (default: cwd). We don't need it — we read PTX
# straight off ``compiled.__ptx__`` — so redirect to a temp dir.
os.environ.setdefault("CUTE_DSL_DUMP_DIR", tempfile.gettempdir())

import torch

import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from cute_exercise.ex1_for_loop.kernel import ElementwiseAdd
from cute_exercise.ex5_ship_pure_ptx._common import (
    ARTIFACTS_DIR,
    DTYPES,
    artifact_stem,
    sm_arch_tag,
)


def _max_supported_ptx_version() -> str:
    """Highest PTX ISA version the CUDA driver on this host accepts.

    The DSL's bundled ``ptxas`` can be newer than the system driver; the
    driver then refuses the dumped PTX with
    ``CUDA_ERROR_UNSUPPORTED_PTX_VERSION``. PTX ISA vs CUDA: 12.x → 8.x,
    13.x → 9.x (NVIDIA PTX ISA release notes).
    """
    import cuda.bindings.driver as cu

    err, ver = cu.cuDriverGetVersion()
    assert err == cu.CUresult.CUDA_SUCCESS, err
    major = ver // 1000
    minor = (ver % 1000) // 10
    return f"{major - 4}.{minor}"


def _rewrite_ptx_version(ptx: str, target_version: str) -> str:
    return re.sub(
        r"^\.version\s+\d+\.\d+",
        f".version {target_version}",
        ptx,
        count=1,
        flags=re.MULTILINE,
    )


def dump(dtype_name: str, m: int, n: int, ptx_version: str | None = None) -> Path:
    dtype = DTYPES[dtype_name]

    a = torch.empty(m, n, dtype=dtype, device="cuda")
    b = torch.empty(m, n, dtype=dtype, device="cuda")
    c = torch.empty(m, n, dtype=dtype, device="cuda")
    a_ = from_dlpack(a, assumed_align=16)
    b_ = from_dlpack(b, assumed_align=16)
    c_ = from_dlpack(c, assumed_align=16)

    op = ElementwiseAdd(variant="vectorized")
    compiled = cute.compile[(cute.KeepPTX,)](op, a_, b_, c_)

    ptx = compiled.__ptx__
    if ptx is None:
        raise RuntimeError("KeepPTX did not produce __ptx__; check cute.compile options.")
    target_version = ptx_version if ptx_version is not None else _max_supported_ptx_version()
    ptx = _rewrite_ptx_version(ptx, target_version)

    if len(compiled.kernel_info) != 1:
        raise RuntimeError(f"Expected exactly 1 kernel; got {list(compiled.kernel_info)}")
    kernel_name = next(iter(compiled.kernel_info))

    # Tile geometry must match ``ElementwiseAdd``'s launch logic in
    # ``cute_exercise/ex1_for_loop/kernel.py``.
    tile_m = op.thr_layout_shape[0] * op.val_layout_rows
    tile_n = op.thr_layout_shape[1] * (op.coalesced_ldst_bytes // dtype.itemsize)
    if m % tile_m or n % tile_n:
        raise ValueError(f"shape ({m}, {n}) not divisible by tile ({tile_m}, {tile_n})")

    arch = sm_arch_tag()
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    stem = artifact_stem(dtype_name, (m, n), arch)
    ptx_path = ARTIFACTS_DIR / f"{stem}.ptx"
    manifest_path = ARTIFACTS_DIR / f"{stem}.manifest.json"

    ptx_path.write_text(ptx)
    manifest = {
        "kernel_name": kernel_name,
        "dtype": dtype_name,
        "arch": arch,
        "shape": [m, n],
        "grid": [(m * n) // (tile_m * tile_n), 1, 1],
        "block": [op.thr_layout_shape[0] * op.thr_layout_shape[1], 1, 1],
        "smem_bytes": 0,
        # ABI: 3 raw device pointers (a, b, c). Static shape ⇒ shape and
        # strides are baked into the cubin as constants.
        "params": [
            {"name": "a_ptr", "ctype": "void_p"},
            {"name": "b_ptr", "ctype": "void_p"},
            {"name": "c_ptr", "ctype": "void_p"},
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return ptx_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=list(DTYPES), nargs="+", default=["fp16", "fp32"])
    parser.add_argument("--shape", nargs=2, type=int, default=[16384, 8192], metavar=("M", "N"))
    parser.add_argument(
        "--ptx-version",
        default=None,
        help="Target PTX ISA version (e.g. '8.4' for CUDA 12.4 drivers). "
             "Defaults to the build host's driver ceiling, which only works "
             "if the deployment driver is at least as new as the build host's.",
    )
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        return 1

    m, n = args.shape
    for dt in args.dtype:
        print(f"wrote {dump(dt, m, n, ptx_version=args.ptx_version)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
