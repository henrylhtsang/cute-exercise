"""Dump PTX/SASS for both ElementwiseAdd variants, per (variant, dtype).

Uses ``compiled.__ptx__`` / ``compiled.__cubin__`` (populated when
``CUTE_DSL_KEEP_PTX=1`` / ``CUTE_DSL_KEEP_CUBIN=1`` are set). Writes
``./dump/{variant}_{dtype}.{ptx,cubin,sass}``.
"""

import os
import subprocess
import sys

DUMP_DIR = os.path.join(os.path.dirname(__file__), "dump")
os.environ["CUTE_DSL_KEEP_PTX"] = "1"
os.environ["CUTE_DSL_KEEP_CUBIN"] = "1"

import torch  # noqa: E402

import cutlass.cute as cute  # noqa: E402
from cutlass.cute.runtime import from_dlpack  # noqa: E402

from cute_exercise.ex1_for_loop.kernel import VARIANTS, ElementwiseAdd  # noqa: E402

NVDISASM = "/usr/local/cuda/bin/nvdisasm"


def dump_one(variant: str, dtype_name: str, dtype: torch.dtype) -> None:
    M, N = 1024, 2048
    a = torch.randn(M, N, device="cuda", dtype=dtype)
    b = torch.randn(M, N, device="cuda", dtype=dtype)
    c = torch.empty_like(a)

    a_ = from_dlpack(a, assumed_align=16)
    b_ = from_dlpack(b, assumed_align=16)
    c_ = from_dlpack(c, assumed_align=16)

    compiled = cute.compile(ElementwiseAdd(variant=variant), a_, b_, c_)

    stem = os.path.join(DUMP_DIR, f"{variant}_{dtype_name}")
    with open(f"{stem}.ptx", "w") as f:
        f.write(compiled.__ptx__)
    with open(f"{stem}.cubin", "wb") as f:
        f.write(compiled.__cubin__)
    with open(f"{stem}.sass", "w") as f:
        subprocess.run([NVDISASM, f"{stem}.cubin"], check=True, stdout=f)


def main() -> None:
    os.makedirs(DUMP_DIR, exist_ok=True)
    for dtype_name, dtype in [("fp16", torch.float16), ("fp32", torch.float32)]:
        for v in VARIANTS:
            dump_one(v, dtype_name, dtype)
            print(f"dumped variant={v} dtype={dtype_name}")
    print(f"\nOutput: {DUMP_DIR}")


if __name__ == "__main__":
    main()
    sys.exit(0)
