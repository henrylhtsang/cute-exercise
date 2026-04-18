"""Dump PTX/SASS for both ElementwiseAdd variants, per (variant, dtype).

Writes files into ./dump/{variant}_{dtype}.{ptx,cubin,sass}.
"""

import glob
import os
import shutil
import subprocess
import sys

DUMP_DIR = os.path.join(os.path.dirname(__file__), "dump")
RAW_DIR = os.path.join(DUMP_DIR, "_raw")

os.environ["CUTE_DSL_KEEP_PTX"] = "1"
os.environ["CUTE_DSL_KEEP_CUBIN"] = "1"
os.environ["CUTE_DSL_DUMP_DIR"] = RAW_DIR

import torch  # noqa: E402

from cute_exercise.ex1_for_loop.kernel import VARIANTS, elementwise_add_interface  # noqa: E402

NVDISASM = "/usr/local/cuda/bin/nvdisasm"


def dump_one(variant: str, dtype_name: str, dtype: torch.dtype) -> None:
    if os.path.isdir(RAW_DIR):
        shutil.rmtree(RAW_DIR)
    os.makedirs(RAW_DIR, exist_ok=True)

    M, N = 1024, 2048
    a = torch.randn(M, N, device="cuda", dtype=dtype)
    b = torch.randn(M, N, device="cuda", dtype=dtype)
    c = torch.empty_like(a)
    elementwise_add_interface(a, b, out=c, variant=variant)

    for ext in ("ptx", "cubin"):
        matches = glob.glob(os.path.join(RAW_DIR, f"*.{ext}"))
        assert len(matches) == 1, f"expected 1 {ext} file, got {matches}"
        dst = os.path.join(DUMP_DIR, f"{variant}_{dtype_name}.{ext}")
        shutil.copyfile(matches[0], dst)

    cubin = os.path.join(DUMP_DIR, f"{variant}_{dtype_name}.cubin")
    sass = os.path.join(DUMP_DIR, f"{variant}_{dtype_name}.sass")
    with open(sass, "w") as f:
        subprocess.run([NVDISASM, cubin], check=True, stdout=f)


def main() -> None:
    os.makedirs(DUMP_DIR, exist_ok=True)
    from cute_exercise.ex1_for_loop.kernel import _compile_cache
    for dtype_name, dtype in [("fp16", torch.float16), ("fp32", torch.float32)]:
        for v in VARIANTS:
            _compile_cache.clear()
            dump_one(v, dtype_name, dtype)
            print(f"dumped variant={v} dtype={dtype_name}")
    shutil.rmtree(RAW_DIR, ignore_errors=True)
    print(f"\nOutput: {DUMP_DIR}")


if __name__ == "__main__":
    main()
    sys.exit(0)
