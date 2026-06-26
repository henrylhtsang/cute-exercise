"""Dump PTX/SASS for the global atomic-add microbenchmark."""

import os
import subprocess
import sys

DUMP_DIR = os.path.join(os.path.dirname(__file__), "dump")

os.environ["CUTE_DSL_KEEP_PTX"] = "1"
os.environ["CUTE_DSL_KEEP_CUBIN"] = "1"
os.environ.setdefault("CUTE_DSL_DUMP_DIR", "/tmp/cute_dsl_dump")
os.makedirs(os.environ["CUTE_DSL_DUMP_DIR"], exist_ok=True)

import torch  # noqa: E402

import cutlass.cute as cute  # noqa: E402
from cutlass import Int32  # noqa: E402
from cutlass.cute.runtime import from_dlpack  # noqa: E402

from cute_exercise.ex23_global_atomic_add.kernel import GlobalAtomicAdd, VARIANTS  # noqa: E402

NVDISASM = "/usr/local/cuda/bin/nvdisasm"


def dump_one(variant: str) -> None:
    out = torch.zeros(1 if variant == "hot" else 8192, device="cuda", dtype=torch.int32)
    out_ = from_dlpack(out)

    compiled = cute.compile(
        GlobalAtomicAdd(variant=variant, threads=256, iters=16),
        out_,
        Int32(out.numel()),
        152,
    )

    os.makedirs(DUMP_DIR, exist_ok=True)
    stem = os.path.join(DUMP_DIR, variant)
    with open(f"{stem}.ptx", "w") as f:
        f.write(compiled.__ptx__)
    with open(f"{stem}.cubin", "wb") as f:
        f.write(compiled.__cubin__)
    with open(f"{stem}.sass", "w") as f:
        subprocess.run([NVDISASM, f"{stem}.cubin"], check=True, stdout=f)
    print(f"dumped {variant} to {stem}.{{ptx,cubin,sass}}")


def main() -> None:
    for variant in VARIANTS:
        dump_one(variant)


if __name__ == "__main__":
    main()
    sys.exit(0)
