"""Dump PTX and SASS for the ex27 native CUDA probe."""

import argparse
import os
import subprocess
from pathlib import Path

from build import BINARY, BUILD_DIR, ROOT, SOURCE, build, detect_arch, tool


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nvcc", default=os.environ.get("NVCC", tool("nvcc")))
    parser.add_argument(
        "--nvdisasm", default=os.environ.get("NVDISASM", tool("nvdisasm"))
    )
    parser.add_argument("--arch", default=detect_arch())
    args = parser.parse_args()

    build(args.nvcc, args.arch)
    ptx = BUILD_DIR / "load_probe.ptx"
    cubin = BUILD_DIR / "load_probe.cubin"
    sass = BUILD_DIR / "load_probe.sass"
    subprocess.run(
        [
            args.nvcc,
            f"-arch={args.arch}",
            "-O3",
            "--std=c++17",
            "-lineinfo",
            "-ptx",
            str(SOURCE),
            "-o",
            str(ptx),
        ],
        check=True,
    )
    subprocess.run(
        [
            args.nvcc,
            f"-arch={args.arch}",
            "-O3",
            "--std=c++17",
            "-lineinfo",
            "-cubin",
            str(SOURCE),
            "-o",
            str(cubin),
        ],
        check=True,
    )
    with sass.open("w") as f:
        subprocess.run([args.nvdisasm, str(cubin)], check=True, stdout=f)
    print(f"PTX:  {ptx.relative_to(ROOT)}")
    print(f"CUBIN: {cubin.relative_to(ROOT)}")
    print(f"SASS: {sass.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
