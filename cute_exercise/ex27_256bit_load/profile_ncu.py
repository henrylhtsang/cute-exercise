"""Profile ex27 load variants with Nsight Compute."""

import argparse
import os
import subprocess
from pathlib import Path

from build import BINARY, build, detect_arch, tool


METRICS = (
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,"
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,"
    "smsp__sass_inst_executed_op_global_ld.sum"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nvcc", default=os.environ.get("NVCC", tool("nvcc")))
    parser.add_argument("--ncu", default=os.environ.get("NCU", tool("ncu")))
    parser.add_argument("--arch", default=detect_arch())
    parser.add_argument("--n", type=int, default=1 << 26)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "build")
    args = parser.parse_args()

    build(args.nvcc, args.arch)
    args.output_dir.mkdir(exist_ok=True)

    for variant in ("u32", "v4_u32", "v8_u32"):
        report = args.output_dir / f"ncu_{variant}"
        cmd = [
            args.ncu,
            "--target-processes",
            "all",
            "--metrics",
            METRICS,
            "--kernel-name",
            f"regex:copy_{variant}_kernel",
            "--export",
            str(report),
            "--force-overwrite",
            str(BINARY),
            "--variant",
            variant,
            "--n",
            str(args.n),
            "--iters",
            str(args.iters),
        ]
        subprocess.run(cmd, check=True)
        print(f"{variant}: {report}.ncu-rep")


if __name__ == "__main__":
    main()
