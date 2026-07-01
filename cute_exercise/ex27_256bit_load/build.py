"""Build and optionally run the ex27 native CUDA probe."""

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BUILD_DIR = ROOT / "build"
SOURCE = ROOT / "load_probe.cu"
BINARY = BUILD_DIR / "load_probe"


def tool(name: str) -> str:
    found = shutil.which(name)
    if found is not None:
        return found
    cuda_tool = Path("/usr/local/cuda/bin") / name
    if cuda_tool.exists():
        return str(cuda_tool)
    return name


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, text=True, **kwargs)


def nvcc_version(nvcc: str) -> str:
    return run([nvcc, "--version"], stdout=subprocess.PIPE).stdout.strip()


def detect_arch() -> str:
    query = [
        "nvidia-smi",
        "--query-gpu=compute_cap",
        "--format=csv,noheader",
    ]
    try:
        out = run(query, stdout=subprocess.PIPE).stdout.splitlines()[0].strip()
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError):
        return "sm_100"
    match = re.match(r"(\d+)\.(\d+)", out)
    if not match:
        return "sm_100"
    return f"sm_{match.group(1)}{match.group(2)}"


def build(nvcc: str, arch: str) -> None:
    BUILD_DIR.mkdir(exist_ok=True)
    cmd = [
        nvcc,
        f"-arch={arch}",
        "-O3",
        "--std=c++17",
        "-lineinfo",
        "--ptxas-options=-v",
        str(SOURCE),
        "-o",
        str(BINARY),
    ]
    run(cmd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nvcc", default=os.environ.get("NVCC", tool("nvcc")))
    parser.add_argument("--arch", default=detect_arch())
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--n", type=int, default=1 << 26)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--variant", default="all")
    args = parser.parse_args()

    version = nvcc_version(args.nvcc)
    build(args.nvcc, args.arch)

    if not args.run:
        print(BINARY)
        return

    probe = run(
        [
            str(BINARY),
            "--variant",
            args.variant,
            "--n",
            str(args.n),
            "--iters",
            str(args.iters),
        ],
        stdout=subprocess.PIPE,
    )
    payload = json.loads(probe.stdout)
    payload["nvcc_version"] = version
    payload["arch"] = args.arch
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
