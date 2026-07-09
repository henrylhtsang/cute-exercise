import argparse
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "cuda_graph_if_node_demo.cu"
BUILD_DIR = ROOT / "build"
BINARY = BUILD_DIR / "cuda_graph_if_node_demo"


def find_nvcc() -> str:
    cuda_home = os.environ.get("CUDA_HOME")
    candidates = []
    if cuda_home:
        candidates.append(Path(cuda_home) / "bin" / "nvcc")
    candidates.extend(
        [
            Path("/usr/local/cuda/bin/nvcc"),
            Path("/usr/local/cuda-13.0/bin/nvcc"),
            Path("/usr/local/cuda-12.8/bin/nvcc"),
        ]
    )

    path_nvcc = shutil.which("nvcc")
    if path_nvcc:
        candidates.insert(0, Path(path_nvcc))

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    raise SystemExit(
        "Could not find nvcc. Set CUDA_HOME or install CUDA 12.8+; "
        "conditional graph nodes require recent CUDA headers."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run the compiled CUDA Graph IF-node demo after building it.",
    )
    parser.add_argument(
        "--run-pytorch",
        action="store_true",
        help="Run the PyTorch torch.cond CUDA Graph IF-node demo.",
    )
    parser.add_argument(
        "--arch",
        default="sm_100",
        help="CUDA architecture passed to nvcc, e.g. sm_90 or sm_100.",
    )
    args = parser.parse_args()

    BUILD_DIR.mkdir(exist_ok=True)
    nvcc = find_nvcc()
    command = [
        nvcc,
        "-std=c++17",
        f"-arch={args.arch}",
        str(SOURCE),
        "-o",
        str(BINARY),
    ]
    print(" ".join(command))
    subprocess.run(command, check=True)

    if args.run:
        subprocess.run([str(BINARY)], check=True)

    if args.run_pytorch:
        subprocess.run(
            ["python", str(ROOT / "pytorch_if_node_demo.py")],
            check=True,
        )


if __name__ == "__main__":
    main()
