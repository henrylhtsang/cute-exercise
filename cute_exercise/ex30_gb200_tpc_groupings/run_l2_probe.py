"""Run the L2 pointer-chase probe and cluster physical TPC groups."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cute_exercise.ex30_gb200_tpc_groupings.analysis import (  # noqa: E402
    cluster_tpc_signatures,
)


BUILD_DIR = ROOT / "build"
SOURCE = ROOT / "l2_latency_probe.cu"
BINARY = BUILD_DIR / "l2_latency_probe"
DEFAULT_TOPOLOGY = ROOT / "artifacts" / "gb200_tpc_probe_latest.json"


def tool(name: str) -> str:
    found = shutil.which(name)
    if found is not None:
        return found
    return name


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, text=True, **kwargs)


def nvcc_version(nvcc: str) -> str:
    return run([nvcc, "--version"], stdout=subprocess.PIPE).stdout.strip()


def detect_arch() -> str:
    query = ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"]
    try:
        out = run(query, stdout=subprocess.PIPE).stdout.splitlines()[0].strip()
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError):
        return "sm_100"
    match = re.match(r"(\d+)\.(\d+)", out)
    if match is None:
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


def load_smid_to_tpc(path: Path | None, multi_processor_count: int) -> dict[int, int]:
    if path is not None and path.exists():
        payload = json.loads(path.read_text())
        return {
            int(smid): int(tpc)
            for smid, tpc in payload["topology"]["smid_to_tpc"].items()
        }
    return {smid: smid // 2 for smid in range(multi_processor_count)}


def grouping_to_json(grouping) -> dict:
    return {
        "group_counts": list(grouping.group_counts),
        "groups": [list(group) for group in grouping.groups],
        "merge_distances": [round(value, 6) for value in grouping.merge_distances],
    }


def print_summary(payload: dict) -> None:
    device = payload["device"]
    launch = payload["launch"]
    grouping = payload["physical_grouping"]
    print(
        f"device: {device['name']} cc={device['compute_capability']} "
        f"SMs={device['multi_processor_count']} "
        f"working_set={launch['working_set_bytes']} bytes"
    )
    print(
        f"signature: buckets={launch['buckets']} "
        f"samples_per_bucket={launch['samples_per_bucket']} "
        f"stride_words={launch['stride_words']}"
    )
    print(f"group_counts: {grouping['group_counts']}")
    print("groups:")
    for group in grouping["groups"]:
        print("  " + ",".join(str(tpc) for tpc in group))
    print("merge_distances:")
    print("  " + ",".join(str(value) for value in grouping["merge_distances"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nvcc", default=os.environ.get("NVCC", tool("nvcc")))
    parser.add_argument("--arch", default=detect_arch())
    parser.add_argument("--buckets", type=int, default=4096)
    parser.add_argument("--samples-per-bucket", type=int, default=16)
    parser.add_argument("--stride-words", type=int, default=256)
    parser.add_argument("--blocks", type=int, default=0)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--dynamic-smem", type=int, default=0)
    parser.add_argument("--warmup-rounds", type=int, default=1)
    parser.add_argument("--concurrent", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--target-groups", type=int)
    parser.add_argument("--topology", type=Path, default=DEFAULT_TOPOLOGY)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    version = nvcc_version(args.nvcc)
    build(args.nvcc, args.arch)

    cmd = [
        str(BINARY),
        "--buckets",
        str(args.buckets),
        "--samples-per-bucket",
        str(args.samples_per_bucket),
        "--stride-words",
        str(args.stride_words),
        "--threads",
        str(args.threads),
        "--warmup-rounds",
        str(args.warmup_rounds),
        "--seed",
        str(args.seed),
    ]
    if args.blocks:
        cmd += ["--blocks", str(args.blocks)]
    if args.dynamic_smem:
        cmd += ["--dynamic-smem", str(args.dynamic_smem)]
    if args.concurrent:
        cmd += ["--concurrent"]

    probe_payload = json.loads(run(cmd, stdout=subprocess.PIPE).stdout)
    smid_to_tpc = load_smid_to_tpc(args.topology, probe_payload["device"]["multi_processor_count"])
    signatures = {
        int(row["smid"]): tuple(float(value) for value in row["signature"])
        for row in probe_payload["rows"]
    }
    grouping = cluster_tpc_signatures(
        signatures,
        smid_to_tpc,
        target_groups=args.target_groups,
    )
    payload = {
        "nvcc_version": version,
        "arch": args.arch,
        "device": probe_payload["device"],
        "launch": probe_payload["launch"],
        "physical_grouping": grouping_to_json(grouping),
        "rows": probe_payload["rows"],
    }

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print_summary(payload)
        if args.output is not None:
            print()
            print(f"raw JSON: {args.output}")


if __name__ == "__main__":
    main()
