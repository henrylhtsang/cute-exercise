"""Build, run, and analyze the GB200 TPC grouping probe."""

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
    infer_topology,
    observations_from_rows,
    smid_inventory,
)


BUILD_DIR = ROOT / "build"
SOURCE = ROOT / "tpc_probe.cu"
BINARY = BUILD_DIR / "tpc_probe"


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


def parse_cluster_sizes(value: str) -> list[int]:
    sizes = [int(part) for part in value.split(",") if part]
    if not sizes or any(size < 1 for size in sizes):
        raise argparse.ArgumentTypeError("cluster sizes must be positive integers")
    return sizes


def topology_to_json(topology) -> dict:
    return {
        "tpc_groups": [list(group) for group in topology.tpc_groups],
        "cluster_groups": {
            str(size): [list(group) for group in groups]
            for size, groups in topology.cluster_groups.items()
        },
        "cluster_group_observation_counts": {
            str(size): {
                ",".join(str(smid) for smid in group): count
                for group, count in counts.items()
            }
            for size, counts in topology.cluster_group_observation_counts.items()
        },
        "ambiguous_groups": {
            str(size): [list(group) for group in groups]
            for size, groups in topology.ambiguous_groups.items()
        },
        "smid_to_tpc": {str(smid): tpc for smid, tpc in topology.smid_to_tpc.items()},
        "notes": topology.notes,
    }


def _probe_command(args, cluster_size: int, blocks: int, iters: int) -> list[str]:
    cmd = [
        str(BINARY),
        "--cluster-size",
        str(cluster_size),
        "--iters",
        str(iters),
        "--threads",
        str(args.threads),
        "--spin-cycles",
        str(args.spin_cycles),
    ]
    if blocks:
        cmd += ["--blocks", str(blocks)]
    if args.dynamic_smem:
        cmd += ["--dynamic-smem", str(args.dynamic_smem)]
    return cmd


def has_duplicate_smids(payload: dict) -> bool:
    for iteration in range(payload["launch"]["iters"]):
        smids = [
            row["smid"]
            for row in payload["rows"]
            if row["iteration"] == iteration
        ]
        if len(smids) != len(set(smids)):
            return True
    return False


def calibrate_blocks(args, cluster_size: int, multi_processor_count: int) -> int:
    if args.blocks:
        return args.blocks

    candidate = (multi_processor_count // cluster_size) * cluster_size
    while candidate >= cluster_size:
        payload = json.loads(
            run(
                _probe_command(args, cluster_size, candidate, 1),
                stdout=subprocess.PIPE,
            ).stdout
        )
        if not has_duplicate_smids(payload):
            return candidate
        candidate -= cluster_size
    raise RuntimeError(f"could not find one-wave block count for cluster size {cluster_size}")


def run_probe_for_size(args, cluster_size: int, multi_processor_count: int) -> dict:
    blocks = calibrate_blocks(args, cluster_size, multi_processor_count)
    return json.loads(
        run(
            _probe_command(args, cluster_size, blocks, args.iters),
            stdout=subprocess.PIPE,
        ).stdout
    )


def summarize_rows(rows: list[dict]) -> dict:
    per_size: dict[str, dict[str, int]] = {}
    for row in rows:
        key = str(row["cluster_size"])
        stats = per_size.setdefault(
            key,
            {
                "rows": 0,
                "unique_smids": 0,
                "available_clusters_per_iteration": 0,
            },
        )
        stats["rows"] += 1
    for key, stats in per_size.items():
        size_rows = [row for row in rows if str(row["cluster_size"]) == key]
        iterations = {row["iteration"] for row in size_rows}
        stats["unique_smids"] = len({row["smid"] for row in size_rows})
        if iterations:
            stats["available_clusters_per_iteration"] = (
                len(size_rows) // len(iterations) // int(key)
            )
    observed, missing = smid_inventory(rows)
    return {
        "observed_smids": list(observed),
        "missing_smids_in_dense_range": list(missing),
        "per_cluster_size": per_size,
    }


def print_table(payload: dict) -> None:
    device = payload["device"]
    print(
        f"device: {device['name']} cc={device['compute_capability']} "
        f"SMs={device['multi_processor_count']} "
        f"driver={device['driver_version']} runtime={device['runtime_version']}"
    )
    inventory = payload["inventory"]
    print(
        f"observed SM ids: {len(inventory['observed_smids'])}; "
        f"missing in dense range: {inventory['missing_smids_in_dense_range']}"
    )
    print()
    print("tpc_id  smids")
    print("------  -----")
    for tpc_id, group in enumerate(payload["topology"]["tpc_groups"]):
        print(f"{tpc_id:<6}  {','.join(str(smid) for smid in group)}")
    print()
    print("cluster_size  group_id  observations  smids")
    print("------------  --------  ------------  -----")
    for size, groups in sorted(
        payload["topology"]["cluster_groups"].items(), key=lambda item: int(item[0])
    ):
        for group_id, group in enumerate(groups):
            group_key = ",".join(str(smid) for smid in group)
            observations = payload["topology"]["cluster_group_observation_counts"][
                size
            ][group_key]
            print(
                f"{size:<12}  {group_id:<8}  {observations:<12}  "
                f"{','.join(str(smid) for smid in group)}"
            )
    if payload["topology"]["notes"]:
        print()
        print(f"notes: {payload['topology']['notes']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nvcc", default=os.environ.get("NVCC", tool("nvcc")))
    parser.add_argument("--arch", default=detect_arch())
    parser.add_argument("--cluster-sizes", type=parse_cluster_sizes, default=[1, 2, 4, 8, 16])
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--blocks", type=int, default=0)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--dynamic-smem", type=int, default=0)
    parser.add_argument("--spin-cycles", type=int, default=1000)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    version = nvcc_version(args.nvcc)
    build(args.nvcc, args.arch)

    device_probe = json.loads(
        run(
            _probe_command(args, cluster_size=1, blocks=1, iters=1),
            stdout=subprocess.PIPE,
        ).stdout
    )
    multi_processor_count = device_probe["device"]["multi_processor_count"]
    runs = [
        run_probe_for_size(args, cluster_size, multi_processor_count)
        for cluster_size in args.cluster_sizes
    ]
    rows = [row for result in runs for row in result["rows"]]
    observations = observations_from_rows(rows)
    topology = infer_topology(observations)
    payload = {
        "nvcc_version": version,
        "arch": args.arch,
        "device": runs[0]["device"],
        "launches": [result["launch"] for result in runs],
        "inventory": summarize_rows(rows),
        "topology": topology_to_json(topology),
        "rows": rows,
    }

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print_table(payload)
        if args.output is not None:
            print()
            print(f"raw JSON: {args.output}")


if __name__ == "__main__":
    main()
