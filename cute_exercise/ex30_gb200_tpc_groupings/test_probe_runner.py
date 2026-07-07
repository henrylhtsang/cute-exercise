import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc required")
@pytest.mark.skipif(shutil.which("nvidia-smi") is None, reason="CUDA GPU required")
def test_probe_runner_captures_raw_cluster_rows():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "run_probe.py"),
            "--cluster-sizes",
            "1,2",
            "--iters",
            "1",
            "--json",
        ],
        check=True,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
    )

    payload = json.loads(result.stdout)
    assert payload["device"]["name"]
    assert payload["device"]["multi_processor_count"] >= 1
    assert payload["rows"]
    assert {row["cluster_size"] for row in payload["rows"]} == {1, 2}
    assert payload["topology"]["cluster_groups"]


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc required")
@pytest.mark.skipif(shutil.which("nvidia-smi") is None, reason="CUDA GPU required")
def test_l2_probe_runner_captures_latency_signatures():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "run_l2_probe.py"),
            "--buckets",
            "8",
            "--samples-per-bucket",
            "2",
            "--target-groups",
            "2",
            "--json",
        ],
        check=True,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
    )

    payload = json.loads(result.stdout)
    assert payload["device"]["name"]
    assert payload["rows"]
    assert len(payload["rows"][0]["signature"]) == 8
    assert payload["physical_grouping"]["group_counts"]
