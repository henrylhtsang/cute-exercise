import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc required")
@pytest.mark.skipif(shutil.which("nvidia-smi") is None, reason="CUDA GPU required")
def test_native_cuda_probe_runs_with_cuda_13_3():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "build.py"),
            "--run",
            "--n",
            "1048576",
            "--iters",
            "3",
        ],
        check=True,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
    )

    payload = json.loads(result.stdout)
    assert "release 13.3" in payload["nvcc_version"]
    assert payload["device"]
    assert payload["variants"]["u32"]["ok"]
    assert payload["variants"]["v4_u32"]["ok"]
    assert payload["variants"]["v8_u32"]["ok"]
    assert payload["variants"]["v8_u32"]["bytes_per_thread"] == 32
