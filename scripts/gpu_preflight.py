"""Read-only host checks. Optional GPU container probe uses an already-cached image."""

import argparse
import csv
import io
import json
import subprocess
from pathlib import Path


def command(args, timeout=20):
    try:
        run = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
        return run.returncode == 0, run.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return False, ""


def inspect_host(image=None, minimum_mib=4096):
    gpu_ok, gpu_output = command(["nvidia-smi", "--query-gpu=name,driver_version,compute_cap,memory.total,memory.free",
                                  "--format=csv,noheader,nounits"])
    devices = []
    if gpu_ok:
        for row in csv.reader(io.StringIO(gpu_output), skipinitialspace=True):
            if len(row) == 5:
                devices.append({"name": row[0], "driver": row[1], "compute_capability": float(row[2]),
                                "total_mib": int(row[3]), "free_mib": int(row[4])})
    docker_ok, docker_output = command(["docker", "info", "--format", "{{json .OSType}}"])
    linux_engine = docker_ok and docker_output == '"linux"'
    eligible = any(gpu["compute_capability"] >= 7.5 and gpu["free_mib"] >= minimum_mib for gpu in devices)
    container_gpu = None
    cached_id = None
    if image:
        cached, cached_id = command(["docker", "image", "inspect", image, "--format", "{{.Id}}"])
        if linux_engine and cached:
            container_gpu, _ = command(["docker", "run", "--rm", "--pull=never", "--gpus", "all",
                                         "--network", "none", "--entrypoint", "nvidia-smi", image, "-L"], timeout=60)
        else:
            container_gpu = False
    return {"devices": devices, "linux_docker": linux_engine, "hardware_floor_passed": eligible,
            "minimum_free_mib": minimum_mib, "container_gpu_verified": container_gpu,
            "probe_image": image, "probe_image_id": cached_id,
            "ready_for_canary": bool(eligible and linux_engine and container_gpu),
            "model_inference_verified": False,
            "limits": ["Memory floor is not a measured model VRAM requirement.",
                       "Driver/CUDA image compatibility still requires a successful inference canary.",
                       "No model download, host migration or traffic switch was performed by this check."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container-image", help="Explicit cached image to probe; never pulled automatically")
    parser.add_argument("--minimum-free-mib", type=int, default=4096)
    parser.add_argument("--output", type=Path, default=Path("results/gpu-preflight.json"))
    args = parser.parse_args()
    report = inspect_host(args.container_image, args.minimum_free_mib)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["ready_for_canary"] else 2)
