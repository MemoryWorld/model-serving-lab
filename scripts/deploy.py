"""Deploy a locally built image, verify real requests and restore the last verified release on failure.

This operates one dedicated Compose project, never a cloud account or a bare-metal
process. No database volume is removed. A first failed release stops app containers.
"""

import argparse
import json
import os
import re
import subprocess
from contextlib import contextmanager
from pathlib import Path

from gpu_preflight import inspect_host
from stack_smoke import verify

ROOT = Path(__file__).resolve().parents[1]


def run(args, env, timeout=660):
    result = subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        # Compose interpolation/logs may contain credentials. Never copy them into release evidence.
        raise RuntimeError(f"Command failed: {args[0]} {args[1]} (exit {result.returncode})")
    return result.stdout.strip()


def image_id(image, env):
    return run(["docker", "image", "inspect", image, "--format", "{{.Id}}"], env, timeout=20)


def compose(project, profile):
    command = ["docker", "compose", "-p", project, "-f", "compose.delivery.yaml"]
    if profile == "gpu":
        command.extend(["-f", "compose.delivery.gpu.yaml"])
    return command


@contextmanager
def exclusive_release(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    # The operator must inspect a stale lock after an unclean process termination.
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, str(os.getpid()).encode())
        yield
    finally:
        os.close(descriptor)
        path.unlink()


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def release(project, image, profile, state_dir):
    if not re.fullmatch(r"[a-z][a-z0-9-]{1,40}", project):
        raise ValueError("Use a dedicated lowercase Compose project name")
    env = os.environ.copy()
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / f"{project}.json"
    event_path = state_dir / f"{project}.last-attempt.json"
    with exclusive_release(state_dir / f"{project}.lock"):
        previous = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else None
        target = {"image": image_id(image, env), "profile": profile, "project": project,
                  "bento_port": env.get("BENTO_PORT", "18877"),
                  "retrieval_port": env.get("RETRIEVAL_PORT", "18876"),
                  "model": env.get("UPSTREAM_MODEL", "Qwen/Qwen3-0.6B")}
        # Do not silently take ownership of an existing project, or move its bound ports.
        running = run(compose(project, profile) + ["ps", "-a", "-q"], env, timeout=20)
        if running and previous is None:
            raise RuntimeError("Existing project has no verified release record; use a fresh project name")
        if previous and any(previous[field] != target[field] for field in ("bento_port", "retrieval_port")):
            raise RuntimeError("Port changes need a separate project and explicit traffic switch")
        if profile == "gpu":
            if not env.get("API_KEY") or not env.get("UPSTREAM_KEY") or env["API_KEY"] == env["UPSTREAM_KEY"]:
                raise ValueError("GPU release requires distinct gateway and upstream keys")
            if not env.get("VLLM_IMAGE"):
                raise ValueError("GPU release requires an explicit, locally cached VLLM_IMAGE")
            target["vllm_image"] = image_id(env["VLLM_IMAGE"], env)
            preflight = inspect_host(target["vllm_image"])
            if not preflight["ready_for_canary"]:
                atomic_json(event_path, {"status": "preflight_failed", "checks": preflight})
                raise RuntimeError("GPU preflight failed; no release change was made")

        def activate(record):
            deploy_env = {**env, "LAB_IMAGE": record["image"], "UPSTREAM_MODEL": record["model"],
                          "BENTO_PORT": record["bento_port"], "RETRIEVAL_PORT": record["retrieval_port"]}
            if record.get("vllm_image"):
                deploy_env["VLLM_IMAGE"] = record["vllm_image"]
            run(compose(project, record["profile"]) + ["up", "-d", "--no-build", "--pull", "never",
                                                       "--remove-orphans", "--wait", "--wait-timeout", "600"], deploy_env)
            return verify(f"http://127.0.0.1:{record['bento_port']}/gateway",
                          f"http://127.0.0.1:{record['retrieval_port']}", env.get("API_KEY", ""),
                          model="qwen" if record["profile"] == "gpu" else "offline-small")

        try:
            checks = activate(target)
        except (Exception, KeyboardInterrupt):
            event = {"status": "release_failed", "candidate": target, "rollback": "not_attempted"}
            try:
                if previous:
                    event["rollback_checks"] = activate(previous)
                    event["rollback"] = "verified"
                else:
                    run(compose(project, profile) + ["stop", "bento", "retrieval"], env)
                    if profile == "gpu":
                        run(compose(project, profile) + ["stop", "vllm"], env)
                    event["rollback"] = "no_previous_release_apps_stopped_database_retained"
            except Exception:
                event["rollback"] = "failed_operator_attention_required"
            atomic_json(event_path, event)
            raise RuntimeError(f"Release failed; rollback={event['rollback']}") from None
        atomic_json(state_path, target)
        atomic_json(event_path, {"status": "verified", "release": target, "checks": checks})
        return {"status": "verified", "profile": profile, "evidence": str(event_path)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--image", required=True, help="Already built local image; resolved to immutable image ID")
    parser.add_argument("--profile", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--state-dir", type=Path, default=ROOT / "results" / "releases")
    args = parser.parse_args()
    try:
        print(json.dumps(release(args.project, args.image, args.profile, args.state_dir)))
    except (RuntimeError, ValueError, OSError, subprocess.TimeoutExpired) as exc:
        parser.exit(1, str(exc) + "\n")
