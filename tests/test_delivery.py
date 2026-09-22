import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preflight_cannot_report_model_inference_or_cuda_success_without_probe(monkeypatch):
    gpu = load_script("gpu_preflight")
    monkeypatch.setattr(gpu, "command", lambda args, timeout=20: (True, '"linux"') if args[0] == "docker"
                        else (True, "Synthetic GPU, 591.59, 7.5, 6144, 6000"))
    report = gpu.inspect_host()
    assert report["hardware_floor_passed"] and report["linux_docker"]
    assert not report["ready_for_canary"] and report["container_gpu_verified"] is None
    assert not report["model_inference_verified"]


def test_release_lock_excludes_concurrent_publish_and_atomic_record(tmp_path):
    deploy = load_script("deploy")
    lock = tmp_path / "test.lock"
    with deploy.exclusive_release(lock):
        with pytest.raises(FileExistsError):
            with deploy.exclusive_release(lock):
                pytest.fail("Second deployment obtained the same lock")
    assert not lock.exists()
    deploy.atomic_json(tmp_path / "state.json", {"image": "sha256:test"})
    assert '"image": "sha256:test"' in (tmp_path / "state.json").read_text()


def test_failed_release_rolls_back_previous_verified_image_not_candidate(monkeypatch, tmp_path):
    deploy = load_script("deploy")
    previous = {"image": "sha256:previous", "profile": "cpu", "project": "test-release",
                "bento_port": "18877", "retrieval_port": "18876", "model": "Qwen/Qwen3-0.6B"}
    deploy.atomic_json(tmp_path / "test-release.json", previous)
    monkeypatch.delenv("BENTO_PORT", raising=False)
    monkeypatch.delenv("RETRIEVAL_PORT", raising=False)
    monkeypatch.setattr(deploy, "image_id", lambda image, env: "sha256:candidate")
    images = []

    def fake_run(args, env, timeout=660):
        if "up" in args:
            images.append(env["LAB_IMAGE"])
            if env["LAB_IMAGE"] == "sha256:candidate":
                raise RuntimeError("Synthetic failed deployment")
        return "container"

    monkeypatch.setattr(deploy, "run", fake_run)
    monkeypatch.setattr(deploy, "verify", lambda *a, **k: {"synthetic_unit_check": True})
    with pytest.raises(RuntimeError, match="rollback=verified"):
        deploy.release("test-release", "candidate", "cpu", tmp_path)
    assert images == ["sha256:candidate", "sha256:previous"]
    assert deploy.json.loads((tmp_path / "test-release.json").read_text()) == previous


def test_deployer_refuses_to_adopt_unrecorded_existing_project(monkeypatch, tmp_path):
    deploy = load_script("deploy")
    monkeypatch.setattr(deploy, "image_id", lambda image, env: "sha256:candidate")
    monkeypatch.setattr(deploy, "run", lambda *a, **k: "existing-container")
    with pytest.raises(RuntimeError, match="no verified release record"):
        deploy.release("test-release", "candidate", "cpu", tmp_path)
