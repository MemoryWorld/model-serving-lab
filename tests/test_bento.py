import os

import pytest
from starlette.testclient import TestClient

pytest.importorskip("bentoml")
os.environ.setdefault("BENTOML_DO_NOT_TRACK", "true")


def test_real_bento_asgi_mount_preserves_auth_lifecycle_and_streaming(monkeypatch):
    from serving_lab.app import create_app
    from serving_lab.bento_service import ModelServingLab
    from serving_lab.config import Settings

    gateway = create_app(Settings(api_key="bento-test-key", offline_delay=0))
    monkeypatch.setattr(ModelServingLab, "mount_apps", [(gateway, "/gateway", "gateway")])
    with TestClient(ModelServingLab.to_asgi()) as client:
        assert client.get("/gateway/readyz").json()["ready"]
        assert client.get("/gateway/v1/models").status_code == 401
        headers = {"Authorization": "Bearer bento-test-key"}
        request = {"model": "offline-small", "messages": [{"role": "user", "content": "synthetic"}], "stream": True}
        response = client.post("/gateway/v1/chat/completions", json=request, headers=headers)
        assert response.status_code == 200 and "[DONE]" in response.text
        assert "text/event-stream" in response.headers["content-type"]
        assert gateway.state.runtime.capacities["offline-small"].active == 0
        assert gateway.state.runtime.backends["offline-small"].open_streams == 0
        assert client.get("/gateway/v1/models", headers=headers).status_code == 200
    assert gateway.state.runtime.stopping
