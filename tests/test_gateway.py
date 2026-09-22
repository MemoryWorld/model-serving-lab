import asyncio
import json
import socket
import threading
import time
from dataclasses import replace

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient

from serving_lab.app import ChatRequest, create_app
from serving_lab.backends import BackendError, Fault, VLLMBackend
from serving_lab.config import Settings
from serving_lab.controls import Capacity, CircuitBreaker, CircuitOpen, Overloaded, TokenBucket

BASE = Settings(offline_delay=0, rate_burst=1000)


def payload(**changes):
    return {"model": "offline-small", "messages": [{"role": "user", "content": "Fictional public example"}], **changes}


def test_routing_auth_request_ids_and_readiness():
    app = create_app(replace(BASE, api_key="test-key"))
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").json()["ready"]
        assert client.get("/v1/models").status_code == 401
        headers = {"Authorization": "Bearer test-key", "X-Request-ID": "case-123"}
        models = client.get("/v1/models", headers=headers).json()["data"]
        assert {item["id"] for item in models} == {"offline-small", "offline-explain"}
        result = client.post("/v1/chat/completions", headers=headers, json=payload())
        assert result.status_code == 200 and result.headers["x-request-id"] == "case-123"
        assert result.json()["id"] == "case-123"
        other = client.post("/v1/chat/completions", headers=headers, json=payload(model="offline-explain"))
        assert other.json()["choices"] != result.json()["choices"]
        assert client.post("/v1/chat/completions", headers=headers, json=payload(model="arbitrary/model")).status_code == 404
        assert client.post("/v1/chat/completions", headers=headers, json=payload(fault={"delay_ms": 1})).status_code == 400
        assert client.post("/v1/chat/completions", headers=headers, content=b"x" * 65537).status_code == 413
        assert client.post("/v1/chat/completions", headers=headers, json=payload(max_tokens=0)).status_code == 422


def test_stream_protocol_and_metrics_no_sensitive_labels(caplog):
    app = create_app(BASE)
    with TestClient(app) as client:
        result = client.post("/v1/chat/completions", headers={"X-Request-ID": "safe-id"}, json=payload(stream=True, messages=[{"role": "user", "content": "SENSITIVE_PROMPT_917"}]))
        assert result.headers["content-type"].startswith("text/event-stream")
        lines = [line[6:] for line in result.text.splitlines() if line.startswith("data: ")]
        assert lines[-1] == "[DONE]"
        assert json.loads(lines[0])["choices"][0]["delta"]["content"] == "Offline "
        assert json.loads(lines[-2])["choices"][0]["finish_reason"] == "stop"
        text = client.get("/metrics").text
        assert 'serving_requests_total{model="offline-small",outcome="success"} 1' in text
        assert 'serving_active_requests{model="offline-small"} 0' in text
        assert 'serving_request_duration_seconds_bucket{model="offline-small",le="+Inf"} 1' in text
        assert "SENSITIVE_PROMPT_917" not in text + caplog.text
        assert "safe-id" in caplog.text
        runtime = app.state.runtime
        assert runtime.backends["offline-small"].open_streams == 0


def test_rate_limit_and_invalid_request_id():
    with TestClient(create_app(replace(BASE, rate_burst=1, rate_per_second=0.001))) as client:
        first = client.post("/v1/chat/completions", headers={"X-Request-ID": "unsafe id"}, json=payload())
        assert first.status_code == 200 and len(first.headers["x-request-id"]) == 32
        rejected = client.post("/v1/chat/completions", json=payload())
        assert rejected.status_code == 429 and rejected.headers["Retry-After"] == "1"


@pytest.mark.parametrize("stream", [False, True])
def test_timeout_closes_stream_and_records_failure(stream):
    app = create_app(replace(BASE, request_timeout=0.025, enable_faults=True))
    with TestClient(app) as client:
        result = client.post("/v1/chat/completions", json=payload(stream=stream, fault={"delay_ms": 100}))
        if stream:
            assert result.status_code == 200 and "event: error" in result.text and "upstream_timeout" in result.text
            assert "[DONE]" not in result.text
        else:
            assert result.status_code == 504
        runtime = app.state.runtime
        assert runtime.capacities["offline-small"].active == 0
        assert runtime.backends["offline-small"].open_streams == 0
        assert runtime.metrics.requests["offline-small", "timeout"] == 1


def test_failure_circuit_open_and_recovery():
    app = create_app(replace(BASE, enable_faults=True, breaker_threshold=2, breaker_reset=0.03))
    with TestClient(app) as client:
        for _ in range(2):
            assert client.post("/v1/chat/completions", json=payload(fault={"fail_after": 0})).status_code == 502
        assert client.post("/v1/chat/completions", json=payload()).status_code == 503
        time.sleep(0.04)
        assert client.post("/v1/chat/completions", json=payload()).status_code == 200
        assert app.state.runtime.breakers["offline-small"].state == "closed"


async def test_capacity_wait_cancel_and_overflow_are_bounded():
    capacity = Capacity(1, 1, 0.05)
    first = await capacity.acquire()
    waiting = asyncio.create_task(capacity.acquire())
    await asyncio.sleep(0)
    assert capacity.active == 1 and capacity.waiting == 1
    with pytest.raises(Overloaded):
        await capacity.acquire()
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    assert capacity.waiting == 0
    first.close()
    first.close()
    last = await capacity.acquire()
    last.close()
    assert capacity.active == 0
    hold = await capacity.acquire()
    with pytest.raises(Overloaded, match="timed out"):
        await capacity.acquire()
    assert capacity.waiting == 0
    hold.close()


def test_half_open_single_probe_and_stale_success():
    now = [0.0]
    circuit = CircuitBreaker(1, 1, lambda: now[0])
    stale = circuit.acquire()
    failed = circuit.acquire()
    failed.finish("failure")
    stale.finish("success")
    assert circuit.state == "open"
    now[0] = 2
    probe = circuit.acquire()
    with pytest.raises(CircuitOpen):
        circuit.acquire()
    probe.finish("cancelled")
    recovery = circuit.acquire()
    recovery.finish("success")
    assert circuit.state == "closed"


def test_bucket_has_bounded_identity_storage():
    bucket = TokenBucket(1, 1, max_clients=2)
    assert bucket.allow("a") and not bucket.allow("a")
    bucket.allow("b")
    bucket.allow("c")
    assert len(bucket.clients) == 2


class TrackedStream(httpx.AsyncByteStream):
    def __init__(self, content):
        self.content, self.closed = content, False

    async def __aiter__(self):
        for chunk in self.content:
            yield chunk

    async def aclose(self):
        self.closed = True


async def test_vllm_adapter_protocol_success_and_lifecycle():
    body = TrackedStream([b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n', b'data: [DONE]\n\n'])
    seen = []
    async def handler(request):
        seen.append(request)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=body)
    client = httpx.AsyncClient(base_url="http://model.invalid", transport=httpx.MockTransport(handler))
    backend = VLLMBackend(replace(BASE, mode="vllm", api_key="gateway", upstream_key="upstream-secret"), client)
    stream = await backend.open(ChatRequest(**payload()), Fault())
    assert [piece async for piece in stream] == ["hello"]
    await stream.aclose()
    assert body.closed
    data = json.loads(seen[0].content)
    assert data["model"] == "Qwen/Qwen3-0.6B" and data["stream"] is True
    assert data["chat_template_kwargs"] == {"enable_thinking": False}
    assert seen[0].headers["authorization"] == "Bearer upstream-secret"
    await backend.aclose()
    assert client.is_closed


@pytest.mark.parametrize("status,content,expected", [
    (401, [], "upstream_http_401"),
    (200, [b"data: broken\n\n"], "upstream_invalid_event"),
    (200, [b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'], "upstream_truncated_stream"),
])
async def test_vllm_faults_never_fallback_and_close(status, content, expected):
    body = TrackedStream(content)
    client = httpx.AsyncClient(base_url="http://model.invalid", transport=httpx.MockTransport(lambda request: httpx.Response(status, headers={"content-type": "text/event-stream"}, stream=body)))
    backend = VLLMBackend(BASE, client)
    stream = None
    try:
        with pytest.raises(BackendError, match=expected):
            stream = await backend.open(ChatRequest(**payload()), Fault())
            _ = [piece async for piece in stream]
    finally:
        if stream:
            await stream.aclose()
        await backend.aclose()
    assert body.closed


def test_invalid_live_configuration_is_rejected():
    with pytest.raises(ValueError, match="API_KEY"):
        Settings(mode="vllm")
    with pytest.raises(ValueError, match="Fault injection"):
        Settings(mode="vllm", api_key="test", enable_faults=True)
    for url in ("http://model.invalid/path", "http://model.invalid?key=secret", "http://model.invalid#fragment"):
        with pytest.raises(ValueError, match="origin"):
            Settings(upstream_url=url)


@pytest.mark.parametrize("content", [[b"data: broken\n\n"], [b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n']])
def test_upstream_stream_failure_through_gateway_releases_permit(content):
    body = TrackedStream(content)
    transport = httpx.MockTransport(lambda request: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=body))
    settings = replace(BASE, mode="vllm", api_key="test-key")
    backend = VLLMBackend(settings, httpx.AsyncClient(base_url="http://model.invalid", transport=transport))
    app = create_app(settings, {"qwen": backend})
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", headers={"Authorization": "Bearer test-key"}, json=payload(model="qwen", stream=True))
        assert response.status_code == 200 and "event: error" in response.text
        assert "[DONE]" not in response.text
        assert body.closed
        assert app.state.runtime.capacities["qwen"].active == 0
        assert app.state.runtime.metrics.requests["qwen", "upstream_error"] == 1


def test_backend_error_does_not_leak_upstream_body_or_key(caplog):
    body = TrackedStream([b"PRIVATE_UPSTREAM_ERROR"])
    transport = httpx.MockTransport(lambda request: httpx.Response(401, stream=body))
    settings = replace(BASE, mode="vllm", api_key="test-key", upstream_key="PRIVATE_KEY_21")
    backend = VLLMBackend(settings, httpx.AsyncClient(base_url="http://model.invalid", transport=transport))
    app = create_app(settings, {"qwen": backend})
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", headers={"Authorization": "Bearer test-key"}, json=payload(model="qwen"))
        assert response.status_code == 502
        assert body.closed
        assert app.state.runtime.capacities["qwen"].active == 0
        for secret in ("PRIVATE_UPSTREAM_ERROR", "PRIVATE_KEY_21"):
            assert secret not in response.text + caplog.text


@pytest.fixture
def live_server():
    application = create_app(replace(BASE, offline_delay=0.15, concurrency=1, max_queue=1, queue_timeout=0.05))
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    address = f"http://127.0.0.1:{listener.getsockname()[1]}"
    server = uvicorn.Server(uvicorn.Config(application, log_level="warning", access_log=False, lifespan="on"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server.started
    yield application, address
    server.should_exit = True
    thread.join(timeout=5)
    listener.close()
    assert not thread.is_alive()


@pytest.mark.parametrize("read_first_chunk", [False, True])
async def test_real_socket_disconnect_releases_stalled_stream(live_server, read_first_chunk):
    application, address = live_server
    async with httpx.AsyncClient(base_url=address, trust_env=False) as client:
        async with client.stream("POST", "/v1/chat/completions", json=payload(stream=True)) as response:
            assert response.status_code == 200
            if read_first_chunk:
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        break
            assert application.state.runtime.capacities["offline-small"].active == 1
        deadline = time.monotonic() + 0.5
        while application.state.runtime.capacities["offline-small"].active and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        runtime = application.state.runtime
        assert runtime.capacities["offline-small"].active == 0
        assert runtime.backends["offline-small"].open_streams == 0
        assert runtime.metrics.requests["offline-small", "cancelled"] == 1
        assert (await client.post("/v1/chat/completions", json=payload(max_tokens=1))).status_code == 200


async def test_real_socket_backpressure_rejects_and_recovers(live_server):
    application, address = live_server
    async with httpx.AsyncClient(base_url=address, trust_env=False) as client:
        async with client.stream("POST", "/v1/chat/completions", json=payload(stream=True)):
            rejected = await client.post("/v1/chat/completions", json=payload())
            assert rejected.status_code == 503
            assert application.state.runtime.capacities["offline-small"].waiting == 0
        await asyncio.sleep(0.05)
        assert (await client.post("/v1/chat/completions", json=payload(max_tokens=1))).status_code == 200
