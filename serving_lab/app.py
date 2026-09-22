import asyncio
import hashlib
import hmac
import json
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

import anyio
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from serving_lab.backends import BackendError, Fault, OfflineBackend, VLLMBackend
from serving_lab.config import Settings
from serving_lab.controls import Capacity, CircuitBreaker, CircuitOpen, Overloaded, TokenBucket
from serving_lab.metrics import Metrics


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class FaultInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    delay_ms: int = Field(default=0, ge=0, le=2000)
    fail_after: int | None = Field(default=None, ge=0, le=512)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = Field(min_length=1, max_length=64)
    messages: list[Message] = Field(min_length=1, max_length=16)
    max_tokens: int = Field(default=32, ge=1, le=512)
    stream: bool = False
    fault: FaultInput | None = None


class RequestContextMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        incoming = headers.get(b"x-request-id", b"").decode("ascii", errors="ignore")
        request_id = incoming if re.fullmatch(r"[A-Za-z0-9._-]{1,64}", incoming) else uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id

        async def traced_send(message):
            if message["type"] == "http.response.start":
                message.setdefault("headers", []).append((b"x-request-id", request_id.encode()))
            await send(message)

        received = 0

        async def bounded_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > 65536:
                    raise HTTPException(status_code=413, detail="Request body exceeds 64 KiB")
            return message

        await self.app(scope, bounded_receive, traced_send)


def sse(data, event=None):
    prefix = f"event: {event}\n" if event else ""
    return (prefix + "data: " + (data if isinstance(data, str) else json.dumps(data, separators=(",", ":"))) + "\n\n").encode()


class Invocation:
    def __init__(self, runtime, model, request_id, lease, ticket):
        self.runtime, self.model, self.request_id = runtime, model, request_id
        self.lease, self.ticket, self.stream = lease, ticket, None
        self.started, self.finished = time.perf_counter(), False
        self.deadline = asyncio.get_running_loop().time() + runtime.settings.request_timeout

    def finish(self, outcome):
        if not self.finished:
            self.finished = True
            self.ticket.finish("success" if outcome == "success" else "failure" if outcome in {"timeout", "upstream_error"} else "cancelled")
            self.runtime.metrics.observe(self.model, outcome, time.perf_counter() - self.started, self.request_id)

    async def close(self):
        try:
            if self.stream is not None:
                # Backend cleanup is bounded and shielded from client cancellation.
                with anyio.CancelScope(shield=True):
                    with anyio.move_on_after(1):
                        await self.stream.aclose()
        finally:
            self.finish("cancelled")
            self.lease.close()


class ManagedStreamResponse(Response):
    """Race ASGI disconnect against streaming, including a stalled next chunk.

    Cleanup belongs to the response call, so cancellation before the generator's
    first iteration cannot leak an already-acquired permit/upstream response.
    """

    def __init__(self, invocation):
        super().__init__(status_code=200, media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        self.raw_headers = [(key, value) for key, value in self.raw_headers if key != b"content-length"]
        self.invocation = invocation

    async def __call__(self, scope, receive, send):
        invocation = self.invocation

        async def stream_response(group):
            try:
                await send({"type": "http.response.start", "status": 200, "headers": self.raw_headers})
                try:
                    async with asyncio.timeout_at(invocation.deadline):
                        async for piece in invocation.stream:
                            invocation.runtime.metrics.chunks[invocation.model] += 1
                            await send({"type": "http.response.body", "body": sse({
                                "id": invocation.request_id, "object": "chat.completion.chunk", "model": invocation.model,
                                "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
                            }), "more_body": True})
                        await send({"type": "http.response.body", "body": sse({
                            "id": invocation.request_id, "object": "chat.completion.chunk", "model": invocation.model,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        }) + sse("[DONE]"), "more_body": True})
                    invocation.finish("success")
                except (TimeoutError, httpx.TimeoutException):
                    invocation.finish("timeout")
                    await send({"type": "http.response.body", "body": sse({"error": {"code": "upstream_timeout"}, "request_id": invocation.request_id}, "error"), "more_body": True})
                except (BackendError, httpx.HTTPError):
                    invocation.finish("upstream_error")
                    await send({"type": "http.response.body", "body": sse({"error": {"code": "upstream_error"}, "request_id": invocation.request_id}, "error"), "more_body": True})
                await send({"type": "http.response.body", "body": b"", "more_body": False})
            except OSError:
                invocation.finish("cancelled")
            finally:
                group.cancel_scope.cancel()

        try:
            async with anyio.create_task_group() as group:
                group.start_soon(stream_response, group)
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        invocation.finish("cancelled")
                        group.cancel_scope.cancel()
                        break
        finally:
            await invocation.close()


class Runtime:
    def __init__(self, settings, backends):
        self.settings, self.backends = settings, backends
        self.capacities = {name: Capacity(settings.concurrency, settings.max_queue, settings.queue_timeout) for name in backends}
        self.breakers = {name: CircuitBreaker(settings.breaker_threshold, settings.breaker_reset) for name in backends}
        self.rate_limit = TokenBucket(settings.rate_per_second, settings.rate_burst)
        self.metrics = Metrics()
        self.stopping = False


def create_app(settings=None, backends=None):
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app):
        selected = backends
        if selected is None:
            selected = ({"offline-small": OfflineBackend("brief", settings.offline_delay),
                         "offline-explain": OfflineBackend("explain", settings.offline_delay)}
                        if settings.mode == "offline" else {"qwen": VLLMBackend(settings)})
        app.state.runtime = Runtime(settings, selected)
        try:
            yield
        finally:
            app.state.runtime.stopping = True
            for backend in selected.values():
                await backend.aclose()

    app = FastAPI(title="Model Serving Lab", version="0.1.0", lifespan=lifespan)
    app.add_middleware(RequestContextMiddleware)

    def authorize(request):
        supplied = request.headers.get("authorization", "").removeprefix("Bearer ")
        if settings.api_key and not hmac.compare_digest(supplied, settings.api_key):
            raise HTTPException(status_code=401, detail="Invalid API key")
        identity = supplied if settings.api_key else (request.client.host if request.client else "local")
        return hashlib.sha256(identity.encode()).hexdigest()

    @app.get("/healthz")
    async def health():
        return {"status": "alive"}

    @app.get("/readyz")
    async def ready(request: Request):
        runtime = request.app.state.runtime
        checks = dict(zip(runtime.backends, await asyncio.gather(*(backend.ready() for backend in runtime.backends.values())), strict=True))
        ok = all(checks.values()) and not runtime.stopping
        return JSONResponse({"ready": ok, "backends": checks}, status_code=200 if ok else 503)

    @app.get("/metrics")
    async def metrics(request: Request):
        authorize(request)
        runtime = request.app.state.runtime
        return Response(runtime.metrics.render(runtime.capacities, runtime.breakers), media_type="text/plain; version=0.0.4")

    @app.get("/v1/models")
    async def models(request: Request):
        authorize(request)
        return {"object": "list", "data": [{"id": name, "object": "model", "owned_by": "independent-lab"} for name in request.app.state.runtime.backends]}

    @app.post("/v1/chat/completions")
    async def chat(payload: ChatRequest, request: Request):
        identity = authorize(request)
        runtime = request.app.state.runtime
        if payload.model not in runtime.backends:
            raise HTTPException(status_code=404, detail="Unknown model alias; inspect /v1/models")
        if payload.fault is not None and not settings.enable_faults:
            raise HTTPException(status_code=400, detail="Fault injection is disabled")
        model, request_id = payload.model, request.state.request_id
        if not runtime.rate_limit.allow(identity):
            runtime.metrics.observe(model, "rate_limited", 0, request_id)
            raise HTTPException(status_code=429, detail="Rate limit exceeded", headers={"Retry-After": "1"})
        if runtime.stopping:
            raise HTTPException(status_code=503, detail="Service is draining")
        try:
            ticket = runtime.breakers[model].acquire()
        except CircuitOpen as exc:
            runtime.metrics.observe(model, "circuit_open", 0, request_id)
            raise HTTPException(status_code=503, detail="Backend circuit is open", headers={"Retry-After": "1"}) from exc
        try:
            lease = await runtime.capacities[model].acquire()
        except Overloaded as exc:
            ticket.finish("cancelled")
            runtime.metrics.observe(model, "overloaded", 0, request_id)
            raise HTTPException(status_code=503, detail="Backend capacity exhausted", headers={"Retry-After": "1"}) from exc
        except BaseException:
            ticket.finish("cancelled")
            raise
        invocation = Invocation(runtime, model, request_id, lease, ticket)
        transferred = False
        try:
            async with asyncio.timeout_at(invocation.deadline):
                invocation.stream = await runtime.backends[model].open(payload, Fault(**payload.fault.model_dump()) if payload.fault else Fault())
                if payload.stream:
                    transferred = True
                    return ManagedStreamResponse(invocation)
                chunks = [piece async for piece in invocation.stream]
            invocation.finish("success")
            return {"id": request_id, "object": "chat.completion", "model": model,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(chunks).strip()}, "finish_reason": "stop"}]}
        except (TimeoutError, httpx.TimeoutException) as exc:
            invocation.finish("timeout")
            raise HTTPException(status_code=504, detail="Backend deadline exceeded") from exc
        except (BackendError, httpx.HTTPError) as exc:
            invocation.finish("upstream_error")
            raise HTTPException(status_code=502, detail="Backend request failed") from exc
        finally:
            if not transferred:
                await invocation.close()

    return app


app = create_app()
