import asyncio
import json
from dataclasses import dataclass

import httpx


class BackendError(Exception):
    """Only sanitized codes cross the backend boundary."""


@dataclass
class Fault:
    delay_ms: int = 0
    fail_after: int | None = None


class OfflineStream:
    def __init__(self, backend, pieces, delay, fail_after):
        self.backend, self.pieces, self.delay, self.fail_after = backend, pieces, delay, fail_after
        self.closed = False
        self.index = 0
        backend.open_streams += 1

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.closed:
            raise StopAsyncIteration
        await asyncio.sleep(self.delay)
        if self.fail_after is not None and self.index >= self.fail_after:
            raise BackendError("injected_failure")
        if self.index >= len(self.pieces):
            raise StopAsyncIteration
        piece = self.pieces[self.index]
        self.index += 1
        return piece

    async def aclose(self):
        if not self.closed:
            self.closed = True
            self.backend.open_streams -= 1
            self.backend.closed_streams += 1


class OfflineBackend:
    def __init__(self, style: str, delay: float):
        self.style, self.delay = style, delay
        self.open_streams = self.closed_streams = 0

    async def open(self, payload, fault: Fault):
        # Public fictional examples; never pretend deterministic text is inference.
        texts = {
            "brief": "Offline demo: route admission stream cleanup verified.",
            "explain": "Offline demo: bounded queues protect capacity and cancellation releases backend resources.",
        }
        words = texts[self.style].split(" ")[:payload.max_tokens]
        return OfflineStream(self, [word + " " for word in words], self.delay + fault.delay_ms / 1000, fault.fail_after)

    async def ready(self):
        return True

    async def aclose(self):
        pass


class OpenAIStream:
    def __init__(self, response):
        self.response, self.closed, self.finished = response, False, False
        self.lines = response.aiter_lines().__aiter__()

    def __aiter__(self):
        return self

    async def __anext__(self):
        while not self.closed:
            try:
                line = await anext(self.lines)
            except StopAsyncIteration as exc:
                if not self.finished:
                    raise BackendError("upstream_truncated_stream") from exc
                raise
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                self.finished = True
                raise StopAsyncIteration
            try:
                event = json.loads(data)
                if "error" in event:
                    raise BackendError("upstream_error_event")
                choices = event.get("choices", [])
                text = choices[0].get("delta", {}).get("content") if choices else None
                if text:
                    return text
            except (ValueError, TypeError, AttributeError, IndexError) as exc:
                raise BackendError("upstream_invalid_event") from exc
        raise StopAsyncIteration

    async def aclose(self):
        if not self.closed:
            self.closed = True
            await self.response.aclose()


class VLLMBackend:
    def __init__(self, settings, client=None):
        self.settings = settings
        self.client = client or httpx.AsyncClient(
            base_url=settings.upstream_url.rstrip("/"),
            timeout=httpx.Timeout(settings.request_timeout, connect=2.0),
            limits=httpx.Limits(max_connections=settings.concurrency + 2, max_keepalive_connections=settings.concurrency),
            trust_env=False,
        )

    async def open(self, payload, fault):
        headers = {"Authorization": f"Bearer {self.settings.upstream_key}"} if self.settings.upstream_key else {}
        request = self.client.build_request("POST", "/v1/chat/completions", headers=headers, json={
            "model": self.settings.upstream_model,
            "messages": [message.model_dump() for message in payload.messages],
            "max_tokens": payload.max_tokens,
            "temperature": 0,
            "stream": True,
            "chat_template_kwargs": {"enable_thinking": False},
        })
        response = await self.client.send(request, stream=True)
        try:
            if response.status_code != 200:
                raise BackendError(f"upstream_http_{response.status_code}")
            if "text/event-stream" not in response.headers.get("content-type", ""):
                raise BackendError("upstream_not_event_stream")
            return OpenAIStream(response)
        except BaseException:
            await response.aclose()
            raise

    async def ready(self):
        try:
            return (await self.client.get("/health", timeout=0.5)).is_success
        except httpx.HTTPError:
            return False

    async def aclose(self):
        await self.client.aclose()
