# Model Serving Lab

A runnable model-serving reliability laboratory: **FastAPI gateway → bounded
admission → OpenAI-compatible vLLM/Qwen serving**, with real SSE delivery,
disconnect cleanup, controlled failure experiments and Prometheus-format metrics.

**This is a new, independent public technical example. It is not EulerAI source
code, a reconstruction of an employer's proprietary system, or evidence of a
historical company deployment.** All examples are fictional and public. The
choice of vLLM is a subsequent modernization exercise alongside prior BentoML
experience; this repository does not claim to contain that earlier work.

The default is fully offline and deterministic. It demonstrates serving
mechanics, not model intelligence. Live mode requires explicit configuration and
never falls back to offline text when the model server fails.

## Run locally

Python 3.12 or 3.13:

```sh
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# PowerShell: .venv/Scripts/Activate.ps1
python -m pip install -e '.[test]'
uvicorn serving_lab.app:app --host 127.0.0.1 --port 8788 --workers 1 --no-access-log
```

In another terminal:

```sh
python scripts/demo.py
python -m pytest -q
```

Explore `/docs`, `/v1/models`, `/healthz`, `/readyz` and `/metrics` on
`http://127.0.0.1:8788`. `scripts/demo.py` prints chunks as they arrive. The two
offline aliases, `offline-small` and `offline-explain`, route to different
deterministic responses; a nonexistent alias returns 404.

```sh
curl -N http://127.0.0.1:8788/v1/chat/completions \
  -H 'Content-Type: application/json' -H 'X-Request-ID: public-example-1' \
  -d '{"model":"offline-small","stream":true,"messages":[{"role":"user","content":"Describe a fictional bookstore assistant."}]}'
```

Set `API_KEY` to require Bearer authentication on inference, model discovery and
metrics. Default offline development binds to loopback. Liveness and readiness
are deliberately unauthenticated. No `.env` file is loaded by the Python process;
credentials are never discovered automatically or written into request logs.

## What is implemented

| Boundary | Behavior |
| --- | --- |
| Model routing | Fixed configured aliases; clients cannot choose an upstream URL. |
| Input admission | Maximum 64 KiB body, 16 messages, 4,000 characters/message and 512 requested output tokens. |
| Rate limiting | Token bucket per authenticated key, or direct peer IP in offline mode; bounded identity cache. |
| Backpressure | Per-model concurrency permits, bounded queue and queue-wait deadline; 503 when full. |
| Circuit breaker | Consecutive backend failures open the circuit; one half-open probe; cancellation does not count as a model failure. |
| Deadline | Backend open + generation share an absolute deadline. Queue wait has its own deadline. |
| Streaming | Actual SSE chunks; race the ASGI disconnect message against the next backend chunk. |
| Cleanup | Response-owned, bounded cleanup closes the upstream response and releases the permit, including cancellation before the first chunk. |
| Failure isolation | HTTP failures, malformed/truncated upstream SSE and missing `[DONE]` fail explicitly. No offline fallback. |
| Observability | Request IDs, redacted JSON completion logs, counters/gauges and latency histogram in Prometheus text format. |

The exporter deliberately uses a small fixed metric vocabulary and no additional
metrics dependency. Labels contain only configured model aliases and fixed
outcome codes. Prompts, completions, keys, request IDs and client identities are
never metric labels. The request ID appears in the response header and sanitized
completion log for correlation. `serving_stream_chunks_total` counts chunks,
**not tokenizer tokens**.

HTTP errors before response headers: 401 authentication, 404 unknown alias,
413 body limit, 422 invalid input, 429 rate limit, 502 upstream failure, 503
capacity/open circuit and 504 backend deadline. After streaming headers are sent,
failure is an `event: error` SSE event and the stream ends **without `[DONE]`**.
Clients must check stream completion rather than interpreting HTTP 200 alone as
generation success.

## Offline fault experiments

```sh
python scripts/benchmark.py --port 8788 --output results/offline-benchmark.json
```

This command starts and stops its own **offline-only local server**; stop the
interactive gateway first or use `--port 0` for an ephemeral local port. It runs
baseline streaming, a concurrent overload burst, an injected deadline failure,
an injected backend failure, open-circuit rejection and recovery. The report
records environment, configuration, individual observations and resource
invariants, and asserts that active requests, queued requests and open streams
return to zero. Measured latency and TTFT are timings of deterministic simulated
work on this machine; they are not GPU inference throughput or model benchmarks.

Fault injection requires `ENABLE_FAULTS=true` and `MODE=offline`. Example request:
`"fault":{"delay_ms":100,"fail_after":2}`. Live mode rejects fault injection at
startup. Public or GPU deployments should keep it disabled.

## Docker and GPU deployment

Offline container:

```sh
docker compose config --quiet
docker compose up --build --wait
python scripts/demo.py
docker compose down
```

For GPU serving, see [deployment guide](docs/deployment.md). The GPU composition
uses vLLM's OpenAI-compatible server with `Qwen/Qwen3-0.6B`, a deliberately small
public model suitable for validating the serving path. The image is pinned to
official vLLM v0.30.0 and its verified digest, with a `VLLM_IMAGE` override for a
hardware-compatible release; there is no floating `latest` default. Set
`UPSTREAM_MODEL` to change the model in both gateway and server. Model weights
download only when the GPU service is launched.
There is no actual GPU run or throughput result in this repository yet.

## Verification and scope

The tests include actual localhost Uvicorn/socket disconnects before/after the
first chunk, bounded wait cancellation, stream resource accounting, circuit
recovery, body-size enforcement, malformed/truncated upstream streams and secret
redaction. MockTransport is used only for the external vLLM protocol boundary;
the gateway's routing/admission/stream lifecycle executes normally.

CI is supplied for Python 3.12/3.13, lint, tests, the offline experiment and a
CPU Docker smoke run. A workflow file is not evidence that a remote run passed;
see [local validation and source status](docs/validation.md).

This is a **single-worker laboratory**. Quotas, queues and circuit state are
process-local, not distributed coordination. Do not add Uvicorn workers and
claim a global quota. It does not implement billing, scheduling across GPUs,
Kubernetes autoscaling, model training or production authentication. TLS and
edge-level abuse controls belong in a deployment layer. A readiness probe checks
upstream health, not model quality or a warmed inference cache.
