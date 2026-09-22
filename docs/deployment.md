# Deployment and failure experiments

## Configured variables

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `MODE` | `offline` | `offline` or `vllm`; no implicit fallback. |
| `API_KEY` | empty | Gateway Bearer key; mandatory in vLLM mode. |
| `UPSTREAM_URL` | `http://127.0.0.1:8001` | Trusted origin only; no path, query, credentials or fragment. |
| `UPSTREAM_KEY` | empty | Optional upstream Bearer key; GPU compose requires it. |
| `UPSTREAM_MODEL` | `Qwen/Qwen3-0.6B` | Actual model sent to vLLM; public alias is `qwen`. |
| `CONCURRENCY` | 2 | Per-model active requests; a streaming connection holds one permit. |
| `MAX_QUEUE` | 4 | Maximum waiting requests per model. |
| `QUEUE_TIMEOUT` | 0.25 | Maximum admission wait in seconds. |
| `REQUEST_TIMEOUT` | 5 | Total backend open + generation deadline in seconds. |
| `RATE_PER_SECOND` / `RATE_BURST` | 20 / 40 | Process-local token-bucket refill and burst. |
| `BREAKER_THRESHOLD` / `BREAKER_RESET` | 3 / 2 | Failure threshold and open hold-down seconds. |
| `OFFLINE_DELAY` | 0.01 | Simulated delay per deterministic chunk, seconds. |
| `ENABLE_FAULTS` | false | Explicitly permits offline-only fault payloads. |

## GPU profile (configuration only until hardware validation)

Prerequisites: NVIDIA GPU compatible with the chosen vLLM release, a compatible
driver, NVIDIA Container Toolkit and Docker Compose supporting `gpus: all`.
Consult the selected vLLM image's release notes instead of assuming a universal
CUDA/driver minimum. This repository has not measured VRAM consumption.

Set `API_KEY` and a different `UPSTREAM_KEY` using your platform's secret
provisioning. Do not paste real values into committed files. The image default
is official `vllm/vllm-openai:v0.30.0`, pinned by digest; that release uses CUDA 13.
The official release also lists `v0.30.0-cu129` for CUDA 12.9. Choose a compatible
`VLLM_IMAGE` override only after checking your driver/GPU; neither variant has
been executed locally. `UPSTREAM_MODEL` changes both server and gateway names.

```sh
docker compose -f compose.gpu.yaml config --quiet
docker compose -f compose.gpu.yaml up --build --wait --wait-timeout 600
python scripts/demo.py --model qwen
docker compose -f compose.gpu.yaml down
```

The vLLM container has no published host port. The gateway talks to it on the
private compose network. Only the gateway is exposed, on host loopback port
8788. Requests use `/v1/chat/completions`, deterministic temperature 0 and the
Qwen3 chat-template setting `enable_thinking=false`. The upstream response must
be SSE and finish with `[DONE]`; errors never become simulated text.

The example model context is capped at 4096 and the server's maximum active
sequences at 16; gateway concurrency defaults to 4 in this composition. These
are conservative configuration choices, not benchmarked optimal settings.
Initial model download can take longer than the health start period.

Avoid `docker compose config` without `--quiet` with real secrets in the
environment, because interpolation can print them. Environment variables keep
keys out of source control but are still visible to privileged container
operators; adopt your platform's secret manager for deployment.

## Observability

Prometheus can scrape `/metrics`; configure its Bearer authorization when
`API_KEY` is enabled. Counters include outcomes such as `success`, `timeout`,
`upstream_error`, `overloaded`, `circuit_open`, `rate_limited` and `cancelled`.
Gauges expose active requests, queued requests and circuit state. The histogram
measures model invocation time; it excludes admission queue time, which is
bounded separately. No prompt/completion/key is recorded by the application log.
Use `--no-access-log` to keep the Uvicorn access logger disabled consistently.

The HTTPX adapter ignores proxy environment variables (`trust_env=False`), so
accidentally configured process proxies do not redirect upstream credentials.
It only connects to the operator-configured origin, never a client-supplied URL.

## Fault interpretation

- Queue rejection is 503 and a retry hint; it is not counted as a backend fault.
- Backend timeout is 504 before headers, or an SSE error after headers. Cleanup
  releases the admission permit and closes the model response.
- A disconnected client is recorded as cancellation and does not trip a circuit.
- Half-open cancellation releases the probe slot so another request can probe.
- A network/HTTP/malformed-stream failure increments the backend circuit.
- Tests and `scripts/benchmark.py` require no model endpoint, GPU, API key or Redis.

Before publishing performance numbers, run on identified hardware and record
model revision, vLLM image digest, precision/quantization, prompt/output token
lengths, warmup, concurrency, queue settings and raw measurements. The bundled
offline experiment cannot substantiate GPU acceleration claims.
