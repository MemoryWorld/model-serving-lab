# Validation and primary-source status

This repository is a newly written independent example. It does not establish
historical employer results or expose proprietary code.

## Local environment

Local verification uses Windows and Python 3.12, with FastAPI 0.141.1, HTTPX
0.28.1 and Uvicorn 0.53.0 already present in the development environment. Direct
dependencies are pinned to those locally exercised versions. Official PyPI JSON
metadata was checked on **2026-09-23 (Australia/Sydney)** and confirmed these
versions as current releases at that check. The same check returned vLLM 0.30.0.

After an initial network outage, the official vLLM Docker documentation, release
v0.30.0 and Qwen model card were read successfully. The release names the CUDA 13
image `vllm/vllm-openai:v0.30.0` and CUDA 12.9 variant `v0.30.0-cu129`. Docker Hub's
official tag API returned the pinned digest
`sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`.
The Qwen card documents `enable_thinking=false`; vLLM environment documentation
documents `VLLM_API_KEY`. Configuration compatibility is not a GPU execution test.
Primary references (the first three and Prometheus/Docker references are API
background, not separately claimed as a live audit):

- [FastAPI streaming responses](https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse)
- [Starlette responses and disconnect handling](https://www.starlette.io/responses/)
- [HTTPX async streaming and cleanup](https://www.python-httpx.org/async/)
- [vLLM online serving](https://docs.vllm.ai/en/latest/serving/online_serving/)
- [vLLM Docker deployment](https://docs.vllm.ai/en/stable/deployment/docker.html)
- [Official Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Prometheus text exposition format](https://prometheus.io/docs/instrumenting/exposition_formats/)
- [Docker Compose GPU support](https://docs.docker.com/compose/how-tos/gpu-support/)
- [FastAPI package release metadata](https://pypi.org/pypi/fastapi/json)
- [vLLM package release metadata](https://pypi.org/pypi/vllm/json)
- [vLLM v0.30.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.30.0)
- [Official Docker image tag metadata](https://hub.docker.com/v2/repositories/vllm/vllm-openai/tags/v0.30.0)
- [vLLM environment variables](https://docs.vllm.ai/en/stable/configuration/env_vars/)

## Scope of evidence

The final local check used this repository's own fresh `.venv` after
`pip install -e '.[test]'`: **20 tests passed**, with two upstream Starlette
test-client deprecation warnings. Ruff and both CPU/GPU `docker compose config
--quiet` checks passed. The measured offline report is preserved at
[evidence/offline-windows.json](evidence/offline-windows.json): 5/5 baseline
requests completed; the 12-request burst admitted 2 and rejected 10 with 503;
timeout, injected failure, open circuit and recovery produced 504, 502, 503 and
200 respectively. Peak active/queued counts were 2/2, and all final resource
counts were zero. This is a local reliability observation, not a capacity claim.

- Local automated tests exercise actual API requests, real localhost streaming
  sockets and disconnects, plus a controlled substitute for the external vLLM
  HTTP protocol. They do not run model inference.
- `scripts/benchmark.py` records actual measurements of deterministic offline
  simulated delays/failures, includes raw observations and asserts resource
  accounting. Inspect the produced JSON in `results/` for the run's timestamp,
  software versions and configuration; do not quote it as GPU performance.
- Docker CLI is installed locally, but the Docker engine was unavailable. The
  compose files can be parsed with `config --quiet`; images have not been built
  or executed locally in this implementation.
- GPU/vLLM configuration is supplied for subsequent hardware verification. No
  GPU inference, VRAM measurement, quantization or deployment success is claimed.
- [GitHub Actions run 35739488792](https://github.com/MemoryWorld/model-serving-lab/actions/runs/35739488792) passed on the published implementation commit `cbd2af9`: Python 3.12 and 3.13 lint/tests/offline experiments succeeded, and the CPU Docker image built, started and returned a successful readiness probe. GPU inference remains untested.
