# EulerAI technology coverage — independent lab evidence

This mapping supports technical explanation and a runnable portfolio. It cannot
prove an earlier internship assignment, leadership role, production deployment,
business metric or employer architecture. No employer source/data is included.

| Technical point | Implemented evidence | Verified scope / remaining boundary |
| --- | --- | --- |
| BentoML | `serving_lab/bento_service.py`, `bentofile.yaml`, `tests/test_bento.py` | Real BentoML 1.4.39 package built; actual Linux Bento server serves the shared authenticated FastAPI gateway; lifespan and SSE cleanup tested. |
| FastAPI model gateway | `serving_lab/app.py`, `backends.py`, existing gateway tests | Existing bounded queue, cancellation cleanup, deadlines, breaker, model routing and metrics are retained. CPU generation is deterministic simulation. |
| PostgreSQL / pgvector | `serving_lab/retrieval.py`, `compose.delivery.yaml`, `tests/test_retrieval.py` | Real PostgreSQL 17 + pgvector 0.8.6: vector insertion/update, exact cosine top-k, namespace filtering, dimension/model guards, committed persistence and HTTP API. Default token-hash vectors are not neural semantic embeddings. |
| Retrieval-to-generation data flow | `scripts/rag_demo.py` | Retrieves stored fictional documents and passes source-marked context to the selected gateway model. Default reply is simulated. Domain answer quality and injection robustness are not established. |
| Build/deployment pipeline | `.github/workflows/delivery.yml`, `Dockerfile.delivery`, `scripts/deploy.py`, `stack_smoke.py` | Builds Bento and Linux image, deploys a complete CPU stack, checks real HTTP serving + persistence, drills a failed-image rollback. Local equivalent executed; a remote CI run must be read after push before claiming that run passed. |
| GPU container preparation | `scripts/gpu_preflight.py`, `compose.delivery.gpu.yaml`, original `compose.gpu.yaml` | Local RTX 2060 / driver 591.59 / compute 7.5 / 6 GiB observed; NVIDIA device visible inside a Linux Docker container. This is a runtime-device test, not a vLLM/CUDA generation benchmark. |
| GPU readiness and rollback procedure | `scripts/deploy.py --profile gpu`, `docs/gpu-migration.md` | Executable preflight and real Qwen canary gate, immutable app/vLLM image records, restoration of previous verified images, and documented source-service traffic rollback. GPU model load/generation and GPU rollback remain unexecuted. |
| Azure pgvector selection | `docs/delivery.md` | Portable adapter, TLS/extension/privilege guidance and reasoned selection criteria. No Azure server or billable resource was created, and no Azure performance comparison was measured. |

## Local validation on 2026-09-23

- `ruff check .`: passed.
- Full Python suite with a real `TEST_PG_DSN`: **29 passed**, no skipped cases.
- `bentoml build --version local-euler-validation`: succeeded.
- Linux delivery image built and deployed using project `ms-lab-euler-audit`;
  gateway bound only to `127.0.0.1:18877`, retrieval to `127.0.0.1:18876`.
- Bento model-backend readiness, normal deterministic generation, properly
  terminated SSE, actual pgvector insert/search and dimension rejection: passed.
- Stopped/restarted PostgreSQL and both application containers, then searched
  the existing canary **without reinserting it**: passed.
- Built a separate intentionally broken entrypoint image; deployment returned
  failure, restored the previous immutable image and passed its canary:
  `rollback=verified`. The previous release record remained unchanged.
- GPU device probe in the cached official pgvector Linux image: passed. The
  probe image and its ID are recorded to prevent confusing it with a vLLM test.

See [machine-readable local evidence](evidence/eulerai-local.json). Release state
and raw intermediate results stay under ignored `results/`; no secret is copied
to evidence. The tests include upstream-library deprecation warnings (Starlette,
Pydantic/BentoML/pathspec), with no test failure; they are not hidden.

For an interview, a precise description is: “I can demonstrate a Bento-hosted
FastAPI service, persistent pgvector retrieval and a verified CPU container
release/rollback. I have a GPU preflight and migration procedure; this local
validation does not yet include vLLM model generation or a historical company
server migration.”
