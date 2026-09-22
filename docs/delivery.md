# BentoML, pgvector and delivery

These are independent implementations, not recovered company code. No Azure
resource, public endpoint or paid model API is created by any default command.

## Real BentoML integration

Install `.[bento]` and use:

```sh
bentoml serve serving_lab.bento_service:ModelServingLab --host 127.0.0.1 --port 18877
bentoml build --version local-demo
```

`ModelServingLab` is an actual BentoML 1.4 class service. Its ASGI mount reuses
the original gateway, including its lifespan, authentication, queue and breaker,
SSE error/end protocol and shutdown cleanup. It runs one worker because capacity
and quotas are process-local. `/readyz` is Bento worker readiness;
`/gateway/readyz` is the model-backend readiness used by our deployment canary.
The default server has no trained model or GPU requirement. Use `MODE=vllm` and
the existing gateway settings to connect a live OpenAI-compatible vLLM server.

The service and package versions were checked against the official
[BentoML service guide](https://docs.bentoml.com/en/latest/build-with-bentoml/services.html)
and [BentoML 1.4.39](https://pypi.org/project/bentoml/1.4.39/) on 2026-09-23.
The Docker delivery image runs the same service source; `bentoml build` separately
validates Bento packaging and resolves its package lock. No BentoCloud is used.

## PostgreSQL adapter and initialization

`PgVectorStore` uses Psycopg 3 and the real pgvector Python adapter. Initialization
uses one transaction and a PostgreSQL advisory lock, creates the vector extension,
and registers a single embedding-space model/dimension. A second initializer
must match those settings. It never drops old data or rewrites existing vectors.
The vector column has a database-enforced dimension. Each operation opens a
connection, checks embedding-space metadata, and commits/rolls back on exit.
This small demo intentionally avoids a pool and uses exact cosine search.

The API accepts finite, bounded, nonzero vectors, enforces length, bounds top-k
to 20 and text size, and uses SQL parameters for user data. Results contain IDs,
source labels and distance. Namespaces separate collections, **not identities**:
one API key can access every namespace. Add identity-bound authorization before
using private tenant data. The local Compose database password is explicitly a
fictional demo credential; its port is not published in the normal stack.

```sh
# Set PG_DSN in the process environment, using a dedicated demonstration database.
python -m serving_lab.retrieval init
uvicorn serving_lab.retrieval:create_retrieval_app --factory --host 127.0.0.1 --port 18876 --no-access-log
```

`PUT /v1/namespaces/manual/documents` takes `id`, `title`, `content`, `source`,
and optional `embedding`. Reusing `(namespace,id)` atomically updates that row.
`POST /v1/namespaces/manual/search` takes exactly one of `text` or `embedding`
and optional `top_k`. Raw vectors must come from the configured model. The server
can validate dimensions/numeric bounds but cannot prove a client's model provenance.

Default `demo-lexical-v1` text embeddings use normalized SHA-256 token buckets;
this avoids model downloads and makes persistence/ranking tests repeatable.
Hash collisions and lexical overlap limit quality. It is not neural embedding,
semantic evaluation or an ANN throughput claim. For another model use a fresh
database, set its name/dimension, re-embed the corpus and supply query vectors.
The server refuses automatic conversion between incompatible embedding spaces.

### Real database tests

```sh
docker compose -p serving-dbtest -f compose.delivery.yaml -f compose.test.yaml up -d --wait postgres
# PG_TEST_PORT defaults to 18875; the override binds only to 127.0.0.1.
# Set TEST_PG_DSN=postgresql://lab:local-demo-password@127.0.0.1:18875/serving_lab
python -m pytest -q
docker compose -p serving-dbtest -f compose.delivery.yaml -f compose.test.yaml down
```

The tests insert actual vectors, check ordered top-k and same-ID updates, reopen
connections, isolate namespaces, reject mismatched initialization without losing
rows, and exercise HTTP authentication and malformed vector inputs. Without
`TEST_PG_DSN`, integration cases are visibly skipped, never replaced by SQLite.
The CI database job sets it, so its integration cases must run.

## Build, deployment and rollback

`scripts/deploy.py` is a local single-host deployer. A successful release records
image ID, mode, model and ports; a separate last-attempt file records canary and
rollback results. Concurrent releases sharing a project/state directory are
excluded by an exclusive lock. An existing unrecorded project is never adopted.
Run from the same checkout and retain `results/releases/` between releases.

On failure the previous image is restarted and canary-tested. On first-release
failure only app containers are stopped; the database remains. The script never
uses `down -v`. A crashed process may leave a lock: verify that its PID is no
longer deploying before manually removing that one lock. Rollback failure is
reported as requiring operator attention, never as a successful release.

The database schema here is additive and compatible across the example images.
This is **application-image rollback**, not database downgrade, arbitrary Compose
configuration rollback or secret rollback. Secrets remain supplied by the current
environment. Keep the Compose files and credentials stable during a release;
change ports/schema/embedding space via a separately tested deployment.

The CI flow deploys on an ephemeral GitHub-hosted Linux machine, calls real HTTP
endpoints, stops/restarts the database and apps, then uses a deliberately broken
entrypoint image to require a verified rollback. This is verifiable CI deployment;
it is not a claim of deployment to a production company or cloud environment.

## Azure pgvector compatibility and selection boundary

Azure Database for PostgreSQL Flexible Server supports the `vector` extension;
the operator must allowlist it before database-level initialization. Our adapter
uses standard PostgreSQL/vector SQL and a configurable libpq DSN. No Azure SDK
or provider-specific schema is needed. For an existing Azure server, configure
its host/database/user and TLS certificate verification (`sslmode=verify-full`
with the appropriate root certificate), inject credentials outside source control,
and use a separate privileged migration identity to enable the extension/schema.
Runtime only needs metadata read and document read/write privileges.

Selection rationale: PostgreSQL joins, transactions, backup/restore and existing
relational operations can make pgvector a practical RAG store. Selection still
requires workload-specific recall/latency, filtered ANN behavior, index memory,
regional availability and cost evaluation. This repo validates portable SQL and
local persistence; it has **not** benchmarked or provisioned Azure.

Sources checked 2026-09-23: [pgvector extension and Docker tags](https://github.com/pgvector/pgvector),
[pgvector Python/Psycopg adapter](https://pypi.org/project/pgvector/0.4.2/),
[Psycopg 3.3.6](https://pypi.org/project/psycopg/3.3.6/),
[Microsoft's pgvector guide](https://learn.microsoft.com/en-au/azure/postgresql/extensions/how-to-use-pgvector).
