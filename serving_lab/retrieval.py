"""Explicit PostgreSQL/pgvector retrieval adapter. No database fallback or cloud provisioning."""

import argparse
import hashlib
import hmac
import math
import os
import re
from dataclasses import dataclass

import numpy as np
import psycopg
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pgvector.psycopg import register_vector
from psycopg import sql
from psycopg.rows import dict_row
from pydantic import BaseModel, ConfigDict, Field, model_validator

from serving_lab.app import RequestContextMiddleware


@dataclass(frozen=True)
class RetrievalSettings:
    dsn: str
    dimension: int = 32
    embedding_model: str = "demo-lexical-v1"
    api_key: str = ""

    def __post_init__(self):
        if not self.dsn:
            raise ValueError("PG_DSN must explicitly select a database")
        if not 1 <= self.dimension <= 2000:
            raise ValueError("PGVECTOR_DIMENSION must be between 1 and 2000")
        if not re.fullmatch(r"[A-Za-z0-9._/-]{1,120}", self.embedding_model):
            raise ValueError("Invalid EMBEDDING_MODEL identifier")

    @classmethod
    def from_env(cls):
        return cls(os.getenv("PG_DSN", ""), int(os.getenv("PGVECTOR_DIMENSION", "32")),
                   os.getenv("EMBEDDING_MODEL", "demo-lexical-v1"), os.getenv("API_KEY", ""))


def validate_vector(values: list[float], dimension: int) -> np.ndarray:
    if len(values) != dimension:
        raise ValueError(f"Expected {dimension} vector dimensions")
    if any(not math.isfinite(value) or abs(value) > 1e10 for value in values):
        raise ValueError("Vector components must be finite and bounded")
    vector = np.asarray(values, dtype=np.float32)
    if float(np.linalg.norm(vector)) <= 1e-12:
        raise ValueError("Cosine search requires a nonzero vector")
    return vector


def demo_embedding(text: str, dimension: int) -> list[float]:
    """Deterministic token hashing, not a trained semantic embedding model."""
    values = [0.0] * dimension
    for token in re.findall(r"\w+", text.casefold()):
        digest = hashlib.sha256(token.encode()).digest()
        values[int.from_bytes(digest[:4], "big") % dimension] += 1
    if not any(values):
        raise ValueError("Text must contain at least one word")
    norm = math.sqrt(sum(value * value for value in values))
    return [value / norm for value in values]


class PgVectorStore:
    """One database embedding space, version checked on every connection.

    Exact cosine search is deliberate for the small demonstration corpus. HNSW
    is not silently enabled: filtered ANN recall would need separate evaluation.
    """

    def __init__(self, settings: RetrievalSettings):
        self.settings = settings

    def connect(self, *, vector=True):
        connection = psycopg.connect(self.settings.dsn, connect_timeout=3, row_factory=dict_row,
                                     options="-c statement_timeout=5000 -c lock_timeout=3000")
        try:
            if vector:
                register_vector(connection)
            return connection
        except Exception:
            connection.close()
            raise

    def _check_space(self, connection):
        row = connection.execute("SELECT dimension, model FROM serving_lab.embedding_space WHERE id = 1").fetchone()
        if row != {"dimension": self.settings.dimension, "model": self.settings.embedding_model}:
            raise ValueError("Database embedding space differs; use a new database and re-embed explicitly")

    def initialize(self):
        # No DROP/ALTER existing vectors, and transactional advisory lock makes concurrent init safe.
        with self.connect(vector=False) as connection:
            connection.execute("SELECT pg_advisory_xact_lock(783001924)")
            connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
            connection.execute("CREATE SCHEMA IF NOT EXISTS serving_lab")
            connection.execute("""CREATE TABLE IF NOT EXISTS serving_lab.embedding_space (
                id integer PRIMARY KEY CHECK (id = 1), dimension integer NOT NULL, model text NOT NULL)""")
            connection.execute("""INSERT INTO serving_lab.embedding_space VALUES (1, %s, %s)
                ON CONFLICT (id) DO NOTHING""", (self.settings.dimension, self.settings.embedding_model))
            self._check_space(connection)
            connection.execute(sql.SQL("""CREATE TABLE IF NOT EXISTS serving_lab.documents (
                namespace text NOT NULL, id text NOT NULL, title text NOT NULL, content text NOT NULL,
                source text NOT NULL, embedding vector({dimension}) NOT NULL,
                updated_at timestamptz NOT NULL DEFAULT now(), PRIMARY KEY (namespace, id))""").format(
                    dimension=sql.Literal(self.settings.dimension)))

    def ready(self):
        with self.connect() as connection:
            self._check_space(connection)
            connection.execute("SELECT id FROM serving_lab.documents LIMIT 0")
            version = connection.execute("SELECT extversion FROM pg_extension WHERE extname='vector'").fetchone()
            return {"ready": True, "extension": version["extversion"], "dimension": self.settings.dimension,
                    "embedding_model": self.settings.embedding_model, "search": "exact_cosine"}

    def embed(self, text, values):
        if values is None:
            if self.settings.embedding_model != "demo-lexical-v1":
                raise ValueError("Supply vectors from the configured embedding model")
            values = demo_embedding(text, self.settings.dimension)
        return validate_vector(values, self.settings.dimension)

    def put(self, namespace: str, document):
        vector = self.embed(document.content, document.embedding)
        with self.connect() as connection:
            self._check_space(connection)
            connection.execute("""INSERT INTO serving_lab.documents(namespace,id,title,content,source,embedding)
                VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT (namespace,id) DO UPDATE
                SET title=EXCLUDED.title, content=EXCLUDED.content, source=EXCLUDED.source,
                    embedding=EXCLUDED.embedding, updated_at=now()""",
                               (namespace, document.id, document.title, document.content, document.source, vector))
        return {"id": document.id, "namespace": namespace, "stored": True}

    def search(self, namespace: str, query):
        vector = self.embed(query.text or "", query.embedding)
        with self.connect() as connection:
            self._check_space(connection)
            rows = connection.execute("""SELECT id,title,content,source,embedding <=> %s AS distance
                FROM serving_lab.documents WHERE namespace=%s
                ORDER BY distance ASC, id ASC LIMIT %s""", (vector, namespace, query.top_k)).fetchall()
        return {"namespace": namespace, "embedding_model": self.settings.embedding_model,
                "search": "exact_cosine", "matches": rows}


class Document(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[A-Za-z0-9._-]{1,80}$")
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1, max_length=12000)
    source: str = Field(min_length=1, max_length=500)
    embedding: list[float] | None = Field(default=None, min_length=1, max_length=2000)


class Search(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str | None = Field(default=None, min_length=1, max_length=4000)
    embedding: list[float] | None = Field(default=None, min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)

    @model_validator(mode="after")
    def require_one_input(self):
        if (self.text is None) == (self.embedding is None):
            raise ValueError("Supply exactly one of text or embedding")
        return self


def create_retrieval_app(settings=None, store=None):
    settings = settings or RetrievalSettings.from_env()
    store = store or PgVectorStore(settings)
    app = FastAPI(title="Persistent retrieval laboratory", version="0.2.0")
    app.add_middleware(RequestContextMiddleware)

    def authorize(request: Request):
        if settings.api_key and not hmac.compare_digest(
            request.headers.get("authorization", ""), f"Bearer {settings.api_key}"
        ):
            raise HTTPException(401, "Invalid credentials")

    def namespace(value: str):
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", value):
            raise HTTPException(422, "Invalid namespace")
        return value

    @app.exception_handler(psycopg.Error)
    async def database_error(request, exc):
        # Never return a DSN, SQL parameters, database diagnostics or document content.
        return JSONResponse({"detail": "retrieval_database_unavailable"}, status_code=503)

    @app.exception_handler(ValueError)
    async def value_error(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.get("/healthz")
    def health():
        return {"status": "alive"}

    @app.get("/readyz")
    def ready():
        try:
            return store.ready()
        except (psycopg.Error, ValueError):
            return JSONResponse({"ready": False}, status_code=503)

    @app.put("/v1/namespaces/{space}/documents", dependencies=[Depends(authorize)])
    def put(space: str, document: Document):
        return store.put(namespace(space), document)

    @app.post("/v1/namespaces/{space}/search", dependencies=[Depends(authorize)])
    def search(space: str, query: Search):
        return store.search(namespace(space), query)

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["init", "check"])
    args = parser.parse_args()
    target = PgVectorStore(RetrievalSettings.from_env())
    try:
        if args.command == "init":
            target.initialize()
        print(target.ready())
    except (psycopg.Error, ValueError):
        parser.exit(1, "Database initialization/readiness failed; check configuration and permissions.\n")
