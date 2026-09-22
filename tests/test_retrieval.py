import os
import uuid
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("psycopg")
from serving_lab.retrieval import (
    Document,
    PgVectorStore,
    RetrievalSettings,
    Search,
    create_retrieval_app,
    demo_embedding,
    validate_vector,
)


def test_embedding_is_deterministic_and_dimension_guard_is_strict():
    assert demo_embedding("Fire EXIT", 32) == demo_embedding("fire exit", 32)
    assert len(demo_embedding("synthetic", 32)) == 32
    for vector in ([1.0], [0.0, 0.0], [float("nan"), 1.0], [float("inf"), 1.0], [1e30, 1.0]):
        with pytest.raises(ValueError):
            validate_vector(vector, 2)
    with pytest.raises(ValueError):
        demo_embedding("!!!", 32)


@pytest.fixture
def database():
    dsn = os.getenv("TEST_PG_DSN")
    if not dsn:
        pytest.skip("TEST_PG_DSN is required for real PostgreSQL integration; no emulation")
    settings = RetrievalSettings(dsn=dsn, api_key="retrieval-test-key")
    store = PgVectorStore(settings)
    store.initialize()
    space = f"test-{uuid.uuid4().hex}"
    yield settings, store, space
    with store.connect() as connection:
        connection.execute("DELETE FROM serving_lab.documents WHERE namespace=%s", (space,))


def test_real_pgvector_upsert_topk_namespaces_and_new_connection_persistence(database):
    settings, store, space = database
    one = [1.0] + [0.0] * 31
    two = [0.0, 1.0] + [0.0] * 30
    store.put(space, Document(id="fire", title="Original", content="Fictional fire policy", source="synthetic:a", embedding=one))
    store.put(space, Document(id="solar", title="Solar", content="Fictional solar policy", source="synthetic:b", embedding=two))
    store.put(space, Document(id="fire", title="Updated", content="Fictional revised policy", source="synthetic:c", embedding=one))
    # A different store/connection reads committed data; no process-local vector list exists.
    reopened = PgVectorStore(settings)
    result = reopened.search(space, Search(embedding=one, top_k=1))["matches"]
    assert len(result) == 1 and result[0]["id"] == "fire" and result[0]["title"] == "Updated"
    assert result[0]["distance"] == pytest.approx(0)
    assert reopened.search(f"empty-{space}", Search(embedding=one))["matches"] == []
    assert len(reopened.search(space, Search(embedding=one))["matches"]) == 2
    # Reinitializing with another dimension/model must preserve existing data, not reset it.
    for changed in (replace(settings, dimension=16), replace(settings, embedding_model="other-model")):
        with pytest.raises(ValueError, match="differs"):
            PgVectorStore(changed).initialize()
    assert reopened.search(space, Search(embedding=one))["matches"][0]["id"] == "fire"


def test_real_retrieval_http_auth_insert_search_dimension_and_query_bounds(database):
    settings, store, space = database
    with TestClient(create_retrieval_app(settings, store)) as client:
        path = f"/v1/namespaces/{space}"
        assert client.get("/readyz").json()["ready"]
        assert client.post(f"{path}/search", json={"text": "fire"}).status_code == 401
        headers = {"Authorization": "Bearer retrieval-test-key"}
        document = {"id": "fire", "title": "Synthetic", "content": "fire exit door", "source": "synthetic:test"}
        assert client.put(f"{path}/documents", headers=headers, json=document).status_code == 200
        search = client.post(f"{path}/search", headers=headers, json={"text": "fire exit", "top_k": 1})
        assert search.json()["matches"][0]["id"] == "fire"
        for invalid in ({"embedding": [1.0]}, {"text": "fire", "top_k": 21}, {},
                        {"text": "fire", "embedding": [1.0]}, {"embedding": [0.0] * 32}):
            assert client.post(f"{path}/search", headers=headers, json=invalid).status_code == 422


def test_retrieval_database_failure_is_unready_and_sanitized():
    settings = RetrievalSettings("postgresql://privateuser:PRIVATE_PASSWORD@127.0.0.1:1/missing")
    with TestClient(create_retrieval_app(settings)) as client:
        assert client.get("/readyz").status_code == 503
        result = client.post("/v1/namespaces/test/search", json={"text": "fire"})
        assert result.status_code == 503 and result.json() == {"detail": "retrieval_database_unavailable"}
        assert "PRIVATE_PASSWORD" not in result.text
