"""
tests/integration/test_infra_integration.py

Opt-in tests that hit REAL Postgres+pgvector and REAL Redis Stack — useful
for verifying your actual minikube deployment (e.g. after port-forwarding
postgres-service and redis-service to localhost).

These are skipped automatically if the infra isn't reachable, so
`pytest` (without -m integration) never fails in an environment with no
cluster running. To run them on purpose:

    kubectl port-forward -n rag-demo svc/postgres-service 5432:5432 &
    kubectl port-forward -n rag-demo svc/redis-service 6379:6379 &
    RUN_INTEGRATION_TESTS=1 pytest -m integration tests/integration

They do NOT call OpenAI/Tavily — embeddings are faked so no API key is
needed even for this "integration" tier.
"""
import os
import uuid

import pytest

pytestmark = pytest.mark.integration

RUN = os.environ.get("RUN_INTEGRATION_TESTS") == "1"
EMBEDDING_DIM = 1536


def _postgres_reachable() -> bool:
    if not RUN:
        return False
    try:
        from app import db

        with db.get_conn():
            return True
    except Exception:
        return False


def _redis_reachable() -> bool:
    if not RUN:
        return False
    try:
        from app.services import cache_service

        cache_service.get_client().ping()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _postgres_reachable(), reason="Postgres not reachable (see module docstring)")
def test_insert_and_similarity_search_roundtrip():
    from app import db

    db.init_db()
    fake_embedding = [0.001 * i for i in range(EMBEDDING_DIM)]
    marker = f"integration-test-{uuid.uuid4()}"

    doc_id = db.insert_document(marker, fake_embedding, source="integration-test")
    assert doc_id > 0

    rows = db.similarity_search(fake_embedding, limit=5)
    assert any(r[0] == doc_id for r in rows)


@pytest.mark.skipif(not _redis_reachable(), reason="Redis not reachable (see module docstring)")
def test_semantic_cache_roundtrip_against_real_redis():
    from app.services import cache_service

    embedding = [0.001 * i for i in range(EMBEDDING_DIM)]
    question = f"integration test question {uuid.uuid4()}"
    answer = "integration test answer"

    assert cache_service.check_cache(embedding) is None
    cache_service.write_cache(question, embedding, answer)
    assert cache_service.check_cache(embedding) == answer
