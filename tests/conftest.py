"""
tests/conftest.py

Runs before any other test module is imported, so we set dummy env vars
here (module level, not inside a fixture) to guarantee app.config.settings
never accidentally tries to use real credentials or reach real
infrastructure during unit/api tests. Real network calls are additionally
blocked per-test by monkeypatching the relevant service functions.

Only tests under tests/integration/ are meant to talk to real
Postgres/Redis/RabbitMQ (see tests/integration/README or the `integration`
marker), and only when that infra is actually reachable.
"""
import os

os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("TAVILY_API_KEY", "test-tavily-key")
os.environ.setdefault("LANGSMITH_API_KEY", "")
os.environ.setdefault("LANGSMITH_TRACING", "false")
os.environ.setdefault("DATABASE_URL", "postgresql://raguser:ragpass@localhost:5432/ragdb_test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/15")
os.environ.setdefault("CELERY_BROKER_URL", "amqp://guest:guest@localhost:5672//")
os.environ.setdefault("CELERY_RESULT_BACKEND", "redis://localhost:6379/15")

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_embedding_client_cache():
    """embedding_service._client() is @lru_cache'd; clear it between tests
    so a monkeypatch in one test can't leak a fake client into the next."""
    from app.services import embedding_service

    embedding_service._client.cache_clear()
    yield
    embedding_service._client.cache_clear()


@pytest.fixture
def sample_documents():
    return [
        {"content": "Kubernetes is a container orchestration platform.", "source": "doc1"},
        {"content": "pgvector adds vector similarity search to Postgres.", "source": "doc2"},
    ]
