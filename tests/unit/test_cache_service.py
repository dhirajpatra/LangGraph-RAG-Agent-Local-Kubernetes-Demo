"""
Unit tests for app/services/cache_service.py — the Redis semantic cache.
The real Redis client is replaced with an in-memory fake so no Redis Stack
instance is needed to run these.
"""
from types import SimpleNamespace

import numpy as np
import redis as redis_module

from app.services import cache_service


class _FakeFT:
    """Fake of client.ft(index_name)."""

    def __init__(self, store, index_exists=True):
        self._store = store
        self._index_exists = index_exists

    def info(self):
        if not self._index_exists:
            raise redis_module.ResponseError("no such index")
        return {"ok": True}

    def create_index(self, schema, definition):
        self._index_exists = True

    def search(self, query, query_params=None):
        # Return the closest stored vector by naive cosine similarity,
        # mimicking a KNN search result shape closely enough for the
        # cache_service code under test.
        if not self._store:
            return SimpleNamespace(docs=[])

        query_vec = np.frombuffer(query_params["vec"], dtype=np.float32)
        best_key, best_doc, best_score = None, None, None
        for key, doc in self._store.items():
            stored_vec = np.frombuffer(doc["embedding"], dtype=np.float32)
            denom = (np.linalg.norm(query_vec) * np.linalg.norm(stored_vec)) or 1e-9
            cosine_sim = float(np.dot(query_vec, stored_vec) / denom)
            distance = 1 - cosine_sim
            if best_score is None or distance < best_score:
                best_key, best_doc, best_score = key, doc, distance

        fake_doc = SimpleNamespace(answer=best_doc["answer"], score=best_score)
        return SimpleNamespace(docs=[fake_doc])


class _FakeRedisClient:
    def __init__(self, index_exists=True):
        self._store = {}
        self._ft = _FakeFT(self._store, index_exists=index_exists)

    def ft(self, index_name):
        return self._ft

    def hset(self, key, mapping):
        self._store[key] = mapping


def test_ensure_index_creates_index_when_missing(monkeypatch):
    fake_client = _FakeRedisClient(index_exists=False)
    monkeypatch.setattr(cache_service, "get_client", lambda: fake_client)

    cache_service.ensure_index()

    assert fake_client._ft._index_exists is True


def test_write_then_check_cache_hits_for_identical_embedding(monkeypatch):
    fake_client = _FakeRedisClient()
    monkeypatch.setattr(cache_service, "get_client", lambda: fake_client)

    embedding = [1.0, 0.0, 0.0]
    cache_service.write_cache("What is Kubernetes?", embedding, "An orchestration platform.")

    cached = cache_service.check_cache(embedding)

    assert cached == "An orchestration platform."


def test_check_cache_misses_when_below_threshold(monkeypatch):
    fake_client = _FakeRedisClient()
    monkeypatch.setattr(cache_service, "get_client", lambda: fake_client)

    cache_service.write_cache("What is Kubernetes?", [1.0, 0.0, 0.0], "An orchestration platform.")

    # an orthogonal vector -> cosine similarity 0, well below the threshold
    cached = cache_service.check_cache([0.0, 1.0, 0.0])

    assert cached is None


def test_check_cache_returns_none_on_empty_cache(monkeypatch):
    fake_client = _FakeRedisClient()
    monkeypatch.setattr(cache_service, "get_client", lambda: fake_client)

    assert cache_service.check_cache([1.0, 0.0, 0.0]) is None
