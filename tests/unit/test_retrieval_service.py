"""
Unit tests for app/services/retrieval_service.py. app.db.similarity_search
is monkeypatched so no real Postgres/pgvector is needed.
"""
from app.services import retrieval_service


def test_retrieve_splits_rows_into_contexts_and_sources(monkeypatch):
    fake_rows = [
        (1, "Kubernetes orchestrates containers.", "doc1", 0.05),
        (2, "pgvector stores embeddings in Postgres.", "doc2", 0.12),
    ]

    captured = {}

    def fake_similarity_search(embedding, limit):
        captured["embedding"] = embedding
        captured["limit"] = limit
        return fake_rows

    monkeypatch.setattr(retrieval_service.db, "similarity_search", fake_similarity_search)

    contexts, sources = retrieval_service.retrieve([0.1, 0.2, 0.3])

    assert contexts == [
        "Kubernetes orchestrates containers.",
        "pgvector stores embeddings in Postgres.",
    ]
    assert sources == ["doc1", "doc2"]
    assert captured["embedding"] == [0.1, 0.2, 0.3]
    assert captured["limit"] == retrieval_service.settings.vector_search_limit


def test_retrieve_falls_back_to_doc_id_when_source_missing(monkeypatch):
    fake_rows = [(7, "some content", None, 0.2)]
    monkeypatch.setattr(retrieval_service.db, "similarity_search", lambda e, l: fake_rows)

    _, sources = retrieval_service.retrieve([0.0])

    assert sources == ["doc:7"]


def test_retrieve_returns_empty_lists_when_no_matches(monkeypatch):
    monkeypatch.setattr(retrieval_service.db, "similarity_search", lambda e, l: [])

    contexts, sources = retrieval_service.retrieve([0.0])

    assert contexts == []
    assert sources == []
