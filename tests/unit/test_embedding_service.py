"""
Unit tests for app/services/embedding_service.py. The real OpenAIEmbeddings
client is never constructed here — _client() is monkeypatched entirely so
no network call (and no real API key) is needed.
"""
from app.services import embedding_service


class _FakeEmbeddings:
    def embed_query(self, text):
        return [float(len(text))]

    def embed_documents(self, texts):
        return [[float(len(t))] for t in texts]


def test_embed_query_delegates_to_client(monkeypatch):
    monkeypatch.setattr(embedding_service, "_client", lambda: _FakeEmbeddings())

    result = embedding_service.embed_query("hello")

    assert result == [5.0]


def test_embed_documents_delegates_to_client(monkeypatch):
    monkeypatch.setattr(embedding_service, "_client", lambda: _FakeEmbeddings())

    result = embedding_service.embed_documents(["a", "bb", "ccc"])

    assert result == [[1.0], [2.0], [3.0]]
