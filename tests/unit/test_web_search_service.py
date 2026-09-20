"""
Unit tests for app/services/web_search_service.py. TavilyClient is
monkeypatched so no network call/API key is needed.
"""
from app.services import web_search_service


class _FakeTavilyClient:
    def __init__(self, api_key=None):
        self.api_key = api_key

    def search(self, query, max_results):
        return {
            "results": [
                {"content": f"result 1 for {query}", "url": "https://example.com/1"},
                {"content": f"result 2 for {query}", "url": "https://example.com/2"},
            ]
        }


def test_search_returns_contexts_and_sources(monkeypatch):
    monkeypatch.setattr(web_search_service, "TavilyClient", _FakeTavilyClient)

    contexts, sources = web_search_service.search("what is pgvector?")

    assert contexts == [
        "result 1 for what is pgvector?",
        "result 2 for what is pgvector?",
    ]
    assert sources == ["https://example.com/1", "https://example.com/2"]


def test_search_handles_empty_results(monkeypatch):
    class _EmptyTavilyClient:
        def __init__(self, api_key=None):
            pass

        def search(self, query, max_results):
            return {"results": []}

    monkeypatch.setattr(web_search_service, "TavilyClient", _EmptyTavilyClient)

    contexts, sources = web_search_service.search("anything")

    assert contexts == []
    assert sources == []
