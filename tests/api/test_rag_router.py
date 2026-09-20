"""
Tests for app/routers/rag.py — /api/v1/advanced/{query,ingest-async,
ingest-status}. Both the LangGraph agent (run_query) and the Celery task
(.delay / AsyncResult) are monkeypatched, so these tests never touch
OpenAI, Tavily, Postgres, Redis or RabbitMQ.
"""
from types import SimpleNamespace

from fastapi.testclient import TestClient


def _client(monkeypatch):
    from app import db, main

    monkeypatch.setattr(db, "init_db", lambda: None)
    return TestClient(main.app)


def test_query_endpoint_returns_agent_result(monkeypatch):
    client = _client(monkeypatch)

    from app.agents import rag_graph

    fake_result = {
        "answer": "Kubernetes orchestrates containers.",
        "strategy": "local_pgvector",
        "contexts_used": ["Kubernetes orchestrates containers."],
        "sources": ["doc1"],
    }
    monkeypatch.setattr(rag_graph, "run_query", lambda question: fake_result)

    resp = client.post("/api/v1/advanced/query", json={"question": "What is Kubernetes?"})

    assert resp.status_code == 200
    assert resp.json() == fake_result


def test_ingest_async_enqueues_celery_task(monkeypatch):
    client = _client(monkeypatch)

    from app import celery_worker

    monkeypatch.setattr(
        celery_worker.ingest_document_task,
        "delay",
        lambda text, source: SimpleNamespace(id="fake-task-id"),
    )

    resp = client.post(
        "/api/v1/advanced/ingest-async",
        json={"text": "some document text", "source": "unit-test"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["task_id"] == "fake-task-id"
    assert body["status"] == "queued"


def test_ingest_status_reports_task_state(monkeypatch):
    client = _client(monkeypatch)

    from app import celery_worker

    fake_result = SimpleNamespace(status="SUCCESS", ready=lambda: True, result={"num_chunks": 3})
    monkeypatch.setattr(celery_worker.celery_app, "AsyncResult", lambda task_id: fake_result)

    resp = client.get("/api/v1/advanced/ingest-status/fake-task-id")

    assert resp.status_code == 200
    body = resp.json()
    assert body["task_id"] == "fake-task-id"
    assert body["status"] == "SUCCESS"
    assert body["result"] == {"num_chunks": 3}


def test_ingest_status_pending_task_has_no_result_yet(monkeypatch):
    client = _client(monkeypatch)

    from app import celery_worker

    fake_result = SimpleNamespace(status="PENDING", ready=lambda: False, result=None)
    monkeypatch.setattr(celery_worker.celery_app, "AsyncResult", lambda task_id: fake_result)

    resp = client.get("/api/v1/advanced/ingest-status/some-task-id")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "PENDING"
    assert body["result"] is None


def test_query_endpoint_rejects_missing_question(monkeypatch):
    client = _client(monkeypatch)

    resp = client.post("/api/v1/advanced/query", json={})

    assert resp.status_code == 422
