"""
Tests for app/main.py. db.init_db is monkeypatched so the app can start up
without a real Postgres connection, and we avoid using TestClient as a
context manager so the lifespan/startup event never actually fires
(these tests only care about routing, not startup behaviour).
"""
from fastapi.testclient import TestClient


def test_health_endpoint(monkeypatch):
    from app import db, main

    monkeypatch.setattr(db, "init_db", lambda: None)

    client = TestClient(main.app)
    resp = client.get("/health")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
