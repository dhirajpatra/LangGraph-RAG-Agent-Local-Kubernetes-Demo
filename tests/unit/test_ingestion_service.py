"""
Unit tests for app/services/ingestion_service.py — pure chunking logic plus
the ingest() orchestration, with db + embedding calls mocked out.
"""
from app.services import ingestion_service


def test_chunk_text_splits_long_text_with_overlap():
    text = "x" * 2000
    chunks = ingestion_service.chunk_text(text, chunk_size=800, overlap=100)

    assert len(chunks) >= 2
    # every chunk (except possibly the last) is exactly chunk_size long
    assert all(len(c) <= 800 for c in chunks)
    # consecutive chunks overlap by the requested amount
    assert chunks[0][-100:] == chunks[1][:100]


def test_chunk_text_short_text_returns_single_chunk():
    text = "short piece of text"
    chunks = ingestion_service.chunk_text(text, chunk_size=800, overlap=100)
    assert chunks == [text]


def test_chunk_text_drops_empty_chunks():
    # a string of exactly chunk_size followed by only whitespace should not
    # produce a trailing empty/blank chunk
    text = "a" * 800 + "   "
    chunks = ingestion_service.chunk_text(text, chunk_size=800, overlap=0)
    assert all(c.strip() for c in chunks)


def test_ingest_chunks_embeds_and_inserts(monkeypatch):
    calls = {"init_db": 0, "inserted": []}

    def fake_init_db():
        calls["init_db"] += 1

    def fake_embed_documents(chunks):
        return [[0.1, 0.2, 0.3] for _ in chunks]

    def fake_insert_document(content, embedding, source=""):
        doc_id = len(calls["inserted"]) + 1
        calls["inserted"].append((doc_id, content, embedding, source))
        return doc_id

    monkeypatch.setattr(ingestion_service.db, "init_db", fake_init_db)
    monkeypatch.setattr(ingestion_service.db, "insert_document", fake_insert_document)
    monkeypatch.setattr(
        ingestion_service.embedding_service, "embed_documents", fake_embed_documents
    )

    result = ingestion_service.ingest("hello world, this is a test document.", source="unit-test")

    assert calls["init_db"] == 1
    assert result["num_chunks"] == len(calls["inserted"])
    assert result["source"] == "unit-test"
    assert result["inserted_ids"] == [c[0] for c in calls["inserted"]]
    assert all(c[3] == "unit-test" for c in calls["inserted"])
