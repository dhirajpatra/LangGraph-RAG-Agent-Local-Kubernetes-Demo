"""
app/services/ingestion_service.py

Chunk + embed + store logic used by the Celery worker — matches
"insert document chunks + vectors" flowing into Postgres+pgvector in the
diagram. Pulled out of celery_worker.py so it can be unit-tested (or
reused by a batch/CLI ingestion script) without needing Celery running.
"""
from logging import getLogger
from typing import List

from app import db
from app.services import embedding_service

logger = getLogger(__name__)

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    logger.debug("chunk_text: split text into %d chunks", len(chunks))
    return [c for c in chunks if c.strip()]


def ingest(text: str, source: str = "manual-upload") -> dict:
    db.init_db()
    chunks = chunk_text(text)
    vectors = embedding_service.embed_documents(chunks)

    ids = []
    for chunk, vector in zip(chunks, vectors):
        doc_id = db.insert_document(chunk, vector, source=source)
        ids.append(doc_id)

    logger.info("ingest: inserted %d chunks for source=%r", len(chunks), source)
    return {"inserted_ids": ids, "num_chunks": len(chunks), "source": source}
