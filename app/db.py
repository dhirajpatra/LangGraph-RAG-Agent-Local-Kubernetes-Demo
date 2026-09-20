"""
Postgres + pgvector access layer.
Matches the diagram's "Postgres + pgvector" box (Deployment: postgres,
Service: postgres-service) used by both step 3 (retrieve_documents) of the
LangGraph agent and by the Celery ingestion worker.
"""
import logging
from contextlib import contextmanager
from typing import List, Tuple

import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector

from app.config import settings

logger = logging.getLogger(__name__)

import os
from dotenv import load_dotenv
load_dotenv()

EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIMENSION"))  # text-embedding-3-small


@contextmanager
def get_conn():
    logger.debug("Opening DB connection")
    conn = psycopg2.connect(settings.database_url)
    try:
        register_vector(conn)
        yield conn
    finally:
        conn.close()
        logger.debug("DB connection closed")


def init_db() -> None:
    """Idempotent schema bootstrap. Safe to call on every backend startup."""
    logger.info("Initialising DB schema")
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS documents (
                id BIGSERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                source TEXT,
                embedding VECTOR({EMBEDDING_DIM}) NOT NULL,
                created_at TIMESTAMPTZ DEFAULT now()
            );
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS documents_embedding_idx
            ON documents USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
            """
        )
        conn.commit()
    logger.info("DB schema ready")


def insert_document(content: str, embedding: List[float], source: str = "") -> int:
    logger.debug("Inserting document source=%r len=%d", source, len(content))
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO documents (content, source, embedding) VALUES (%s, %s, %s) RETURNING id;",
            (content, source, embedding),
        )
        doc_id = cur.fetchone()[0]
        conn.commit()
    logger.info("Inserted document id=%d source=%r", doc_id, source)
    return doc_id


def similarity_search(embedding: List[float], limit: int) -> List[Tuple[int, str, str, float]]:
    """Returns (id, content, source, cosine_distance) ordered by closeness."""
    logger.debug("Similarity search limit=%d", limit)
    with get_conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT id, content, source, embedding <=> %s AS distance
            FROM documents
            ORDER BY embedding <=> %s
            LIMIT %s;
            """,
            (embedding, embedding, limit),
        )
        rows = cur.fetchall()
    results = [(r["id"], r["content"], r["source"], r["distance"]) for r in rows]
    logger.info("Similarity search returned %d rows", len(results))
    return results
