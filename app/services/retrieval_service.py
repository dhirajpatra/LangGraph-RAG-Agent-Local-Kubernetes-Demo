"""
app/services/retrieval_service.py

Wraps app/db.py's pgvector similarity search — matches
"3. retrieve_documents / pgvector similarity search" in the diagram.
"""
from typing import List, Tuple

from app import db
from app.config import settings


def retrieve(embedding: List[float]) -> Tuple[List[str], List[str]]:
    """Returns (contexts, sources) ordered by closeness."""
    rows = db.similarity_search(embedding, settings.vector_search_limit)
    contexts = [r[1] for r in rows]
    sources = [r[2] or f"doc:{r[0]}" for r in rows]
    return contexts, sources
