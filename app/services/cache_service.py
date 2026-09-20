"""
app/services/cache_service.py

Redis semantic cache — matches "2. check_cache / Redis semantic cache" and
"8. write_cache / Store semantic answer" in the LangGraph diagram.

Uses Redis Stack's vector search (RediSearch + HNSW) so a *semantically*
similar question (not just an exact string match) can hit the cache.
Requires the redis-stack-server image (see k8s/infrastructure.yaml).

(Moved here from app/cache.py — app/cache.py now just re-exports these
functions so any old imports keep working.)
"""
import os
import struct
from typing import List, Optional
from dotenv import load_dotenv

import numpy as np
import redis
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from app.config import settings
load_dotenv()

INDEX_NAME = "idx:semantic_cache"
PREFIX = "cache:"
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 1536))
SEMANTIC_CACHE_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", 0.8))

_client: Optional[redis.Redis] = None


def get_client() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.from_url(settings.redis_url)
    return _client


def ensure_index() -> None:
    """Idempotent: create the vector index once, ignore if it already exists."""
    r = get_client()
    try:
        r.ft(INDEX_NAME).info()
        return  # already exists
    except redis.ResponseError:
        pass

    schema = (
        TextField("question"),
        TextField("answer"),
        VectorField(
            "embedding",
            "HNSW",
            {
                "TYPE": "FLOAT32",
                "DIM": EMBEDDING_DIM,
                "DISTANCE_METRIC": "COSINE",
            },
        ),
    )
    r.ft(INDEX_NAME).create_index(
        schema, definition=IndexDefinition(prefix=[PREFIX], index_type=IndexType.HASH)
    )


def _to_bytes(vec: List[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def check_cache(embedding: List[float]):
    """Returns cached answer string if a semantically close question exists, else None."""
    ensure_index()
    r = get_client()
    query = (
        Query("*=>[KNN 1 @embedding $vec AS score]")
        .sort_by("score")
        .return_fields("answer", "score")
        .dialect(2)
    )
    try:
        results = r.ft(INDEX_NAME).search(query, query_params={"vec": _to_bytes(embedding)})
    except redis.ResponseError:
        return None

    if not results.docs:
        return None

    doc = results.docs[0]
    score = float(doc.score)  # cosine distance: 0 = identical
    similarity = 1 - score
    if similarity >= settings.semantic_cache_threshold:
        return doc.answer
    return None


def write_cache(question: str, embedding: List[float], answer: str) -> None:
    ensure_index()
    r = get_client()
    key = f"{PREFIX}{abs(hash(question))}"
    r.hset(
        key,
        mapping={
            "question": question,
            "answer": answer,
            "embedding": np.array(embedding, dtype=np.float32).tobytes(),
        },
    )
