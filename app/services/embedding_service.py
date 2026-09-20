"""
app/services/embedding_service.py

Wraps LangChain's OpenAIEmbeddings so every caller (the graph's
embed_question node, and the Celery ingestion service) shares one place
that knows which model/key to use.
"""
from functools import lru_cache
from typing import List

from langchain_openai import OpenAIEmbeddings

from app.config import settings


@lru_cache(maxsize=1)
def _client() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.openai_embedding_model, api_key=settings.openai_api_key
    )


def embed_query(text: str) -> List[float]:
    return _client().embed_query(text)


def embed_documents(texts: List[str]) -> List[List[float]]:
    return _client().embed_documents(texts)
