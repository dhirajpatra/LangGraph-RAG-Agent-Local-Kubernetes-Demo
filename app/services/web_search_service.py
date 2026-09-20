"""
app/services/web_search_service.py

Matches "5. web_search / Tavily internet search" in the diagram — the
fallback path used when local pgvector context is weak or empty.
"""
import logging
from typing import List, Tuple

from tavily import TavilyClient

from app.config import settings

logger = logging.getLogger(__name__)

def search(question: str) -> Tuple[List[str], List[str]]:
    """Returns (contexts, sources) from Tavily."""
    client = TavilyClient(api_key=settings.tavily_api_key)
    resp = client.search(query=question, max_results=settings.vector_search_limit)
    results = resp.get("results", [])
    contexts = [r.get("content", "") for r in results]
    sources = [r.get("url", "") for r in results]
    logger.info("web_search_service: search=%r results=%d", question, len(results))
    return contexts, sources
