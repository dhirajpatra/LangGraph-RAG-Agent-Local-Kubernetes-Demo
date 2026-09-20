"""
app/cache.py

Kept as a thin backward-compatible shim. The real implementation moved to
app/services/cache_service.py as part of splitting business logic into a
services layer. Existing `from app import cache; cache.check_cache(...)`
calls keep working unchanged.
"""
from app.services.cache_service import (  # noqa: F401
    EMBEDDING_DIM,
    INDEX_NAME,
    PREFIX,
    check_cache,
    ensure_index,
    get_client,
    write_cache,
)
