"""
Central settings object. Reads from environment variables, which in the
cluster are injected from ConfigMap "rag-config" (non-secret) and
Secret "rag-secrets" (API keys) via envFrom in every Deployment.
"""
import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIM", 1536))
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
SEMANTIC_CACHE_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", 0.92))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", 3))
VECTOR_SEARCH_LIMIT = int(os.getenv("VECTOR_SEARCH_LIMIT", 5))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # External managed APIs
    openai_api_key: str = ""
    tavily_api_key: str = ""
    openai_embedding_model: str = OPENAI_EMBEDDING_MODEL
    embedding_dimension: int = EMBEDDING_DIMENSION  
    openai_chat_model: str = OPENAI_CHAT_MODEL

    # LangSmith
    langsmith_api_key: str = ""
    langsmith_tracing: bool = True
    langsmith_project: str = "FinalProject"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    # Infra
    database_url: str = "postgresql://raguser:ragpass@localhost:5432/ragdb"
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "amqp://guest:guest@localhost:5672//"
    celery_result_backend: str = "redis://localhost:6379/1"

    # RAG tuning
    vector_search_limit: int = VECTOR_SEARCH_LIMIT
    rerank_top_n: int = RERANK_TOP_N
    semantic_cache_threshold: float = SEMANTIC_CACHE_THRESHOLD

    # Frontend
    backend_url: str = "http://localhost:8000"


settings = Settings()

# Wire up LangSmith tracing via env vars expected by the langsmith SDK
if settings.langsmith_tracing:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    if settings.langsmith_api_key:
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
