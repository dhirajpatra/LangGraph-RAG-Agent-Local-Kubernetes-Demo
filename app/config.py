"""
Central settings object. Reads from environment variables, which in the
cluster are injected from ConfigMap "rag-config" (non-secret) and
Secret "rag-secrets" (API keys) via envFrom in every Deployment.
"""
import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # External managed APIs
    openai_api_key: str = ""
    tavily_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"

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
    vector_search_limit: int = 5
    rerank_top_n: int = 3
    semantic_cache_threshold: float = 0.92

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
