"""
app/celery_worker.py

Matches "Celery Worker Pod / app/celery_worker.py / Deployment: rag-worker"
in the diagram. Consumes ingestion jobs off RabbitMQ ("enqueue ingestion
task") and delegates the actual chunk+embed+insert work to
app/services/ingestion_service.py.
"""
import logging

from celery import Celery

from app.config import settings

logger = logging.getLogger(__name__)

celery_app = Celery(
    "rag_worker",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)


@celery_app.task(name="ingest_document")
def ingest_document_task(text: str, source: str = "manual-upload") -> dict:
    # Imported lazily so worker startup doesn't require an OpenAI key just
    # to boot (only needed when a task actually runs).
    from app.services.ingestion_service import ingest

    logger.info("Task ingest_document started source=%r len=%d", source, len(text))
    try:
        result = ingest(text, source=source)
        logger.info("Task ingest_document completed source=%r result=%s", source, result)
        return result
    except Exception:
        logger.exception("Task ingest_document failed source=%r", source)
        raise
