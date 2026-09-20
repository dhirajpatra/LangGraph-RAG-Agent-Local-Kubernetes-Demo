"""
app/routers/rag.py

Exposes the two endpoints shown on the "Kubernetes App Layer -> FastAPI
Backend Pods" box of the diagram:

  POST /api/v1/advanced/query        -> runs the LangGraph RAG agent
  POST /api/v1/advanced/ingest-async -> enqueues a Celery ingestion task
"""
from fastapi import APIRouter
from pydantic import BaseModel

from app.celery_worker import ingest_document_task

router = APIRouter(prefix="/api/v1/advanced", tags=["rag"])


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    strategy: str
    contexts_used: list[str]
    sources: list[str]


class IngestRequest(BaseModel):
    text: str
    source: str = "manual-upload"


class IngestResponse(BaseModel):
    task_id: str
    status: str


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    # Imported lazily so the module (and its OpenAI client) is only built
    # once environment variables are guaranteed to be loaded.
    from app.agents.rag_graph import run_query

    result = run_query(req.question)
    return QueryResponse(**result)


@router.post("/ingest-async", response_model=IngestResponse)
def ingest_async(req: IngestRequest) -> IngestResponse:
    task = ingest_document_task.delay(req.text, req.source)
    return IngestResponse(task_id=task.id, status="queued")


@router.get("/ingest-status/{task_id}")
def ingest_status(task_id: str) -> dict:
    from app.celery_worker import celery_app

    result = celery_app.AsyncResult(task_id)
    return {"task_id": task_id, "status": result.status, "result": result.result if result.ready() else None}
