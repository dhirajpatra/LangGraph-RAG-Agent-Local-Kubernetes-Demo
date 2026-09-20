"""
app/main.py

Matches "FastAPI Backend Pods x2 / app/main.py / Deployment: rag-backend"
in the diagram. Reachable inside the cluster via
K8s Service rag-backend-service (ClusterIP :8000), which the Kubernetes App
Layer Service (rag-backend-service) fronts for POST /api/v1/advanced/*.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db import init_db
from app.routers import rag

app = FastAPI(title="RAG Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(rag.router)


@app.on_event("startup")
def on_startup() -> None:
    # Idempotent: creates the pgvector extension + documents table if missing.
    init_db()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
