# High-Level Design — LangGraph RAG Agent (Local Kubernetes Demo)

## 1. Purpose
A retrieval-augmented generation service that answers questions over
ingested documents, with a semantic cache, a local-vs-web retrieval
strategy, and full observability. Runs entirely on a single-node minikube
cluster on one laptop.

## 2. Component map

| Layer | Component | Implementation |
|---|---|---|
| UI | Streamlit frontend | `app/frontend.py` — Deployment `rag-frontend`, Service `rag-frontend-service` (NodePort 30085) |
| API | FastAPI backend | `app/main.py`, `app/routers/rag.py` — Deployment `rag-backend` (x2), Service `rag-backend-service` (ClusterIP :8000) |
| Async ingestion | Celery worker | `app/celery_worker.py` — Deployment `rag-worker`, broker RabbitMQ, result backend Redis |
| Agent orchestration | LangGraph state machine | `app/agents/rag_graph.py` — thin wiring only, delegates to `app/services/*` |
| Business logic | Services layer | `app/services/{embedding,cache,retrieval,grading,generation,web_search,ingestion}_service.py` |
| Guardrails | Input/output safety checks | `app/guardrails/{input,output}_guardrails.py` — rule-based, run as graph nodes |
| Evaluation | LLM-as-judge scoring | `app/evaluation/{evaluators,dataset,langsmith_eval,run_evaluation}.py` |
| Vector store | Postgres + pgvector | Deployment `postgres`, Service `postgres-service` |
| Semantic cache | Redis Stack (RediSearch/HNSW) | Deployment `redis`, Service `redis-service` |
| Task broker | RabbitMQ | Deployment `rabbitmq`, Service `rabbitmq-service` |
| Config | ConfigMap `rag-config` | `k8s/config.yaml` |
| Secrets | Secret `rag-secrets` | created imperatively by `setup.sh` from `.env` |
| Observability | LangSmith | tracing enabled via `LANGSMITH_*` env vars, project `FinalProject` |
| External APIs | OpenAI (embeddings + chat), Tavily (web search) | called directly from the backend pods — the only calls that leave the laptop |

## 3. Services layer

Splitting business logic out of the graph nodes keeps `rag_graph.py` as pure
orchestration and makes each capability independently testable/reusable:

- **embedding_service** — wraps `OpenAIEmbeddings`, used by both the agent
  (`embed_question`) and ingestion (`ingestion_service`).
- **cache_service** — Redis Stack vector KNN lookup/write for the semantic
  cache (`check_cache` / `write_cache`).
- **retrieval_service** — pgvector cosine-distance similarity search
  (`app/db.py` does the raw SQL; this wraps it with the app's tuning
  settings).
- **grading_service** — two structured-output LLM calls: grading whether
  retrieved context is `strong`/`weak`/`empty`, and reranking the top-N
  chunks.
- **web_search_service** — Tavily fallback search when local context is
  insufficient.
- **generation_service** — final answer synthesis via
  `ChatPromptTemplate` + `ChatOpenAI`.
- **ingestion_service** — chunk → embed → insert, used by the Celery task
  and reusable from a standalone script if needed.

## 4. Request flow (query)

```
Browser -> rag-frontend-service (30085) -> Streamlit
        -> rag-backend-service (8000) -> FastAPI /api/v1/advanced/query
        -> LangGraph agent:
             check_input_guardrail
               (blocked) -> return apology, strategy=blocked_input
               (allowed) -> embed_question -> check_cache
                 (hit)  -> return cached answer
                 (miss) -> retrieve_documents -> grade_context
                             (strong) -> rerank_context -> generate_answer
                             (weak/empty) -> web_search -> generate_answer
                           -> check_output_guardrail
                             (blocked) -> return apology, strategy=blocked_output (never cached)
                             (allowed) -> write_cache -> return answer
```

## 5. Guardrails vs. evaluation

Two different layers, easy to conflate:

- **Guardrails** (`app/guardrails/`) run on *every* request, inline, inside
  the graph. Rule-based (regex), not an LLM call, so they add negligible
  latency/cost. They gate what reaches the LLM and what reaches the cache.
- **Evaluation** (`app/evaluation/`) runs *offline*, against a held-out
  dataset, using an LLM judge (`EVAL_JUDGE_MODEL`) to score Correctness,
  Faithfulness, Context Relevance, Answer Quality, and User Satisfaction.
  It's how you measure quality over time, not how you protect a live
  request.

## 6. Request flow (ingestion)

```
Streamlit "Ingest a document" -> POST /api/v1/advanced/ingest-async
  -> enqueue Celery task on RabbitMQ
  -> rag-worker picks it up -> ingestion_service.ingest()
     -> chunk_text -> embedding_service.embed_documents
     -> db.insert_document (Postgres/pgvector)
```

## 7. Delivery pipeline (local minikube)

```
Dockerfile -> docker build -> localhost:5001/rag-app:fixed
           -> docker push (minikube registry addon, port-forwarded to 5001)
           -> kubectl apply -f k8s/{namespace,config,infrastructure,application}.yaml
           -> kubectl rollout restart deploy/rag-backend deploy/rag-worker deploy/rag-frontend
```

See `README.md` for the full step-by-step setup.

## 8. Future work
- Swap OpenAI for a local Ollama-served SLM (see README §8).
- Move beyond the built-in sample eval set — evaluate against real ingested
  documents and questions.
- Horizontal pod autoscaling for `rag-backend` under load.
