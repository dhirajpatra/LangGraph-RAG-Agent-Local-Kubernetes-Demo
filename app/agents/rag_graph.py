"""
app/agents/rag_graph.py

The LangGraph RAG Agent from the architecture diagram, node for node. Each
node is now a thin call into app/services/*, which holds the actual
business logic — this file is only orchestration/wiring:

  1. embed_question      -> services.embedding_service
  2. check_cache         -> services.cache_service
     -> Cache hit? -- yes -> Return API response
                    -- no  -> 3
  3. retrieve_documents  -> services.retrieval_service (pgvector)
  4. grade_context       -> services.grading_service
     -> Local docs enough? -- yes, local context strong -> 6. rerank_context
                           -- no, weak or empty          -> 5. web_search
  6. rerank_context       -> services.grading_service
  5. web_search           -> services.web_search_service (Tavily)
  7. generate_answer      -> services.generation_service
  8. write_cache          -> services.cache_service
  -> Return API response (strategy + contexts_used)
"""
from typing import List, Literal, Optional, TypedDict

from langgraph.graph import END, StateGraph

from app.services import (
    cache_service,
    embedding_service,
    generation_service,
    grading_service,
    retrieval_service,
    web_search_service,
)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class GraphState(TypedDict, total=False):
    question: str
    embedding: List[float]
    cache_hit: bool
    answer: Optional[str]
    contexts: List[str]
    context_sources: List[str]
    context_grade: Literal["strong", "weak", "empty"]
    used_web_search: bool
    strategy: str


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def embed_question(state: GraphState) -> GraphState:
    vec = embedding_service.embed_query(state["question"])
    return {"embedding": vec}


def check_cache(state: GraphState) -> GraphState:
    cached_answer = cache_service.check_cache(state["embedding"])
    if cached_answer:
        return {"cache_hit": True, "answer": cached_answer, "strategy": "semantic_cache"}
    return {"cache_hit": False}


def route_cache(state: GraphState) -> str:
    return "hit" if state.get("cache_hit") else "miss"


def retrieve_documents(state: GraphState) -> GraphState:
    contexts, sources = retrieval_service.retrieve(state["embedding"])
    return {"contexts": contexts, "context_sources": sources}


def grade_context(state: GraphState) -> GraphState:
    verdict = grading_service.grade_context(state["question"], state.get("contexts") or [])
    return {"context_grade": verdict}


def route_grade(state: GraphState) -> str:
    return "enough" if state.get("context_grade") == "strong" else "not_enough"


def rerank_context(state: GraphState) -> GraphState:
    ranked = grading_service.rerank(state["question"], state.get("contexts") or [])
    return {"contexts": ranked, "used_web_search": False}


def web_search(state: GraphState) -> GraphState:
    contexts, sources = web_search_service.search(state["question"])
    return {
        "contexts": contexts + (state.get("contexts") or []),
        "context_sources": sources + (state.get("context_sources") or []),
        "used_web_search": True,
    }


def generate_answer(state: GraphState) -> GraphState:
    answer = generation_service.generate_answer(state["question"], state.get("contexts") or [])
    strategy = "web_search" if state.get("used_web_search") else "local_pgvector"
    return {"answer": answer, "strategy": strategy}


def write_cache(state: GraphState) -> GraphState:
    if state.get("answer"):
        cache_service.write_cache(state["question"], state["embedding"], state["answer"])
    return {}


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("embed_question", embed_question)
    graph.add_node("check_cache", check_cache)
    graph.add_node("retrieve_documents", retrieve_documents)
    graph.add_node("grade_context", grade_context)
    graph.add_node("rerank_context", rerank_context)
    graph.add_node("web_search", web_search)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("write_cache", write_cache)

    graph.set_entry_point("embed_question")
    graph.add_edge("embed_question", "check_cache")

    graph.add_conditional_edges(
        "check_cache",
        route_cache,
        {"hit": END, "miss": "retrieve_documents"},
    )

    graph.add_edge("retrieve_documents", "grade_context")

    graph.add_conditional_edges(
        "grade_context",
        route_grade,
        {"enough": "rerank_context", "not_enough": "web_search"},
    )

    graph.add_edge("rerank_context", "generate_answer")
    graph.add_edge("web_search", "generate_answer")
    graph.add_edge("generate_answer", "write_cache")
    graph.add_edge("write_cache", END)

    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_query(question: str) -> dict:
    graph = get_graph()
    final_state = graph.invoke({"question": question})
    return {
        "answer": final_state.get("answer"),
        "strategy": final_state.get("strategy", "semantic_cache"),
        "contexts_used": final_state.get("contexts", []),
        "sources": final_state.get("context_sources", []),
    }
