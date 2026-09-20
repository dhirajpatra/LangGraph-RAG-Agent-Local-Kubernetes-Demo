"""
app/agents/rag_graph.py

The LangGraph RAG Agent from the architecture diagram, node for node, plus
an input guardrail before step 1 and an output guardrail after step 7 —
both fast/rule-based (see app/guardrails/), so they add negligible latency:

  0. check_input_guardrail -> guardrails.input_guardrails
     -> blocked? -- yes -> Return API response (strategy: blocked_input)
                -- no  -> 1
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
  7.5. check_output_guardrail -> guardrails.output_guardrails
     -> blocked? -- yes -> Return API response (strategy: blocked_output), skip cache write
                -- no  -> 8
  8. write_cache          -> services.cache_service
  -> Return API response (strategy + contexts_used)

A blocked answer is NEVER written to the semantic cache — the guardrail
runs as part of the graph itself, before the write_cache node, rather than
as an after-the-fact check in the router, so there's no window where a
rejected answer could still get cached and re-served later.
"""
from typing import List, Literal, Optional, TypedDict

from langgraph.graph import END, StateGraph

from app.guardrails import input_guardrails, output_guardrails
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
    input_blocked: bool
    output_blocked: bool
    block_reason: str


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def check_input_guardrail(state: GraphState) -> GraphState:
    result = input_guardrails.check_input(state["question"])
    if not result.passed:
        return {
            "input_blocked": True,
            "answer": f"I can't process this request: {result.reason}",
            "strategy": "blocked_input",
        }
    # Use the sanitized (PII-redacted) question for everything downstream.
    return {"input_blocked": False, "question": result.sanitized_text or state["question"]}


def route_input_guardrail(state: GraphState) -> str:
    return "blocked" if state.get("input_blocked") else "allowed"


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


def check_output_guardrail(state: GraphState) -> GraphState:
    result = output_guardrails.check_output(state.get("answer"))
    if not result.passed:
        return {
            "output_blocked": True,
            "answer": "I'm not able to share that response — it didn't pass a safety check.",
            "strategy": "blocked_output",
            "block_reason": result.reason or "",
        }
    return {"output_blocked": False, "answer": result.sanitized_text or state.get("answer")}


def route_output_guardrail(state: GraphState) -> str:
    return "blocked" if state.get("output_blocked") else "allowed"


def write_cache(state: GraphState) -> Optional[GraphState]:
    if state.get("answer"):
        cache_service.write_cache(state["question"], state["embedding"], state["answer"])
    return None


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("check_input_guardrail", check_input_guardrail)
    graph.add_node("embed_question", embed_question)
    graph.add_node("check_cache", check_cache)
    graph.add_node("retrieve_documents", retrieve_documents)
    graph.add_node("grade_context", grade_context)
    graph.add_node("rerank_context", rerank_context)
    graph.add_node("web_search", web_search)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("check_output_guardrail", check_output_guardrail)
    graph.add_node("write_cache", write_cache)

    graph.set_entry_point("check_input_guardrail")

    graph.add_conditional_edges(
        "check_input_guardrail",
        route_input_guardrail,
        {"blocked": END, "allowed": "embed_question"},
    )

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
    graph.add_edge("generate_answer", "check_output_guardrail")

    graph.add_conditional_edges(
        "check_output_guardrail",
        route_output_guardrail,
        {"blocked": END, "allowed": "write_cache"},
    )

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
