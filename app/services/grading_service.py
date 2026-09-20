"""
app/services/grading_service.py

Matches "4. grade_context / ChatOpenAI structured output" and
"6. rerank_context / ChatOpenAI structured output" in the diagram.
"""
import logging
from typing import List, Literal

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.config import settings

logger = logging.getLogger(__name__)


class ContextGrade(BaseModel):
    verdict: Literal["strong", "weak", "empty"] = Field(
        description="strong = local docs fully answer the question; "
        "weak = partially relevant; empty = not relevant at all"
    )
    reason: str = Field(description="one short sentence explaining the verdict")


class RerankedContext(BaseModel):
    ranked_chunks: List[str] = Field(description="the most relevant chunks, best first")


def _chat(temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openai_chat_model,
        api_key=settings.openai_api_key,
        temperature=temperature,
    )


def grade_context(question: str, contexts: List[str]) -> Literal["strong", "weak", "empty"]:
    if not contexts:
        logger.debug("grade_context: no contexts, returning 'empty'")
        return "empty"

    logger.debug("grade_context: grading %d chunks for question=%r", len(contexts), question)
    grader = _chat().with_structured_output(ContextGrade)
    joined = "\n---\n".join(contexts)
    result: ContextGrade = grader.invoke(
        [
            (
                "system",
                "You grade whether retrieved context is sufficient to answer a "
                "question. Respond with a strict verdict.",
            ),
            ("human", f"Question: {question}\n\nContext:\n{joined}"),
        ]
    )
    logger.info("grade_context: verdict=%s reason=%r", result.verdict, result.reason)
    return result.verdict


def rerank(question: str, contexts: List[str]) -> List[str]:
    logger.debug("rerank: %d chunks for question=%r", len(contexts), question)
    reranker = _chat().with_structured_output(RerankedContext)
    result: RerankedContext = reranker.invoke(
        [
            (
                "system",
                f"Rank the following passages by relevance to the question. "
                f"Return only the top {settings.rerank_top_n}.",
            ),
            ("human", f"Question: {question}\n\nPassages:\n" + "\n---\n".join(contexts)),
        ]
    )
    ranked = result.ranked_chunks[: settings.rerank_top_n]
    logger.info("rerank: returned %d chunks", len(ranked))
    return ranked
