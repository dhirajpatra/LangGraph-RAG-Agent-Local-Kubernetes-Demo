"""
app/evaluation/evaluators.py

The five LLM-as-judge checks shown in the diagram's "Evaluation (LLM as
Judge) / LangSmith Evaluations" box:
  - Correctness
  - Faithfulness (Groundedness)
  - Context Relevance
  - Answer Quality
  - User Satisfaction

Each judge call uses `settings.eval_judge_model` (defaults to gpt-4o-mini,
same as the diagram's "LLM Judge / gpt-4o-mini (or other model)"), asked
for a structured 1-5 score plus a one-line rationale. These are meant for
offline/batch scoring (see run_evaluation.py and langsmith_eval.py) — NOT
per-request gating, which is what app/guardrails/ is for.
"""
from typing import List, Optional

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.config import settings


class JudgeScore(BaseModel):
    score: int = Field(ge=1, le=5, description="1 (worst) to 5 (best)")
    rationale: str = Field(description="one short sentence justifying the score")


def _judge() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.eval_judge_model, api_key=settings.openai_api_key, temperature=0.0
    )


def _score(system_prompt: str, human_prompt: str) -> JudgeScore:
    judge = _judge().with_structured_output(JudgeScore)
    return judge.invoke([("system", system_prompt), ("human", human_prompt)])


def evaluate_correctness(question: str, answer: str, reference_answer: str) -> JudgeScore:
    """Does the answer match the reference answer's factual content?"""
    return _score(
        "You grade the correctness of an AI answer against a reference answer. "
        "Score 5 = fully correct, 1 = completely wrong or contradicts the reference.",
        f"Question: {question}\n\nReference answer: {reference_answer}\n\n"
        f"AI answer to grade: {answer}",
    )


def evaluate_faithfulness(answer: str, contexts: List[str]) -> JudgeScore:
    """Is every claim in the answer grounded in the provided context
    (i.e. no hallucination beyond what was retrieved)?"""
    joined_context = "\n---\n".join(contexts) or "(no context was provided)"
    return _score(
        "You grade whether an AI answer is faithful to (grounded in) the provided "
        "context, with no unsupported claims. Score 5 = fully grounded, "
        "1 = mostly fabricated / contradicts the context.",
        f"Context:\n{joined_context}\n\nAI answer to grade: {answer}",
    )


def evaluate_context_relevance(question: str, contexts: List[str]) -> JudgeScore:
    """Was the retrieved context actually relevant to the question?"""
    joined_context = "\n---\n".join(contexts) or "(no context was retrieved)"
    return _score(
        "You grade how relevant the retrieved context is to the question, "
        "independent of whether a good answer was ultimately produced. "
        "Score 5 = highly relevant, 1 = irrelevant.",
        f"Question: {question}\n\nRetrieved context:\n{joined_context}",
    )


def evaluate_answer_quality(question: str, answer: str) -> JudgeScore:
    """Is the answer clear, complete, and well-written, independent of
    factual correctness?"""
    return _score(
        "You grade the overall quality (clarity, completeness, conciseness) "
        "of an AI answer, independent of whether it is factually correct. "
        "Score 5 = excellent, 1 = confusing or unusably incomplete.",
        f"Question: {question}\n\nAI answer to grade: {answer}",
    )


def evaluate_user_satisfaction(
    question: str, answer: str, reference_answer: Optional[str] = None
) -> JudgeScore:
    """Simulated end-user satisfaction — a proxy used when no real user
    feedback signal exists yet. Swap this out for real thumbs-up/down data
    once you're collecting it."""
    ref_line = f"\n\nA strong reference answer looks like: {reference_answer}" if reference_answer else ""
    return _score(
        "You act as a typical end user judging whether this answer would "
        "satisfy them. Score 5 = fully satisfied, 1 = would be frustrated.",
        f"Question: {question}\n\nAI answer: {answer}{ref_line}",
    )


def evaluate_all(
    question: str,
    answer: str,
    contexts: List[str],
    reference_answer: Optional[str] = None,
) -> dict:
    """Runs every judge and returns a flat dict, e.g. for logging as a row
    in a results table or as LangSmith evaluation feedback."""
    results = {
        "faithfulness": evaluate_faithfulness(answer, contexts),
        "context_relevance": evaluate_context_relevance(question, contexts),
        "answer_quality": evaluate_answer_quality(question, answer),
        "user_satisfaction": evaluate_user_satisfaction(question, answer, reference_answer),
    }
    if reference_answer:
        results["correctness"] = evaluate_correctness(question, answer, reference_answer)
    return results
