"""
app/guardrails/input_guardrails.py

Fast, deterministic (no LLM call) checks run on every incoming question
before it reaches the LangGraph agent — sits between the FastAPI router and
`app/agents/rag_graph.py`. Kept rule-based on purpose: guardrails run on
every single request, so they need to add near-zero latency and cost.
(The LLM-as-judge checks in app/evaluation/ are for offline/batch scoring,
not per-request gating.)

Checks, in order:
  1. length limit
  2. prompt-injection heuristics ("ignore previous instructions", etc.)
  3. blocked-topic keywords
  4. PII redaction (emails, phone numbers, card-like numbers) — redacts
     rather than blocks, so a question isn't rejected outright for
     incidentally containing a phone number
"""
import re
from typing import List

from app.config import settings
from app.guardrails.schemas import GuardrailResult

# Heuristic phrases commonly used to try to override a system prompt or
# exfiltrate instructions. Not exhaustive — this is a cheap first line of
# defense, not a substitute for the model's own training.
PROMPT_INJECTION_PATTERNS = [
    r"ignore (all |any )?previous instructions",
    r"ignore (all |any )?prior instructions",
    r"disregard (all |any )?(previous|prior|above) instructions",
    r"reveal (your |the )?system prompt",
    r"what (is|are) your (system prompt|instructions)",
    r"you are now (in )?(developer|debug|dan) mode",
    r"act as if you have no (restrictions|guidelines|filters)",
]

BLOCKED_TOPIC_KEYWORDS: List[str] = [
    # Placeholder list — replace/extend with whatever's actually out of
    # scope for your deployment (e.g. topics unrelated to the ingested
    # knowledge base, or categories your org wants hard-blocked).
]

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
CARD_LIKE_RE = re.compile(r"\b(?:\d[ -]*?){13,16}\b")

_INJECTION_RE = re.compile("|".join(PROMPT_INJECTION_PATTERNS), re.IGNORECASE)


def _redact_pii(text: str) -> str:
    text = EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    text = PHONE_RE.sub("[REDACTED_PHONE]", text)
    text = CARD_LIKE_RE.sub("[REDACTED_NUMBER]", text)
    return text


def check_input(question: str) -> GuardrailResult:
    if not question or not question.strip():
        return GuardrailResult(passed=False, reason="Question is empty.")

    if len(question) > settings.max_question_length:
        return GuardrailResult(
            passed=False,
            reason=f"Question exceeds max length of {settings.max_question_length} characters.",
        )

    if _INJECTION_RE.search(question):
        return GuardrailResult(
            passed=False,
            reason="Question matched a prompt-injection heuristic and was blocked.",
        )

    lowered = question.lower()
    for keyword in BLOCKED_TOPIC_KEYWORDS:
        if keyword.lower() in lowered:
            return GuardrailResult(
                passed=False, reason=f"Question references a blocked topic: '{keyword}'."
            )

    sanitized = _redact_pii(question)
    return GuardrailResult(passed=True, sanitized_text=sanitized)
