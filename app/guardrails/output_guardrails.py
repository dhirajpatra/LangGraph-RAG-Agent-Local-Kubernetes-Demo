"""
app/guardrails/output_guardrails.py

Fast, deterministic checks run on the agent's answer before it's returned
to the caller and before it's written to the semantic cache — a bad answer
should never get cached and re-served to the next person who asks
something similar.

Checks, in order:
  1. non-empty / not a refusal-shaped placeholder
  2. length limit
  3. PII leak redaction on the way out (in case the LLM echoed something
     from context it shouldn't have)
"""
from app.config import settings
from app.guardrails.input_guardrails import _redact_pii
from app.guardrails.schemas import GuardrailResult

_EMPTY_ANSWER_MARKERS = ("", "none", "n/a")


def check_output(answer: str) -> GuardrailResult:
    if answer is None or answer.strip().lower() in _EMPTY_ANSWER_MARKERS:
        return GuardrailResult(passed=False, reason="Model returned an empty answer.")

    if len(answer) > settings.max_answer_length:
        return GuardrailResult(
            passed=False,
            reason=f"Answer exceeds max length of {settings.max_answer_length} characters.",
        )

    sanitized = answer
    if settings.guardrails_block_pii_in_output:
        sanitized = _redact_pii(answer)

    return GuardrailResult(passed=True, sanitized_text=sanitized)
