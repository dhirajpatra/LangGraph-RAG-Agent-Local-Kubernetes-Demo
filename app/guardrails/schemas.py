"""
app/guardrails/schemas.py

Shared result type for both input and output guardrails, so the router
can handle them identically regardless of which check fired.
"""
from typing import Optional

from pydantic import BaseModel


class GuardrailResult(BaseModel):
    passed: bool
    reason: Optional[str] = None
    # The (possibly redacted/truncated) text to actually use going forward.
    # Equal to the original input when passed=True and nothing needed
    # changing.
    sanitized_text: Optional[str] = None
