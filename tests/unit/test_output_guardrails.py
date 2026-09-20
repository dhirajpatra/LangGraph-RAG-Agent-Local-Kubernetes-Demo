"""
Unit tests for app/guardrails/output_guardrails.py — pure rule-based
logic, no mocking needed.
"""
from app.guardrails import output_guardrails
from app.config import settings


def test_normal_answer_passes_unchanged():
    result = output_guardrails.check_output("Kubernetes orchestrates containers.")

    assert result.passed is True
    assert result.sanitized_text == "Kubernetes orchestrates containers."


def test_empty_answer_is_rejected():
    result = output_guardrails.check_output("")

    assert result.passed is False
    assert "empty" in result.reason.lower()


def test_none_answer_is_rejected():
    result = output_guardrails.check_output(None)

    assert result.passed is False


def test_overly_long_answer_is_rejected(monkeypatch):
    monkeypatch.setattr(settings, "max_answer_length", 10)

    result = output_guardrails.check_output("x" * 11)

    assert result.passed is False
    assert "length" in result.reason.lower()


def test_pii_in_answer_is_redacted_when_enabled(monkeypatch):
    monkeypatch.setattr(settings, "guardrails_block_pii_in_output", True)

    result = output_guardrails.check_output("Contact support at help@example.com for more info.")

    assert result.passed is True
    assert "help@example.com" not in result.sanitized_text
    assert "[REDACTED_EMAIL]" in result.sanitized_text


def test_pii_redaction_can_be_disabled(monkeypatch):
    monkeypatch.setattr(settings, "guardrails_block_pii_in_output", False)

    result = output_guardrails.check_output("Contact support at help@example.com for more info.")

    assert result.passed is True
    assert "help@example.com" in result.sanitized_text
