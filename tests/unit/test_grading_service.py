"""
Unit tests for app/services/grading_service.py. The ChatOpenAI client is
never constructed — grading_service._chat() is monkeypatched to return a
fake chat object whose .with_structured_output(...).invoke(...) returns a
fixed Pydantic instance, so no OpenAI API key or network call is needed.
"""
from app.services.grading_service import ContextGrade, RerankedContext
from app.services import grading_service


class _FakeStructuredRunnable:
    def __init__(self, ret):
        self._ret = ret

    def invoke(self, messages):
        return self._ret


class _FakeChat:
    def __init__(self, ret):
        self._ret = ret

    def with_structured_output(self, model):
        return _FakeStructuredRunnable(self._ret)


def test_grade_context_returns_empty_when_no_contexts():
    # short-circuits before ever calling the LLM
    verdict = grading_service.grade_context("What is X?", [])
    assert verdict == "empty"


def test_grade_context_returns_llm_verdict(monkeypatch):
    fake_grade = ContextGrade(verdict="strong", reason="fully answers the question")
    monkeypatch.setattr(grading_service, "_chat", lambda temperature=0.0: _FakeChat(fake_grade))

    verdict = grading_service.grade_context("What is Kubernetes?", ["Kubernetes is a container orchestrator."])

    assert verdict == "strong"


def test_grade_context_weak_verdict(monkeypatch):
    fake_grade = ContextGrade(verdict="weak", reason="only partially relevant")
    monkeypatch.setattr(grading_service, "_chat", lambda temperature=0.0: _FakeChat(fake_grade))

    verdict = grading_service.grade_context("What is Kubernetes?", ["Some unrelated text."])

    assert verdict == "weak"


def test_rerank_returns_top_n_chunks(monkeypatch):
    fake_rerank = RerankedContext(ranked_chunks=["best chunk", "second best", "third"])
    monkeypatch.setattr(grading_service, "_chat", lambda temperature=0.0: _FakeChat(fake_rerank))
    monkeypatch.setattr(grading_service.settings, "rerank_top_n", 2)

    ranked = grading_service.rerank("question", ["a", "b", "c"])

    assert ranked == ["best chunk", "second best"]
