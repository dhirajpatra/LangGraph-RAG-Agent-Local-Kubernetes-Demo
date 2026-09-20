"""
Unit tests for app/evaluation/evaluators.py. The ChatOpenAI judge is
monkeypatched (same pattern as tests/unit/test_grading_service.py) so no
OpenAI API key or network call is needed.
"""
from app.evaluation import evaluators
from app.evaluation.evaluators import JudgeScore


class _FakeStructuredRunnable:
    def __init__(self, ret):
        self._ret = ret

    def invoke(self, messages):
        return self._ret


class _FakeJudge:
    def __init__(self, ret):
        self._ret = ret

    def with_structured_output(self, model):
        return _FakeStructuredRunnable(self._ret)


def test_evaluate_correctness_returns_judge_score(monkeypatch):
    fake = JudgeScore(score=5, rationale="matches the reference exactly")
    monkeypatch.setattr(evaluators, "_judge", lambda: _FakeJudge(fake))

    result = evaluators.evaluate_correctness(
        "What is Kubernetes?", "It orchestrates containers.", "A container orchestration platform."
    )

    assert result.score == 5
    assert "matches" in result.rationale


def test_evaluate_faithfulness_flags_ungrounded_answer(monkeypatch):
    fake = JudgeScore(score=1, rationale="answer invents facts not in context")
    monkeypatch.setattr(evaluators, "_judge", lambda: _FakeJudge(fake))

    result = evaluators.evaluate_faithfulness("made up answer", ["unrelated context"])

    assert result.score == 1


def test_evaluate_context_relevance(monkeypatch):
    fake = JudgeScore(score=4, rationale="mostly on topic")
    monkeypatch.setattr(evaluators, "_judge", lambda: _FakeJudge(fake))

    result = evaluators.evaluate_context_relevance("question", ["some relevant context"])

    assert result.score == 4


def test_evaluate_answer_quality(monkeypatch):
    fake = JudgeScore(score=3, rationale="a bit terse but correct")
    monkeypatch.setattr(evaluators, "_judge", lambda: _FakeJudge(fake))

    result = evaluators.evaluate_answer_quality("question", "short answer")

    assert result.score == 3


def test_evaluate_user_satisfaction(monkeypatch):
    fake = JudgeScore(score=4, rationale="would likely satisfy the user")
    monkeypatch.setattr(evaluators, "_judge", lambda: _FakeJudge(fake))

    result = evaluators.evaluate_user_satisfaction("question", "answer")

    assert result.score == 4


def test_evaluate_all_runs_every_metric_and_includes_correctness_when_reference_given(monkeypatch):
    fake = JudgeScore(score=5, rationale="great")
    monkeypatch.setattr(evaluators, "_judge", lambda: _FakeJudge(fake))

    results = evaluators.evaluate_all(
        "question", "answer", ["context"], reference_answer="reference answer"
    )

    assert set(results.keys()) == {
        "faithfulness",
        "context_relevance",
        "answer_quality",
        "user_satisfaction",
        "correctness",
    }
    assert all(v.score == 5 for v in results.values())


def test_evaluate_all_skips_correctness_without_reference(monkeypatch):
    fake = JudgeScore(score=5, rationale="great")
    monkeypatch.setattr(evaluators, "_judge", lambda: _FakeJudge(fake))

    results = evaluators.evaluate_all("question", "answer", ["context"])

    assert "correctness" not in results
