"""
Unit tests for app/services/generation_service.py. ChatOpenAI is swapped
for a LangChain RunnableLambda so the real `ANSWER_PROMPT | chat` pipe
still works end-to-end (exercising the actual prompt template) without any
network call or API key.
"""
from types import SimpleNamespace

from langchain_core.runnables import RunnableLambda

from app.services import generation_service


def _fake_chat_factory(response_text):
    def _fake_llm(prompt_value):
        # A minimal stand-in for an AIMessage: only `.content` is used by
        # generation_service.generate_answer.
        return SimpleNamespace(content=response_text)

    return RunnableLambda(_fake_llm)


def test_generate_answer_returns_llm_content(monkeypatch):
    monkeypatch.setattr(
        generation_service,
        "ChatOpenAI",
        lambda model, api_key, temperature: _fake_chat_factory("Kubernetes is a container orchestrator."),
    )

    answer = generation_service.generate_answer(
        "What is Kubernetes?", ["Kubernetes orchestrates containers."]
    )

    assert answer == "Kubernetes is a container orchestrator."


def test_generate_answer_handles_no_context(monkeypatch):
    captured = {}

    def _fake_llm(prompt_value):
        captured["prompt_text"] = prompt_value.to_string()
        return SimpleNamespace(content="I don't have enough context to answer.")

    monkeypatch.setattr(
        generation_service,
        "ChatOpenAI",
        lambda model, api_key, temperature: RunnableLambda(_fake_llm),
    )

    answer = generation_service.generate_answer("What is Kubernetes?", [])

    assert "no context found" in captured["prompt_text"]
    assert answer == "I don't have enough context to answer."
