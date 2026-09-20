"""
app/services/generation_service.py

Matches "7. generate_answer / ChatPromptTemplate + ChatOpenAI" in the
diagram.
"""
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.config import settings

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer the question using ONLY the "
            "provided context. If the context is insufficient, say so plainly.",
        ),
        ("human", "Question: {question}\n\nContext:\n{context}"),
    ]
)


def generate_answer(question: str, contexts: List[str]) -> str:
    context_text = "\n---\n".join(contexts) or "(no context found)"
    chat = ChatOpenAI(
        model=settings.openai_chat_model, api_key=settings.openai_api_key, temperature=0.2
    )
    chain = ANSWER_PROMPT | chat
    result = chain.invoke({"question": question, "context": context_text})
    return result.content
