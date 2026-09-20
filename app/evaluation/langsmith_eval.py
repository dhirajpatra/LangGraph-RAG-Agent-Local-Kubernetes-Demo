"""
app/evaluation/langsmith_eval.py

Wires the local evaluators (app/evaluation/evaluators.py) and dataset
(app/evaluation/dataset.py) up to LangSmith's evaluate() API, matching the
diagram's "LangSmith Observability & Evaluation" box: traces from the
LangGraph agent already flow into the `LANGSMITH_PROJECT` project (see
app/config.py); this module additionally runs the agent against a
dataset and logs the five judge scores as evaluation feedback in
`LANGSMITH_EVAL_PROJECT`.

Only runs if LANGSMITH_API_KEY is set — call from run_evaluation.py, or
directly:

    python -m app.evaluation.langsmith_eval
"""
from typing import Any, Dict

from app.config import settings
from app.evaluation import evaluators
from app.evaluation.dataset import DEFAULT_DATASET_PATH, load_dataset

DATASET_NAME = "rag-demo-eval-dataset"


def _target(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """The function LangSmith calls for each example — runs the real
    LangGraph agent end to end (real OpenAI/Tavily calls happen here)."""
    from app.agents.rag_graph import run_query

    result = run_query(inputs["question"])
    return {"answer": result["answer"], "contexts": result["contexts_used"]}


def _make_evaluator(name: str, judge_fn):
    """Adapts one of our JudgeScore-returning functions to the
    {key, score, comment} shape LangSmith's evaluate() expects."""

    def _evaluator(run, example) -> Dict[str, Any]:
        outputs = run.outputs or {}
        inputs = example.inputs or {}
        reference = (example.outputs or {}).get("reference_answer")

        if name == "correctness":
            result = judge_fn(inputs["question"], outputs.get("answer", ""), reference)
        elif name == "faithfulness":
            result = judge_fn(outputs.get("answer", ""), outputs.get("contexts", []))
        elif name == "context_relevance":
            result = judge_fn(inputs["question"], outputs.get("contexts", []))
        elif name == "answer_quality":
            result = judge_fn(inputs["question"], outputs.get("answer", ""))
        else:  # user_satisfaction
            result = judge_fn(inputs["question"], outputs.get("answer", ""), reference)

        return {"key": name, "score": result.score / 5.0, "comment": result.rationale}

    return _evaluator


def ensure_dataset(client, examples=None):
    """Creates the LangSmith dataset if it doesn't already exist, and
    (re)populates it from the local JSON file. Idempotent-ish: safe to
    call every run."""
    examples = examples or load_dataset()

    existing = list(client.list_datasets(dataset_name=DATASET_NAME))
    if existing:
        dataset = existing[0]
    else:
        dataset = client.create_dataset(
            DATASET_NAME,
            description=f"Local sample eval set from {DEFAULT_DATASET_PATH.name}",
        )
        client.create_examples(
            inputs=[{"question": e["question"]} for e in examples],
            outputs=[{"reference_answer": e["reference_answer"]} for e in examples],
            dataset_id=dataset.id,
        )
    return dataset


def run_langsmith_evaluation():
    if not settings.langsmith_api_key:
        raise RuntimeError(
            "LANGSMITH_API_KEY is not set — set it in .env to run LangSmith evaluation."
        )

    from langsmith import Client
    from langsmith.evaluation import evaluate

    client = Client(api_key=settings.langsmith_api_key, api_url=settings.langsmith_endpoint)
    dataset = ensure_dataset(client)

    judge_map = {
        "correctness": evaluators.evaluate_correctness,
        "faithfulness": evaluators.evaluate_faithfulness,
        "context_relevance": evaluators.evaluate_context_relevance,
        "answer_quality": evaluators.evaluate_answer_quality,
        "user_satisfaction": evaluators.evaluate_user_satisfaction,
    }
    langsmith_evaluators = [_make_evaluator(name, fn) for name, fn in judge_map.items()]

    return evaluate(
        _target,
        data=dataset.name,
        evaluators=langsmith_evaluators,
        experiment_prefix="rag-demo",
        client=client,
    )


if __name__ == "__main__":
    results = run_langsmith_evaluation()
    print(results)
