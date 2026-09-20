"""
app/evaluation/run_evaluation.py

CLI entrypoint for evaluation. Two modes:

  Local (always works, no LangSmith account needed):
    python -m app.evaluation.run_evaluation
    -> runs the real LangGraph agent over app/evaluation/data/eval_dataset.json,
       scores each answer with the five LLM-as-judge evaluators, and prints
       a summary table. Uses real OpenAI (and possibly Tavily) calls.

  LangSmith (needs LANGSMITH_API_KEY):
    python -m app.evaluation.run_evaluation --langsmith
    -> also uploads the dataset to LangSmith and runs evaluate() there, so
       results show up in the LangSmith UI under LANGSMITH_EVAL_PROJECT.
"""
import argparse
import statistics
import sys

from app.config import settings
from app.evaluation import evaluators
from app.evaluation.dataset import load_dataset


def run_local_evaluation() -> None:
    from app.agents.rag_graph import run_query

    dataset = load_dataset()
    all_scores: dict[str, list[int]] = {}

    print(f"Running local evaluation over {len(dataset)} examples...\n")

    for i, example in enumerate(dataset, start=1):
        question = example["question"]
        reference = example["reference_answer"]

        result = run_query(question)
        answer = result["answer"]
        contexts = result["contexts_used"]

        scores = evaluators.evaluate_all(question, answer, contexts, reference)

        print(f"[{i}/{len(dataset)}] {question}")
        print(f"  strategy: {result['strategy']}")
        print(f"  answer:   {answer[:200]}{'...' if len(answer) > 200 else ''}")
        for metric, judge_score in scores.items():
            print(f"  {metric:<18} {judge_score.score}/5 — {judge_score.rationale}")
            all_scores.setdefault(metric, []).append(judge_score.score)
        print()

    print("=" * 60)
    print("Summary (mean score out of 5)")
    print("=" * 60)
    for metric, values in all_scores.items():
        print(f"  {metric:<18} {statistics.mean(values):.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG agent evaluation.")
    parser.add_argument(
        "--langsmith",
        action="store_true",
        help="Also upload the dataset and run evaluate() via LangSmith.",
    )
    args = parser.parse_args()

    if args.langsmith:
        if not settings.langsmith_api_key:
            print("LANGSMITH_API_KEY is not set in .env — cannot run --langsmith mode.", file=sys.stderr)
            sys.exit(1)
        from app.evaluation.langsmith_eval import run_langsmith_evaluation

        results = run_langsmith_evaluation()
        print(results)
    else:
        run_local_evaluation()


if __name__ == "__main__":
    main()
