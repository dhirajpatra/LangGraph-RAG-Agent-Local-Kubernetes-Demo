"""
app/evaluation/dataset.py

Loads the local sample eval set (app/evaluation/data/eval_dataset.json) —
question + reference_answer pairs used by run_evaluation.py and, if you
choose to upload it, as a LangSmith Dataset (see langsmith_eval.py).

Swap in your own dataset by replacing the JSON file, or point
DATASET_PATH at something else via the eval_dataset_path argument.
"""
import json
from pathlib import Path
from typing import List, TypedDict

DEFAULT_DATASET_PATH = Path(__file__).parent / "data" / "eval_dataset.json"


class EvalExample(TypedDict):
    question: str
    reference_answer: str


def load_dataset(path: Path = DEFAULT_DATASET_PATH) -> List[EvalExample]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
