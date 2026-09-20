"""
Unit test for app/evaluation/dataset.py — just confirms the bundled sample
dataset loads and has the expected shape.
"""
from app.evaluation.dataset import load_dataset


def test_load_dataset_returns_question_reference_pairs():
    dataset = load_dataset()

    assert len(dataset) > 0
    for example in dataset:
        assert "question" in example
        assert "reference_answer" in example
        assert isinstance(example["question"], str) and example["question"]
        assert isinstance(example["reference_answer"], str) and example["reference_answer"]
