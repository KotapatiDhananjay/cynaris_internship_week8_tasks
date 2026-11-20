import pytest

from src.evaluator import (
    precision_at_k,
    recall_at_k,
    f1_score,
    evaluate_retrieval,
    validate_dataset
)


def test_precision():
    retrieved = ["doc1", "doc2", "doc3"]
    relevant = ["doc1", "doc3"]

    assert precision_at_k(retrieved, relevant, 3) == pytest.approx(2 / 3)


def test_recall():
    retrieved = ["doc1", "doc2", "doc3"]
    relevant = ["doc1", "doc3"]

    assert recall_at_k(retrieved, relevant, 3) == 1.0


def test_f1_score():
    assert f1_score(1.0, 1.0) == 1.0


def test_zero_f1():
    assert f1_score(0.0, 0.0) == 0.0


def test_invalid_k():
    with pytest.raises(ValueError):
        precision_at_k(["doc1"], ["doc1"], 0)


def test_empty_retrieval():
    assert precision_at_k([], ["doc1"], 3) == 0.0


def test_evaluate_retrieval():
    result = evaluate_retrieval(
        ["doc1", "doc2"],
        ["doc1"],
        2
    )

    assert result["precision"] == 0.5
    assert result["recall"] == 1.0


def test_valid_dataset():
    records = [{
        "question": "What is RAG?",
        "retrieved": ["doc1"],
        "relevant": ["doc1"]
    }]

    assert validate_dataset(records) is True


def test_empty_dataset():
    with pytest.raises(ValueError):
        validate_dataset([])


def test_missing_fields():
    with pytest.raises(ValueError):
        validate_dataset([{"question": "What is RAG?"}])


def test_empty_question():
    with pytest.raises(ValueError):
        validate_dataset([{
            "question": "",
            "retrieved": [],
            "relevant": []
        }])