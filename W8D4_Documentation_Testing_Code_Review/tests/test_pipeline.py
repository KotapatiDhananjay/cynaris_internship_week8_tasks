import pytest

from src.pipeline import run_pipeline


def sample_records():
    return [
        {
            "question": "What is AI?",
            "retrieved": ["doc1", "doc2", "doc3"],
            "relevant": ["doc1", "doc2"]
        },
        {
            "question": "What is RAG?",
            "retrieved": ["doc4", "doc5", "doc6"],
            "relevant": ["doc4"]
        }
    ]


def test_pipeline_returns_two_results():
    results, summary = run_pipeline(sample_records())

    assert len(results) == 2
    assert "precision" in summary
    assert "recall" in summary
    assert "f1_score" in summary


def test_pipeline_score_range():
    results, summary = run_pipeline(sample_records())

    for score in summary.values():
        assert 0.0 <= score <= 1.0


def test_pipeline_empty_dataset():
    with pytest.raises(ValueError):
        run_pipeline([])


def test_pipeline_invalid_k():
    with pytest.raises(ValueError):
        run_pipeline(sample_records(), k=0)