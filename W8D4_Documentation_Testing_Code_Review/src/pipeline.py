"""RAG evaluation pipeline."""

from src.evaluator import (
    evaluate_retrieval,
    validate_dataset
)


def run_pipeline(records, k=3):
    """Evaluate all questions and calculate average metrics."""

    validate_dataset(records)

    results = []

    for record in records:

        scores = evaluate_retrieval(
            retrieved=record["retrieved"],
            relevant=record["relevant"],
            k=k
        )

        results.append({
            "question": record["question"],
            "precision": scores["precision"],
            "recall": scores["recall"],
            "f1_score": scores["f1_score"]
        })

    count = len(results)

    summary = {
        "precision": sum(x["precision"] for x in results) / count,
        "recall": sum(x["recall"] for x in results) / count,
        "f1_score": sum(x["f1_score"] for x in results) / count
    }

    return results, summary
