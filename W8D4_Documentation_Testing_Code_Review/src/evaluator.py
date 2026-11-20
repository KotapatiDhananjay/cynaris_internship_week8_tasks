"""Evaluation metrics for RAG retrieval systems."""


def precision_at_k(retrieved, relevant, k=3):
    """Calculate the fraction of relevant documents in top-k results."""

    if k <= 0:
        raise ValueError("k must be greater than zero")

    if not retrieved:
        return 0.0

    top_results = retrieved[:k]

    relevant_count = sum(
        1 for document in top_results
        if document in relevant
    )

    return relevant_count / len(top_results)


def recall_at_k(retrieved, relevant, k=3):
    """Calculate the fraction of relevant documents retrieved."""

    if k <= 0:
        raise ValueError("k must be greater than zero")

    if not relevant:
        return 0.0

    relevant_count = sum(
        1 for document in retrieved[:k]
        if document in relevant
    )

    return relevant_count / len(relevant)


def f1_score(precision, recall):
    """Calculate the harmonic mean of precision and recall."""

    if precision < 0 or recall < 0:
        raise ValueError("Scores cannot be negative")

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def evaluate_retrieval(retrieved, relevant, k=3):
    """Calculate all retrieval evaluation metrics."""

    precision = precision_at_k(retrieved, relevant, k)
    recall = recall_at_k(retrieved, relevant, k)

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score(precision, recall)
    }


def validate_dataset(records):
    """Validate the structure of a RAG evaluation dataset."""

    if not isinstance(records, list) or not records:
        raise ValueError("Dataset must be a non-empty list")

    required_fields = {"question", "retrieved", "relevant"}

    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Each record must be a dictionary")

        if not required_fields.issubset(record.keys()):
            raise ValueError("Required dataset fields are missing")

        if not isinstance(record["question"], str) or not record["question"].strip():
            raise ValueError("Question cannot be empty")

        if not isinstance(record["retrieved"], list):
            raise ValueError("Retrieved documents must be a list")

        if not isinstance(record["relevant"], list):
            raise ValueError("Relevant documents must be a list")

    return True