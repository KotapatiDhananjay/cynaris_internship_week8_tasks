import os
import pandas as pd

from src.pipeline import retrieve


QUESTIONS = [
    {
        "question": "What is Artificial Intelligence and Machine Learning?",
        "expected_source": "document1.pdf"
    },
    {
        "question": "What are supervised and unsupervised learning?",
        "expected_source": "document1.pdf"
    },
    {
        "question": "What is Natural Language Processing?",
        "expected_source": "document2.pdf"
    },
    {
        "question": "What is tokenization in NLP?",
        "expected_source": "document2.pdf"
    },
    {
        "question": "What is Retrieval Augmented Generation?",
        "expected_source": "document3.pdf"
    },
    {
        "question": "How does RAG improve factual grounding?",
        "expected_source": "document3.pdf"
    },
    {
        "question": "What are vector embeddings?",
        "expected_source": "document4.pdf"
    },
    {
        "question": "What is semantic similarity search?",
        "expected_source": "document4.pdf"
    },
    {
        "question": "What is BM25Retriever in Haystack?",
        "expected_source": "document5.pdf"
    },
    {
        "question": "What is precision in information retrieval?",
        "expected_source": "document5.pdf"
    }
]


def evaluate_method(method):

    rows = []

    for index, item in enumerate(QUESTIONS, start=1):

        result = retrieve(
            item["question"],
            method=method,
            top_k=3
        )

        sources = [
            doc["source"] for doc in result["documents"]
        ]

        relevant_count = sum(
            1 for source in sources
            if source == item["expected_source"]
        )

        precision = relevant_count / 3

        rows.append({
            "question_number": index,
            "question": item["question"],
            "expected_source": item["expected_source"],
            "retrieved_sources": ", ".join(sources),
            "relevant_documents": relevant_count,
            "precision_at_3": precision
        })

        print(f"\nQ{index}: {item['question']}")
        print("Expected:", item["expected_source"])
        print("Retrieved:", sources)
        print(f"Precision@3: {precision:.2f}")

    return rows


def run_evaluation():

    base = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    results_dir = os.path.join(base, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("\n========== BM25 EVALUATION ==========")

    bm25_rows = evaluate_method("bm25")

    print("\n========== DENSE EVALUATION ==========")

    dense_rows = evaluate_method("dense")

    bm25_df = pd.DataFrame(bm25_rows)
    dense_df = pd.DataFrame(dense_rows)

    bm25_df.to_csv(
        os.path.join(results_dir, "bm25_results.csv"),
        index=False
    )

    dense_df.to_csv(
        os.path.join(results_dir, "dense_results.csv"),
        index=False
    )

    comparison = pd.DataFrame({
        "question_number": range(1, 11),
        "question": [q["question"] for q in QUESTIONS],
        "bm25_precision": bm25_df["precision_at_3"],
        "dense_precision": dense_df["precision_at_3"]
    })

    comparison.to_csv(
        os.path.join(results_dir, "comparison.csv"),
        index=False
    )

    print("\n========== FINAL COMPARISON ==========")

    print(comparison.to_string(index=False))

    print("\nBM25 Average Precision@3:",
          round(bm25_df["precision_at_3"].mean(), 4))

    print("Dense Average Precision@3:",
          round(dense_df["precision_at_3"].mean(), 4))

    return comparison