"""Main execution file for W8D4."""

import os
import json
import pandas as pd
import mlflow

from src.pipeline import run_pipeline


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)


# Sample RAG retrieval dataset
records = [
    {
        "question": "What is Artificial Intelligence?",
        "retrieved": ["ai.pdf", "ml.pdf", "rag.pdf"],
        "relevant": ["ai.pdf", "ml.pdf"]
    },
    {
        "question": "What is Machine Learning?",
        "retrieved": ["ml.pdf", "ai.pdf", "nlp.pdf"],
        "relevant": ["ml.pdf"]
    },
    {
        "question": "What is Deep Learning?",
        "retrieved": ["dl.pdf", "ml.pdf", "ai.pdf"],
        "relevant": ["dl.pdf", "ml.pdf"]
    },
    {
        "question": "What is NLP?",
        "retrieved": ["nlp.pdf", "rag.pdf", "ai.pdf"],
        "relevant": ["nlp.pdf"]
    },
    {
        "question": "What is RAG?",
        "retrieved": ["rag.pdf", "ai.pdf", "ml.pdf"],
        "relevant": ["rag.pdf"]
    }
]


def main():

    print("\nW8D4 DOCUMENTATION AND TESTING PROJECT")
    print("=" * 50)

    results, summary = run_pipeline(records, k=3)

    dataframe = pd.DataFrame(results)

    output_file = os.path.join(
        RESULTS_DIR,
        "evaluation_results.csv"
    )

    dataframe.to_csv(output_file, index=False)

    mlflow.set_experiment("W8D4_Code_Quality")

    with mlflow.start_run(run_name="RAG_Evaluation"):

        mlflow.log_param("dataset_size", len(records))
        mlflow.log_param("retrieval_k", 3)

        for metric, value in summary.items():
            mlflow.log_metric(metric, value)

        mlflow.log_artifact(output_file)

    summary_file = os.path.join(
        RESULTS_DIR,
        "summary.json"
    )

    with open(summary_file, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\nEVALUATION RESULTS")
    print(dataframe.to_string(index=False))

    print("\nAVERAGE METRICS")

    for metric, value in summary.items():
        print(f"{metric}: {value:.4f}")

    print("\nResults saved successfully!")
    print("MLflow tracking completed.")


if __name__ == "__main__":
    main()