import os
import json
import pandas as pd

from src.rag_pipeline import build_rag, generate_answer
from src.evaluator import run_evaluation


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR, "data", "qa_dataset.json"
)

RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)


def run_pipeline(chunk_size, k, name):

    print("\n" + "=" * 55)
    print(f"RUNNING: {name}")
    print("=" * 55)

    retriever, llm = build_rag(
        chunk_size=chunk_size,
        k=k,
        collection=name
    )

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        questions = json.load(file)

    records = []

    for index, item in enumerate(questions, start=1):

        print(f"\nProcessing Q&A {index}/10")

        answer, contexts = generate_answer(
            item["question"],
            retriever,
            llm
        )

        records.append({
            "question": item["question"],
            "answer": answer,
            "contexts": contexts,
            "ground_truth": item["ground_truth"]
        })

        print("Question:", item["question"])
        print("Answer:", answer)

    output_path = os.path.join(
        RESULTS_DIR,
        f"{name}_evaluation.csv"
    )

    scores = run_evaluation(records, output_path)

    print("\nEvaluation Scores:")

    for metric, score in scores.items():
        print(f"{metric}: {score:.4f}" if pd.notna(score) else f"{metric}: N/A")

    return scores


def main():

    print("\nRAG EVALUATION USING RAGAS")
    print("Dataset: 10 Question and Answer Pairs")

    print("\nSTEP 1: BASELINE EVALUATION")

    baseline = run_pipeline(
        chunk_size=500,
        k=2,
        name="baseline"
    )

    print("\nSTEP 2: OPTIMIZED EVALUATION")

    optimized = run_pipeline(
        chunk_size=250,
        k=4,
        name="optimized"
    )

    comparison = pd.DataFrame({
        "Metric": list(baseline.keys()),
        "Baseline": list(baseline.values()),
        "Optimized": list(optimized.values())
    })

    comparison["Improvement"] = (
        comparison["Optimized"] - comparison["Baseline"]
    )

    comparison.to_csv(
        os.path.join(RESULTS_DIR, "comparison.csv"),
        index=False
    )

    print("\n" + "=" * 55)
    print("FINAL COMPARISON")
    print("=" * 55)

    print(comparison.to_string(index=False))

    print("\nAll evaluations completed!")
    print("Check the results folder for CSV files.")


if __name__ == "__main__":
    main()