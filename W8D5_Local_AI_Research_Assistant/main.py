"""Main application for W8D5 Local AI Research Assistant."""

import json
from datetime import datetime
from pathlib import Path

import mlflow

from src.research_assistant import build_assistant, ask_question


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)


def main():

    print("\n" + "=" * 55)
    print("       LOCAL AI RESEARCH ASSISTANT")
    print("=" * 55)

    print("\nLoading research documents...")

    assistant, chunks = build_assistant()

    print(f"Documents indexed into {len(chunks)} chunks.")

    mlflow.set_experiment("W8D5_Local_AI_Research_Assistant")

    print("\nAssistant is ready!")
    print("Ask research questions or type 'exit' to stop.\n")

    while True:

        question = input("You: ").strip()

        if question.lower() in ["exit", "quit"]:
            print("Research session completed.")
            break

        if not question:
            print("Please enter a question.")
            continue

        try:

            with mlflow.start_run(
                run_name="Research_Query",
                nested=True
            ):

                mlflow.log_param("model", "qwen2.5:3b")
                mlflow.log_param("retrieval", "TF-IDF")
                mlflow.log_param("question", question[:250])

                result = ask_question(assistant, question)

                answer = result["answer"]
                sources = result.get("sources", [])

                print("\nAssistant:", answer)

                print("\nSources:")
                for source in sources:
                    print("-", source)

                mlflow.log_metric(
                    "retrieved_sources",
                    len(sources)
                )

                mlflow.log_text(
                    answer,
                    "answer.txt"
                )

                record = {
                    "timestamp": datetime.now().isoformat(),
                    "question": question,
                    "answer": answer,
                    "sources": sources
                }

                with open(
                    RESULTS_DIR / "research_history.jsonl",
                    "a",
                    encoding="utf-8"
                ) as file:

                    file.write(
                        json.dumps(record, ensure_ascii=False) + "\n"
                    )

        except Exception as error:
            print("\nError:", error)

        print("\n" + "-" * 55)


if __name__ == "__main__":
    main()