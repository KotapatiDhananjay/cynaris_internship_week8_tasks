import os

import mlflow
from dotenv import load_dotenv


load_dotenv()


def log_request(
    question: str,
    latency_ms: float,
    retrieved_chunks: int,
    status: str
):
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "file:./mlruns"
    )

    experiment_name = os.getenv(
        "MLFLOW_EXPERIMENT_NAME",
        "W8D1_Production_RAG"
    )

    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run():
            mlflow.log_param("llm_model", os.getenv("OLLAMA_MODEL", "qwen2.5:3b"))
            mlflow.log_param("embedding_model", os.getenv("EMBEDDING_MODEL", "nomic-embed-text"))
            mlflow.log_param("question_length", len(question))

            mlflow.log_metric("latency_ms", latency_ms)
            mlflow.log_metric("retrieved_chunks", retrieved_chunks)
            mlflow.set_tag("request_status", status)

    except Exception:
        # Monitoring failures should not make the RAG API unavailable.
        import logging
        logging.getLogger(__name__).exception(
            "Could not record MLflow request metrics"
        )