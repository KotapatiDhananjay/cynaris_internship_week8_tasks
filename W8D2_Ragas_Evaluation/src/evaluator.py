import pandas as pd

from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_ollama import ChatOllama, OllamaEmbeddings


def run_evaluation(records, output_path):

    dataset = Dataset.from_list(records)

    evaluator_llm = LangchainLLMWrapper(
        ChatOllama(
            model="qwen2.5:3b",
            temperature=0
        )
    )

    evaluator_embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(
            model="nomic-embed-text"
        )
    )

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ]

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        raise_exceptions=False
    )

    dataframe = result.to_pandas()

    dataframe.to_csv(output_path, index=False)

    scores = {}

    for metric in [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall"
    ]:
        scores[metric] = pd.to_numeric(
            dataframe[metric],
            errors="coerce"
        ).mean()

    return scores