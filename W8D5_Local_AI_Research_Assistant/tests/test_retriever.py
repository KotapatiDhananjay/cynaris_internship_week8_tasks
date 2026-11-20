import pytest

from src.retriever import DocumentRetriever
from src.document_loader import split_documents


@pytest.fixture
def sample_chunks():
    return [
        {
            "source": "ai.txt",
            "content": "Artificial Intelligence enables machines to learn."
        },
        {
            "source": "ml.txt",
            "content": "Machine Learning uses data to make predictions."
        },
        {
            "source": "rag.txt",
            "content": "Retrieval Augmented Generation retrieves documents."
        }
    ]


def test_retriever_initialization(sample_chunks):
    retriever = DocumentRetriever(sample_chunks)
    assert retriever is not None


def test_retriever_returns_results(sample_chunks):
    retriever = DocumentRetriever(sample_chunks)

    results = retriever.search("Artificial Intelligence")

    assert len(results) > 0
    assert results[0]["source"] == "ai.txt"


def test_retriever_empty_query(sample_chunks):
    retriever = DocumentRetriever(sample_chunks)

    with pytest.raises(ValueError):
        retriever.search("")


def test_retriever_invalid_k(sample_chunks):
    retriever = DocumentRetriever(sample_chunks)

    with pytest.raises(ValueError):
        retriever.search("AI", top_k=0)


def test_empty_chunks():
    with pytest.raises(ValueError):
        DocumentRetriever([])


def test_document_splitting():

    documents = [{
        "source": "test.txt",
        "content": "Artificial Intelligence " * 100
    }]

    chunks = split_documents(
        documents,
        chunk_size=100,
        overlap=20
    )

    assert len(chunks) > 1


def test_invalid_chunk_size():

    with pytest.raises(ValueError):
        split_documents([], chunk_size=0)