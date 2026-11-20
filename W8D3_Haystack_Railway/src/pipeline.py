import os
from pathlib import Path

from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.converters import PyPDFToDocument
from haystack.components.writers import DocumentWriter
from haystack.components.retrievers.in_memory import (
    InMemoryBM25Retriever,
    InMemoryEmbeddingRetriever
)

from haystack_integrations.components.embedders.sentence_transformers import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder
)


BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"

MODEL = "sentence-transformers/all-MiniLM-L6-v2"

bm25_store = None
dense_store = None
text_embedder = None


def initialize():

    global bm25_store, dense_store, text_embedder

    pdf_files = sorted(DATA.glob("*.pdf"))

    if not pdf_files:
        raise RuntimeError("No PDF documents found in data folder.")

    converter = PyPDFToDocument()

    converted = converter.run(
        sources=pdf_files
    )["documents"]

    print("PDF documents converted:", len(converted))

    for document in converted:
        document.meta["source"] = Path(
            document.meta.get("file_path", "unknown.pdf")
        ).name

    bm25_store = InMemoryDocumentStore()

    bm25_indexing = Pipeline()

    bm25_indexing.add_component("writer", DocumentWriter(
        document_store=bm25_store
    ))

    bm25_indexing.run({
        "writer": {"documents": converted}
    })

    print("BM25 document store indexed:", len(converted))

    dense_store = InMemoryDocumentStore(
        embedding_similarity_function="cosine"
    )

    document_embedder = SentenceTransformersDocumentEmbedder(
        model=MODEL
    )

    document_embedder.warm_up()

    dense_documents = document_embedder.run(
        converted
    )["documents"]

    dense_indexing = Pipeline()

    dense_indexing.add_component(
        "writer",
        DocumentWriter(document_store=dense_store)
    )

    dense_indexing.run({
        "writer": {"documents": dense_documents}
    })

    print("Dense document store indexed:", len(dense_documents))

    text_embedder = SentenceTransformersTextEmbedder(
        model=MODEL
    )

    text_embedder.warm_up()

    return True


def retrieve(query, method="bm25", top_k=3):

    if bm25_store is None:
        initialize()

    if method == "bm25":

        retriever = InMemoryBM25Retriever(
            document_store=bm25_store,
            top_k=top_k
        )

        documents = retriever.run(
            query=query
        )["documents"]

    elif method == "dense":

        retriever = InMemoryEmbeddingRetriever(
            document_store=dense_store,
            top_k=top_k
        )

        embedding = text_embedder.run(
            text=query
        )["embedding"]

        documents = retriever.run(
            query_embedding=embedding
        )["documents"]

    else:
        raise ValueError("Method must be bm25 or dense")

    # Simple extractive Reader
    answer = "\n".join(
        document.content for document in documents
    )

    return {
        "answer": answer,
        "documents": [
            {
                "source": document.meta.get("source", "unknown"),
                "content": document.content,
                "score": document.score
            }
            for document in documents
        ]
    }