import logging
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.rag import ingest_documents, run_rag
from src.monitoring import log_request


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="W8D1 Production RAG API",
    description="Dockerized multi-document Retrieval Augmented Generation API",
    version="1.0.0"
)


class AskRequest(BaseModel):
    question: str = Field(
        min_length=3,
        max_length=2000,
        description="Question to ask the document assistant"
    )


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]
    latency_ms: float
    retrieved_chunks: int


@app.get("/")
def root():
    return {
        "message": "W8D1 Production RAG API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "production-rag-api"
    }


@app.post("/ingest")
def ingest():
    try:
        result = ingest_documents()
        return {
            "status": "success",
            **result
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception:
        logger.exception("Document ingestion failed")

        raise HTTPException(
            status_code=503,
            detail="Document ingestion failed. Check Ollama and API logs."
        )


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    question = request.question.strip()

    if len(question) < 3:
        raise HTTPException(
            status_code=422,
            detail="Question must contain at least 3 characters."
        )

    start = time.perf_counter()

    try:
        result = run_rag(question)

        latency_ms = round(
            (time.perf_counter() - start) * 1000,
            2
        )

        log_request(
            question=question,
            latency_ms=latency_ms,
            retrieved_chunks=result["retrieved_chunks"],
            status="success"
        )

        logger.info(
            "RAG request completed in %.2f ms",
            latency_ms
        )

        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "latency_ms": latency_ms,
            "retrieved_chunks": result["retrieved_chunks"]
        }

    except Exception:
        latency_ms = round(
            (time.perf_counter() - start) * 1000,
            2
        )

        log_request(
            question=question,
            latency_ms=latency_ms,
            retrieved_chunks=0,
            status="failed"
        )

        logger.exception("RAG request failed")

        raise HTTPException(
            status_code=503,
            detail="RAG service unavailable. Check Ollama, document indexing and API logs."
        )