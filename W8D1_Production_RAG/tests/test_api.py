from fastapi.testclient import TestClient

import api.main as api_module


client = TestClient(api_module.app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["docs"] == "/docs"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_ask_endpoint(monkeypatch):
    def fake_run_rag(question):
        return {
            "answer": "RAG combines retrieval and generation.",
            "sources": [
                {"source": "ai_ml_guide.txt", "page": None}
            ],
            "retrieved_chunks": 1
        }

    monkeypatch.setattr(api_module, "run_rag", fake_run_rag)
    monkeypatch.setattr(api_module, "log_request", lambda **kwargs: None)

    response = client.post(
        "/ask",
        json={"question": "What is RAG?"}
    )

    assert response.status_code == 200
    assert "retrieval" in response.json()["answer"]
    assert response.json()["retrieved_chunks"] == 1


def test_short_question_is_rejected():
    response = client.post(
        "/ask",
        json={"question": "a"}
    )

    assert response.status_code == 422


def test_empty_question_is_rejected():
    response = client.post(
        "/ask",
        json={"question": ""}
    )

    assert response.status_code == 422


def test_ingest_endpoint(monkeypatch):
    def fake_ingest():
        return {
            "documents_loaded": 1,
            "chunks_created": 3,
            "collection": "week8_production_rag"
        }

    monkeypatch.setattr(api_module, "ingest_documents", fake_ingest)

    response = client.post("/ingest")

    assert response.status_code == 200
    assert response.json()["chunks_created"] == 3