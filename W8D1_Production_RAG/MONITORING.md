# W8D1 - Production RAG Monitoring Strategy

## 1. Objective

Monitor API reliability, RAG answer quality, retrieval performance,
model availability and resource usage.

## 2. Metrics to Track

### API Metrics
- Total number of requests
- Requests per minute
- Average response latency
- P95 response latency
- HTTP 4xx error count
- HTTP 5xx error count
- API availability

### RAG Quality Metrics
Use Ragas for offline evaluation of:
- Faithfulness
- Answer relevancy
- Context precision
- Context recall

Also track:
- Number of retrieved chunks
- Retrieval relevance
- Empty or unsupported answers
- Source-document coverage

### Infrastructure Metrics
- Docker container health
- CPU utilization
- Memory utilization
- Container restarts
- Ollama availability
- Ollama response time
- ChromaDB errors

### Model and Data Metrics
- LLM model version
- Embedding model version
- Document collection version
- Last successful ingestion time

## 3. Initial Alert Thresholds

| Metric | Initial alert condition |
|---|---|
| API availability | 3 consecutive failed health checks |
| HTTP 5xx errors | Above 5% over 5 minutes |
| P95 latency | Above 10 seconds for 15 minutes |
| Ollama connection | 3 consecutive failures |
| Faithfulness | Below 0.80 |
| Answer relevancy | Below 0.75 |
| Context precision | Below 0.70 |
| Context recall | Below 0.70 |
| Container memory | Above 85% for 10 minutes |

These are starting thresholds. Tune them using actual baseline
measurements before production deployment.

## 4. Evaluation Strategy

Use a fixed set of representative questions and expected answers.

Run Ragas evaluation after:
- Changes to prompts
- Changes to the LLM
- Changes to the embedding model
- Document collection updates
- Retrieval configuration changes

Compare evaluation scores across runs.

Use MLflow to record model parameters, latency, retrieved chunk
counts and evaluation results.

## 5. Re-indexing and Retraining Triggers

### Re-index documents when:
- New documents are added.
- Existing documents are updated or removed.
- The embedding model changes.
- Retrieval quality decreases.
- The knowledge base becomes outdated.

### Consider fine-tuning when:
- A sufficiently large labelled dataset is available.
- Prompt and retrieval improvements do not solve the issue.
- Evaluation shows a consistent model limitation.
- A new model performs better on a controlled evaluation set.

A RAG system usually needs document re-indexing when its knowledge
base changes. It does not automatically require LLM retraining.

## 6. Logging

Record:
- Request timestamp
- Endpoint
- Request duration
- Request status
- Retrieved chunk count
- Model and embedding model versions
- Error details

Do not log passwords, API keys or sensitive user information.

## 7. Incident Response

1. Check the API health endpoint.
2. Inspect Docker logs.
3. Check Ollama availability.
4. Verify ChromaDB connectivity.
5. Inspect retrieved documents.
6. Run the Ragas evaluation dataset.
7. Fix the issue and verify the service again.

## 8. Monitoring Tools

- FastAPI logging: request status and response duration.
- Docker: container health and resource usage.
- MLflow: request metrics and experiment tracking.
- Ragas: retrieval and generation quality evaluation.
- GitHub Actions: build and test status.

## 9. Review Frequency

- API health: continuously.
- Error rates and latency: continuously.
- RAG evaluation: after major changes and on a scheduled cycle.
- Alert thresholds: monthly.