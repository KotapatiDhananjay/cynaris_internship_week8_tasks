# W8D1 - Production RAG: Evaluation & Optimisation

## Project Overview

This project implements a Dockerized Multi-Document RAG API.

The application loads PDF and TXT documents, creates embeddings,
stores them in ChromaDB, retrieves relevant information and uses
an Ollama language model to generate answers.

## Technology Stack

- Python
- FastAPI
- LangGraph
- LangChain
- Ollama
- ChromaDB
- MLflow
- Ragas evaluation strategy
- Docker
- GitHub Actions
- Pytest
- Ruff

## Architecture

Documents
   |
   v
Document Loader
   |
   v
Text Splitter
   |
   v
Ollama Embeddings
   |
   v
ChromaDB
   |
   v
LangGraph Retrieval
   |
   v
Ollama LLM
   |
   v
FastAPI Response

## Setup

Create a virtual environment:

    python -m venv venv

Activate it on Windows:

    .\venv\Scripts\Activate.ps1

Install dependencies:

    pip install -r requirements-dev.txt

Copy environment configuration:

    Copy-Item .env.example .env

## Ollama Setup

Install and start Ollama.

Pull the required models:

    ollama pull qwen2.5:3b
    ollama pull nomic-embed-text

## Run Document Ingestion

Start the API:

    uvicorn api.main:app --reload

Open:

    http://localhost:8000/docs

Call POST /ingest to index documents.

## Ask Questions

Use POST /ask with:

    {
      "question": "What is Retrieval Augmented Generation?"
    }

## Run Tests

    pytest -v

## Run Lint

    ruff check api src tests

## Docker

Build:

    docker compose build

Start:

    docker compose up -d

Check logs:

    docker compose logs -f

Stop:

    docker compose down

## Monitoring

See MONITORING.md.

## Evaluation

See evaluation/README.md.

## API Endpoints

- GET /
- GET /health
- POST /ingest
- POST /ask

## CI/CD

GitHub Actions performs linting, testing, Docker image building
and image publishing to GitHub Container Registry.