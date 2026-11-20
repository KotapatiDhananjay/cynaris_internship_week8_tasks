# W8D5 - 2M Capstone: Local AI Research Assistant

## Project Description

A lightweight AI research assistant that retrieves information from local PDF and TXT documents and generates answers using a locally running language model.

## Technology Stack

- Python
- LangGraph
- Ollama
- Qwen 2.5 3B
- TF-IDF
- Scikit-learn
- PyPDF
- MLflow
- Pytest

## Features

1. Local AI question answering.
2. PDF and TXT document support.
3. Automatic text chunking.
4. TF-IDF-based retrieval.
5. Cosine similarity.
6. LangGraph AI workflow.
7. Local LLM integration.
8. Source identification.
9. MLflow experiment tracking.
10. Automated testing.

## Installation

    python -m venv venv
    .\venv\Scripts\Activate.ps1
    pip install -r requirements.txt

## Ollama Model

    ollama pull qwen2.5:3b

## Run Application

    python main.py

## Run Tests

    pytest tests -v

## MLflow Dashboard

    mlflow ui

Open:

    http://127.0.0.1:5000

## Add Documents

Place PDF and TXT files inside the data folder.

## Output

    results/research_history.jsonl

## Viva Questions

### 1. Explain what you built today.

I developed a Local AI Research Assistant using LangGraph and Ollama. It loads local documents, retrieves relevant information using TF-IDF and generates answers using a locally running Qwen language model. MLflow is used to track experiments.

### 2. What was the hardest part?

The hardest part was connecting document retrieval with the language model through LangGraph. I solved it by creating separate retrieval and generation nodes.

### 3. What would you improve?

I would integrate Ragas for answer evaluation, ChromaDB for semantic retrieval, CrewAI for multiple research agents and Streamlit for a user-friendly interface.