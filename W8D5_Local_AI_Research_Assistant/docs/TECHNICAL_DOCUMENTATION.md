# W8D5 - Local AI Research Assistant

## 1. Project Overview

The Local AI Research Assistant is a prototype that answers research questions using locally stored documents.

It uses a locally running language model to generate answers based on retrieved document content.

## 2. Technologies

- Python
- LangGraph
- Ollama
- Qwen 2.5 3B
- TF-IDF
- Scikit-learn
- PyPDF
- MLflow
- Pytest

## 3. Architecture

1. Document Loader
2. Text Chunking
3. TF-IDF Document Retrieval
4. LangGraph Workflow
5. Local LLM Answer Generation
6. Source Display
7. MLflow Experiment Tracking

## 4. Modules

### document_loader.py
Loads PDF and TXT documents and splits them into chunks.

### retriever.py
Uses TF-IDF and cosine similarity to retrieve relevant document chunks.

### workflow.py
Creates a LangGraph workflow with retrieval and answer-generation nodes.

### research_assistant.py
Connects document loading, retrieval and workflow execution.

### main.py
Runs the interactive research assistant and records experiments using MLflow.

## 5. Local Model

Model: qwen2.5:3b

The model runs through Ollama without requiring a cloud API.

## 6. Execution

Install dependencies:

    pip install -r requirements.txt

Run:

    python main.py

## 7. Testing

    pytest tests -v

## 8. MLflow Tracking

    mlflow ui

Dashboard:

    http://127.0.0.1:5000

## 9. Output

Research questions and generated answers are stored in:

    results/research_history.jsonl

MLflow records experiment parameters, metrics and answer artifacts.

## 10. Future Improvements

- Add Ragas evaluation.
- Add ChromaDB vector retrieval.
- Support DOCX documents.
- Add a Streamlit user interface.
- Add multi-agent research using CrewAI.
- Improve answer evaluation and source citations.