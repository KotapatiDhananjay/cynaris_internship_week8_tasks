# W8D3 - Deployment: Haystack API on Railway

## Project Overview

This project implements a document retrieval system using Haystack.

## Technologies

- Python
- Haystack 2.x
- BM25 Retriever
- Dense Retriever
- Sentence Transformers
- FastAPI
- Docker
- Railway

## Dataset

Five PDF documents:
1. Artificial Intelligence and Machine Learning
2. Natural Language Processing
3. Retrieval Augmented Generation
4. Vector Databases and Embeddings
5. Haystack and Information Retrieval

## Retrieval Methods

### BM25 Retriever
Uses keyword matching to retrieve relevant documents.

### Dense Retriever
Uses vector embeddings and semantic similarity to retrieve relevant documents.

## Evaluation

Ten questions are used to compare both retrieval methods.

Precision@3 = Relevant retrieved documents / 3

## Execution

Activate environment:

    .\venv\Scripts\Activate.ps1

Run evaluation:

    python main.py

Run API:

    uvicorn api:app --reload

Open:

    http://127.0.0.1:8000/docs

## API Endpoints

GET /
POST /search

Example request:

    {
        "question": "What is RAG?",
        "method": "dense",
        "top_k": 3
    }

## Output

- results/bm25_results.csv
- results/dense_results.csv
- results/comparison.csv

## Viva Questions

### 1. Explain what you built today.

I built a document retrieval system using Haystack. I indexed five PDF documents and implemented BM25 and Dense Retrieval. I evaluated both methods using ten questions and compared their Precision@3 scores.

### 2. What was the hardest part?

The hardest part was integrating dense retrieval with the document store. I solved it by generating document embeddings and query embeddings using Sentence Transformers.

### 3. What would you improve?

I would use a persistent vector database, add an LLM-based answer generator, improve retrieval accuracy and deploy the API with production monitoring.