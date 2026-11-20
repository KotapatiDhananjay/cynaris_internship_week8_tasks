# W8D4 - Documentation, Testing & Code Review

## Project Overview
A Python-based RAG evaluation and code quality project.

## Technologies
- Python
- Pytest
- Pandas
- MLflow

## Features
- Retrieval evaluation
- Precision calculation
- Recall calculation
- F1 score calculation
- Dataset validation
- Automated testing
- MLflow experiment tracking
- Technical documentation

## Installation

    python -m venv venv
    .\venv\Scripts\Activate.ps1
    pip install -r requirements.txt

## Run Application

    python main.py

## Run Tests

    pytest tests -v

## Generate Test Report

    pytest tests -v --tb=short

## View MLflow

    mlflow ui

Open http://127.0.0.1:5000

## Results

Check the results folder for:
- evaluation_results.csv
- summary.json

## Documentation

See the docs folder for technical documentation, code review and self-review.

## Viva Questions

### 1. Explain what you built today.

I developed a RAG evaluation project using Python. I implemented precision, recall and F1 score, created automated tests using Pytest and integrated MLflow for experiment tracking.

### 2. What was the hardest part?

The hardest part was testing different evaluation functions and handling invalid inputs. I solved it by creating separate test cases for each function.

### 3. What would you improve?

I would add real document retrieval, increase test coverage, integrate Ragas and LangGraph, and add automated CI/CD testing using GitHub Actions.