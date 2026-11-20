# W8D4 Technical Documentation

## 1. Project Overview

This project implements documentation, testing and code quality practices for an AI/ML RAG evaluation system.

## 2. Technologies Used

- Python
- Pytest
- Pandas
- MLflow

## 3. Project Modules

### evaluator.py
Implements:
- Precision@K
- Recall@K
- F1 Score
- Dataset validation

### pipeline.py
Runs retrieval evaluation and calculates average metrics.

### main.py
Executes the evaluation pipeline, saves CSV and JSON results and tracks metrics using MLflow.

## 4. Testing

Pytest is used for automated unit and integration testing.

Test cases include:
- Precision calculation
- Recall calculation
- F1 calculation
- Invalid input handling
- Empty dataset validation
- Pipeline execution
- Metric range validation

## 5. Evaluation Metrics

Precision = Relevant retrieved documents / Retrieved documents

Recall = Relevant retrieved documents / Total relevant documents

F1 Score = 2 * Precision * Recall / (Precision + Recall)

## 6. Experiment Tracking

MLflow tracks:
- Dataset size
- Retrieval k
- Precision
- Recall
- F1 Score
- Evaluation output artifact

## 7. Output Files

- results/evaluation_results.csv
- results/summary.json

## 8. Conclusion

The project demonstrates clean Python code, automated testing, technical documentation, experiment tracking and code review practices.