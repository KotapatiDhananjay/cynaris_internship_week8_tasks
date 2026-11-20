@'
# W8D2 - RAG Evaluation with Ragas Framework

## Technologies
- Python
- LangChain
- ChromaDB
- Ollama
- Ragas
- Pandas

## Project Features
1. RAG pipeline development.
2. Generation of 10 question-answer pairs.
3. ChromaDB document retrieval.
4. Ragas evaluation.
5. Baseline and optimized comparison.

## Evaluation Metrics
- Faithfulness
- Answer Relevancy
- Context Precision
- Context Recall

## Optimization
Baseline:
- Chunk size: 500
- Retrieval k: 2

Optimized:
- Chunk size: 250
- Retrieval k: 4

## Execution

Activate virtual environment:

    .\venv\Scripts\Activate.ps1

Run:

    python main.py

## Output
The results folder contains:
- baseline_evaluation.csv
- optimized_evaluation.csv
- comparison.csv

## Viva Questions

### 1. What does faithfulness measure in Ragas?
Faithfulness measures whether the generated answer is supported by the retrieved context. It checks whether the answer contains unsupported information.

### 2. Difference between context precision and context recall?
Context precision measures how many retrieved chunks are relevant. Context recall measures whether the retrieved information contains the information needed to answer the question.

### 3. How do you improve answer relevancy?
Improve prompts, retrieve more relevant documents, adjust chunk size and retrieval k, and use a suitable language model.
'@ | Set-Content README.md