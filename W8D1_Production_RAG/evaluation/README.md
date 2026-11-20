# RAG Evaluation with Ragas

Ragas is used to evaluate the quality of a Retrieval Augmented
Generation system.

## Metrics

1. Faithfulness:
   Checks whether the answer is supported by the retrieved context.

2. Answer Relevancy:
   Checks whether the generated answer addresses the question.

3. Context Precision:
   Measures whether relevant retrieved information is ranked well.

4. Context Recall:
   Measures whether the retrieved context contains the information
   required to answer the question.

## Evaluation Dataset

Prepare a dataset containing:
- User questions
- Reference answers
- Retrieved contexts
- Generated answers

## Evaluation Workflow

1. Prepare representative questions.
2. Run each question through the RAG API.
3. Collect the generated answer and retrieved context.
4. Compare answers against reference information.
5. Calculate Ragas metrics.
6. Record evaluation results in MLflow.
7. Compare scores before and after changes.

## Suggested Evaluation Cases

1. What is Artificial Intelligence?
2. Explain supervised learning.
3. What is Retrieval Augmented Generation?
4. Explain the main components of a RAG system.
5. What metrics are used to monitor a production RAG system?
6. Ask a question whose answer is not present in the documents.

## Important

Use a fixed evaluation dataset for meaningful comparisons.
Do not assume that a high answer similarity score alone proves
that a response is factually correct.