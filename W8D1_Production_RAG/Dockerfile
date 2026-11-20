FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DOCUMENTS_DIRECTORY=/app/data/documents
ENV MLFLOW_TRACKING_URI=file:/app/mlruns
ENV MLFLOW_EXPERIMENT_NAME=W8D1_Production_RAG
ENV TOP_K=3

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt "uvicorn[standard]"

COPY api ./api
COPY src ./src
COPY data ./data

RUN mkdir -p /app/mlruns

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
