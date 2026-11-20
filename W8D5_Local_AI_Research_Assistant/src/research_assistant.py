"""Initialize and run the local research assistant."""

from pathlib import Path

from src.document_loader import load_documents, split_documents
from src.workflow import create_workflow


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def build_assistant():
    """Load documents and build the LangGraph assistant."""

    documents = load_documents(DATA_DIR)

    if not documents:
        raise ValueError("Add TXT or PDF files to the data folder.")

    chunks = split_documents(
        documents,
        chunk_size=700,
        overlap=100
    )

    assistant = create_workflow(chunks)

    return assistant, chunks


def ask_question(assistant, question):
    """Ask a question and return the assistant response."""

    if not question or not question.strip():
        raise ValueError("Question cannot be empty")

    result = assistant.invoke({
        "question": question
    })

    return result