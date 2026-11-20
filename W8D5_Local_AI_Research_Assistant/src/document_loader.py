"""Load and split research documents."""

from pathlib import Path
from pypdf import PdfReader


def load_documents(data_dir):
    """Load TXT and PDF files from the data directory."""

    data_path = Path(data_dir)
    documents = []

    data_path.mkdir(parents=True, exist_ok=True)

    for file_path in data_path.iterdir():

        if file_path.suffix.lower() == ".txt":
            text = file_path.read_text(encoding="utf-8")

        elif file_path.suffix.lower() == ".pdf":
            reader = PdfReader(str(file_path))
            text = "\n".join(
                page.extract_text() or ""
                for page in reader.pages
            )

        else:
            continue

        if text.strip():
            documents.append({
                "source": file_path.name,
                "content": text
            })

    return documents


def split_documents(documents, chunk_size=700, overlap=100):
    """Split documents into smaller overlapping chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []

    for document in documents:

        text = document["content"]
        start = 0

        while start < len(text):

            end = start + chunk_size
            content = text[start:end].strip()

            if content:
                chunks.append({
                    "source": document["source"],
                    "content": content
                })

            start += chunk_size - overlap

    return chunks