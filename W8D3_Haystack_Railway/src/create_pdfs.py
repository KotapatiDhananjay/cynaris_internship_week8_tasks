import os
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
os.makedirs(DATA, exist_ok=True)

documents = {
    "document1.pdf": (
        "Artificial Intelligence and Machine Learning",
        "Artificial Intelligence enables machines to perform tasks requiring human intelligence. "
        "Machine Learning is a branch of AI that learns patterns from data. "
        "Supervised learning uses labelled data. Unsupervised learning discovers hidden patterns. "
        "Reinforcement learning learns through rewards and penalties."
    ),
    "document2.pdf": (
        "Natural Language Processing",
        "Natural Language Processing enables computers to understand human language. "
        "Tokenization divides text into smaller units. Stemming reduces words to their root forms. "
        "Sentiment analysis identifies opinions and emotions. Named Entity Recognition identifies "
        "names, locations and organizations."
    ),
    "document3.pdf": (
        "Retrieval Augmented Generation",
        "RAG stands for Retrieval Augmented Generation. It retrieves relevant documents and "
        "provides their context to a language model. RAG improves factual grounding. "
        "A RAG system contains document loading, chunking, embedding, retrieval and generation."
    ),
    "document4.pdf": (
        "Vector Databases and Embeddings",
        "Vector databases store numerical representations called embeddings. "
        "Embeddings represent semantic meaning. Similarity search finds related vectors. "
        "ChromaDB and FAISS are examples of vector databases. Dense retrieval uses embeddings "
        "to find semantically related documents."
    ),
    "document5.pdf": (
        "Haystack and Information Retrieval",
        "Haystack is an open-source framework for building search and RAG applications. "
        "DocumentStore stores documents. BM25Retriever performs keyword-based retrieval. "
        "Dense retrievers use vector embeddings. Precision measures the fraction of retrieved "
        "documents that are relevant. Recall measures the fraction of relevant documents retrieved."
    )
}

for filename, (title, content) in documents.items():
    path = os.path.join(DATA, filename)
    pdf = canvas.Canvas(path, pagesize=A4)
    width, height = A4

    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(50, height - 70, title)

    pdf.setFont("Helvetica", 12)
    y = height - 120

    words = content.split()
    lines = []
    line = ""

    for word in words:
        if len(line + " " + word) > 85:
            lines.append(line)
            line = word
        else:
            line = (line + " " + word).strip()

    if line:
        lines.append(line)

    for line in lines:
        pdf.drawString(50, y, line)
        y -= 25

    pdf.save()
    print("Created:", filename)

print("All 5 PDF documents created successfully.")