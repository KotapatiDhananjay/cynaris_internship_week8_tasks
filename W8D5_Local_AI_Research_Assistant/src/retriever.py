"""TF-IDF based research document retrieval."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class DocumentRetriever:
    """Retrieve relevant document chunks using TF-IDF."""

    def __init__(self, chunks):

        if not chunks:
            raise ValueError("No document chunks available")

        self.chunks = chunks

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )

        self.document_vectors = self.vectorizer.fit_transform([
            chunk["content"] for chunk in chunks
        ])

    def search(self, query, top_k=3):
        """Return the most relevant document chunks."""

        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_vector = self.vectorizer.transform([query])

        scores = cosine_similarity(
            query_vector,
            self.document_vectors
        ).flatten()

        ranked_indices = scores.argsort()[::-1][:top_k]

        results = []

        for index in ranked_indices:

            if scores[index] <= 0:
                continue

            results.append({
                "source": self.chunks[index]["source"],
                "content": self.chunks[index]["content"],
                "score": float(scores[index])
            })

        return results