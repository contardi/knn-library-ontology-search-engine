"""
Embedding service for semantic search using sentence-transformers.
Provides semantic similarity matching between queries and book content.
"""

import os
import numpy as np
from typing import Optional

# Lazy import to avoid loading model until needed
_model = None
_embeddings_cache = {}


def get_model():
    """Lazy load the sentence transformer model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        # Using a lightweight model (22MB, 384-dim, fast inference)
        _model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Loaded sentence-transformer model: all-MiniLM-L6-v2")
    return _model


def get_cache_path() -> str:
    """Get the path for the embeddings cache file."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "data", "embeddings_cache.npz")


def compute_embedding(text: str) -> np.ndarray:
    """Compute embedding for a single text."""
    model = get_model()
    return model.encode(text, convert_to_numpy=True)


def compute_embeddings_batch(texts: list[str]) -> np.ndarray:
    """Compute embeddings for a batch of texts."""
    model = get_model()
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot_product / (norm1 * norm2))


def cosine_similarity_batch(query_vec: np.ndarray, book_vecs: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between a query and multiple book vectors."""
    # Normalize query vector
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    # Normalize book vectors (along axis 1)
    book_norms = book_vecs / (np.linalg.norm(book_vecs, axis=1, keepdims=True) + 1e-10)
    # Compute dot products
    return np.dot(book_norms, query_norm)


class BookEmbeddingsService:
    """Service for managing book embeddings and semantic search."""

    def __init__(self):
        self.book_embeddings: Optional[np.ndarray] = None
        self.book_ids: list[str] = []
        self.book_texts: list[str] = []
        self.initialized = False

    def initialize(self, books: list[dict]) -> None:
        """
        Initialize embeddings for all books.

        Args:
            books: List of book dictionaries with 'uri', 'title', 'description', 'author', 'genre' keys
        """
        if self.initialized and len(self.book_ids) == len(books):
            return

        cache_path = get_cache_path()
        book_ids = [book.get('uri', '') for book in books]

        # Try to load from cache
        if os.path.exists(cache_path):
            try:
                cache_data = np.load(cache_path, allow_pickle=True)
                cached_ids = list(cache_data['book_ids'])
                cached_embeddings = cache_data['embeddings']
                cached_texts = list(cache_data['texts'])

                # Check if cache matches current books
                if cached_ids == book_ids:
                    self.book_ids = cached_ids
                    self.book_embeddings = cached_embeddings
                    self.book_texts = cached_texts
                    self.initialized = True
                    print(f"Loaded embeddings cache for {len(self.book_ids)} books")
                    return
            except Exception as e:
                print(f"Cache load failed: {e}")

        # Compute embeddings for all books
        print(f"Computing embeddings for {len(books)} books...")

        self.book_ids = book_ids
        self.book_texts = []

        for book in books:
            # Combine title, description, author, and genre for richer embedding
            text_parts = []
            if book.get('title'):
                text_parts.append(book['title'])
            if book.get('author'):
                text_parts.append(f"by {book['author']}")
            if book.get('genre'):
                text_parts.append(book['genre'])
            if book.get('description'):
                # Truncate description to avoid very long texts
                desc = book['description'][:500] if len(book.get('description', '')) > 500 else book.get('description', '')
                text_parts.append(desc)

            combined_text = " | ".join(text_parts) if text_parts else book.get('title', '')
            self.book_texts.append(combined_text)

        # Batch compute embeddings
        self.book_embeddings = compute_embeddings_batch(self.book_texts)
        self.initialized = True

        # Save to cache
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.savez(
                cache_path,
                book_ids=np.array(self.book_ids, dtype=object),
                embeddings=self.book_embeddings,
                texts=np.array(self.book_texts, dtype=object)
            )
            print(f"Saved embeddings cache for {len(self.book_ids)} books")
        except Exception as e:
            print(f"Failed to save cache: {e}")

    def semantic_search(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        """
        Find books semantically similar to the query.

        Args:
            query: Search query text
            top_k: Maximum number of results to return

        Returns:
            List of (book_uri, similarity_score) tuples, sorted by score descending
        """
        if not self.initialized or self.book_embeddings is None:
            return []

        # Compute query embedding
        query_embedding = compute_embedding(query)

        # Compute similarities with all books
        similarities = cosine_similarity_batch(query_embedding, self.book_embeddings)

        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Minimum threshold
                results.append((self.book_ids[idx], float(similarities[idx])))

        return results

    def get_similar_books(self, book_uri: str, top_k: int = 5) -> list[tuple[str, float]]:
        """
        Find books similar to a given book.

        Args:
            book_uri: URI of the book to find similar books for
            top_k: Number of similar books to return

        Returns:
            List of (book_uri, similarity_score) tuples
        """
        if not self.initialized or self.book_embeddings is None:
            return []

        try:
            book_idx = self.book_ids.index(book_uri)
        except ValueError:
            return []

        book_embedding = self.book_embeddings[book_idx]
        similarities = cosine_similarity_batch(book_embedding, self.book_embeddings)

        # Get top-k+1 results (excluding the book itself)
        top_indices = np.argsort(similarities)[::-1][:top_k + 1]

        results = []
        for idx in top_indices:
            if idx != book_idx and similarities[idx] > 0.3:  # Exclude self and low similarity
                results.append((self.book_ids[idx], float(similarities[idx])))

        return results[:top_k]


# Global singleton instance
_embeddings_service: Optional[BookEmbeddingsService] = None


def get_embeddings_service() -> BookEmbeddingsService:
    """Get the global embeddings service instance."""
    global _embeddings_service
    if _embeddings_service is None:
        _embeddings_service = BookEmbeddingsService()
    return _embeddings_service
