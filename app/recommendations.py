"""
Book recommendation service using content-based similarity.
Provides "Similar Books" functionality using semantic embeddings.
"""

import os
import json
from typing import Optional
from rdflib import Graph

from app.embeddings import get_embeddings_service


class RecommendationService:
    """Service for generating book recommendations."""

    def __init__(self):
        self.similarity_graph: dict[str, list[tuple[str, float]]] = {}
        self.book_data: dict[str, dict] = {}
        self.initialized = False

    def initialize(self, g: Graph, books: list[dict]) -> None:
        """
        Initialize the recommendation service with book data.

        Args:
            g: RDF graph containing book data
            books: List of book dictionaries
        """
        if self.initialized:
            return

        # Store book data for quick lookup
        for book in books:
            uri = book.get('uri', '')
            if uri:
                self.book_data[uri] = book

        # Initialize embeddings service
        embeddings_service = get_embeddings_service()
        embeddings_service.initialize(books)

        # Pre-compute similarity graph
        print("Computing book similarity graph...")
        for book in books:
            uri = book.get('uri', '')
            if uri:
                similar = embeddings_service.get_similar_books(uri, top_k=5)
                self.similarity_graph[uri] = similar

        self.initialized = True
        print(f"Recommendation service initialized with {len(books)} books")

    def get_similar_books(self, book_uri: str, limit: int = 5) -> list[dict]:
        """
        Get similar books for a given book.

        Args:
            book_uri: URI of the book
            limit: Maximum number of similar books to return

        Returns:
            List of similar book dictionaries with similarity scores
        """
        if not self.initialized:
            return []

        similar_uris = self.similarity_graph.get(book_uri, [])[:limit]

        results = []
        for uri, score in similar_uris:
            book = self.book_data.get(uri)
            if book:
                book_copy = book.copy()
                book_copy['similarity_score'] = round(score * 100, 1)  # Convert to percentage
                results.append(book_copy)

        return results

    def get_recommendations_for_query(self, query_results: list[dict], limit: int = 6) -> list[dict]:
        """
        Get recommendations based on search results.
        Returns books similar to top search results.

        Args:
            query_results: List of books from search results
            limit: Maximum number of recommendations

        Returns:
            List of recommended books not in original results
        """
        if not self.initialized or not query_results:
            return []

        # Get URIs of original results
        result_uris = {book.get('uri') for book in query_results}

        # Collect similar books from top results
        recommendation_scores: dict[str, float] = {}

        for book in query_results[:3]:  # Use top 3 results
            uri = book.get('uri', '')
            similar = self.similarity_graph.get(uri, [])

            for similar_uri, score in similar:
                if similar_uri not in result_uris:
                    if similar_uri in recommendation_scores:
                        recommendation_scores[similar_uri] = max(recommendation_scores[similar_uri], score)
                    else:
                        recommendation_scores[similar_uri] = score

        # Sort by score and return top recommendations
        sorted_recs = sorted(recommendation_scores.items(), key=lambda x: x[1], reverse=True)[:limit]

        results = []
        for uri, score in sorted_recs:
            book = self.book_data.get(uri)
            if book:
                book_copy = book.copy()
                book_copy['similarity_score'] = round(score * 100, 1)
                results.append(book_copy)

        return results


# Global singleton instance
_recommendation_service: Optional[RecommendationService] = None


def get_recommendation_service() -> RecommendationService:
    """Get the global recommendation service instance."""
    global _recommendation_service
    if _recommendation_service is None:
        _recommendation_service = RecommendationService()
    return _recommendation_service
