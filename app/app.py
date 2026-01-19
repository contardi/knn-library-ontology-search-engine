"""
Flask web application for the Library Search Engine.
"""

import os
from flask import Flask, render_template, request, jsonify
from rdflib import Graph

from app.queries import (
    search_books,
    get_all_books,
    get_books_by_genre,
    get_books_by_age_group,
    get_statistics,
    initialize_ai_services,
    get_recommendations_for_results,
    GENRE_NORMALIZATION,
    AGE_GROUPS,
)

app = Flask(__name__)

# Load the ontology
ONTOLOGY_PATH = os.environ.get(
    "ONTOLOGY_PATH",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "ontology", "library_populated.ttl")
)

g = None


def get_graph():
    """Load and cache the RDF graph, initialize AI services."""
    global g
    if g is None:
        g = Graph()
        g.parse(ONTOLOGY_PATH, format="turtle")
        print(f"Loaded ontology with {len(g)} triples")
        # Initialize AI services (embeddings, recommendations, TF-IDF)
        initialize_ai_services(g)
    return g


@app.route("/")
def index():
    """Home page with search interface."""
    graph = get_graph()
    stats = get_statistics(graph)

    # Get genre list for dropdown
    genres = sorted(GENRE_NORMALIZATION.keys())

    # Get age groups for dropdown
    age_groups = [
        ("EarlyChildhood", "Early Childhood (0-5)"),
        ("Children", "Children (6-8)"),
        ("MiddleGrade", "Middle Grade (9-12)"),
        ("YoungAdult", "Young Adult (13-17)"),
        ("Adult", "Adult (18+)"),
    ]

    return render_template(
        "index.html",
        stats=stats,
        genres=genres,
        age_groups=age_groups,
    )


@app.route("/search")
def search():
    """Search books using natural language or filters."""
    graph = get_graph()
    query = request.args.get("q", "").strip()
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 12, type=int)

    # Ensure valid pagination values
    page = max(1, page)
    per_page = min(max(1, per_page), 50)  # Limit per_page to 50

    if not query:
        return render_template(
            "results.html",
            query="",
            results=[],
            params={},
            pagination=None,
            suggestions=[],
            message="Please enter a search query.",
        )

    search_result = search_books(graph, query, page=page, per_page=per_page)

    message = None
    if not search_result["results"] and not search_result["suggestions"]:
        message = "No books found matching your search criteria."
    elif not search_result["results"] and search_result["suggestions"]:
        message = "No exact matches found."

    # Get recommendations based on search results
    recommendations = []
    if search_result["results"]:
        recommendations = get_recommendations_for_results(search_result["results"], limit=6)

    return render_template(
        "results.html",
        query=query,
        results=search_result["results"],
        params=search_result["params"],
        pagination=search_result["pagination"],
        suggestions=search_result["suggestions"],
        recommendations=recommendations,
        search_mode=search_result.get("search_mode", "text"),
        message=message,
    )


@app.route("/browse/genre/<genre>")
def browse_genre(genre):
    """Browse books by genre."""
    graph = get_graph()
    results = get_books_by_genre(graph, genre)

    return render_template(
        "results.html",
        query=f"Genre: {genre}",
        results=results,
        params={"genre": genre},
        pagination=None,
        suggestions=[],
        message=None if results else f"No books found in the {genre} genre.",
    )


@app.route("/browse/age/<age_group>")
def browse_age(age_group):
    """Browse books by age group."""
    graph = get_graph()
    results = get_books_by_age_group(graph, age_group)

    # Get display name for age group
    age_display = {
        "EarlyChildhood": "Early Childhood (0-5)",
        "Children": "Children (6-8)",
        "MiddleGrade": "Middle Grade (9-12)",
        "YoungAdult": "Young Adult (13-17)",
        "Adult": "Adult (18+)",
    }.get(age_group, age_group)

    return render_template(
        "results.html",
        query=f"Age Group: {age_display}",
        results=results,
        params={"age_group": age_group},
        pagination=None,
        suggestions=[],
        message=None if results else f"No books found for {age_display}.",
    )


@app.route("/browse/all")
def browse_all():
    """Browse all books."""
    graph = get_graph()
    results = get_all_books(graph, limit=100)

    return render_template(
        "results.html",
        query="All Books",
        results=results,
        params={},
        pagination=None,
        suggestions=[],
        message=None if results else "No books found in the library.",
    )


@app.route("/api/search")
def api_search():
    """API endpoint for search (returns JSON)."""
    graph = get_graph()
    query = request.args.get("q", "").strip()
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 12, type=int)

    if not query:
        return jsonify({"error": "No query provided", "results": []})

    search_result = search_books(graph, query, page=page, per_page=per_page)

    return jsonify({
        "query": query,
        "params": search_result["params"],
        "count": search_result["pagination"]["total_count"],
        "results": search_result["results"],
        "pagination": search_result["pagination"],
        "suggestions": search_result["suggestions"],
    })


@app.route("/api/stats")
def api_stats():
    """API endpoint for library statistics."""
    graph = get_graph()
    stats = get_statistics(graph)
    return jsonify(stats)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
