"""
SPARQL query functions and natural language parser for the Library Search Engine.
Enhanced with semantic search, TF-IDF scoring, and improved fuzzy matching.
"""

import re
from difflib import SequenceMatcher
from typing import Optional
from rdflib import Graph, Namespace
from rdflib.plugins.sparql import prepareQuery

# Import rapidfuzz for better fuzzy matching
try:
    from rapidfuzz import fuzz
    from rapidfuzz.distance import Levenshtein
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    print("Warning: rapidfuzz not available, using basic fuzzy matching")

# Import sklearn for TF-IDF (optional enhancement)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available, TF-IDF features disabled")

LIB = Namespace("http://example.org/library#")

# Global TF-IDF vectorizer (initialized lazily)
_tfidf_vectorizer: Optional['TfidfVectorizer'] = None
_tfidf_matrix = None
_tfidf_book_uris: list[str] = []

# Stop words to ignore in searches
STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
    'books', 'book', 'for', 'with', 'by', 'of', 'in', 'to', 'on', 'at',
    'it', 'its', 'this', 'that', 'from', 'about', 'into', 'be', 'been',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'can', 'as', 'if', 'than', 'then'
}


def tokenize_query(query: str) -> list[str]:
    """
    Tokenize a query into individual words, removing stop words and short words.
    """
    words = re.findall(r'\b\w+\b', query.lower())
    return [w for w in words if w not in STOP_WORDS and len(w) > 2]


def calculate_relevance_score(book: dict, query_terms: list[str]) -> int:
    """
    Calculate a relevance score for a book based on query terms.
    Higher scores indicate better matches.
    """
    score = 0
    title_lower = book.get("title", "").lower()
    author_lower = book.get("author", "").lower()
    description_lower = book.get("description", "").lower()

    for term in query_terms:
        term_lower = term.lower()
        # Title exact match (title equals the term)
        if title_lower == term_lower:
            score += 100
        # Title contains the term
        elif term_lower in title_lower:
            score += 50

        # Author match
        if term_lower in author_lower:
            score += 40

        # Description match
        if term_lower in description_lower:
            score += 10

    return score


def find_similar(query: str, candidates: list[str], threshold: float = 0.4) -> list[tuple[str, float]]:
    """
    Find similar strings to the query using enhanced fuzzy matching.
    Uses rapidfuzz for better typo tolerance with multiple matching strategies:
    - Levenshtein distance (edit distance)
    - Partial ratio (substring matching)
    - Token set ratio (word-order independent)

    Returns list of (candidate, similarity_ratio) tuples sorted by similarity.
    """
    matches = []
    query_lower = query.lower().strip()

    # Adjust threshold for short queries (more lenient)
    adjusted_threshold = threshold
    if len(query_lower) <= 6:
        adjusted_threshold = 0.3  # More lenient for short queries like "Odissey"

    for candidate in candidates:
        candidate_lower = candidate.lower()

        if RAPIDFUZZ_AVAILABLE:
            # Use multiple matching strategies and take the best
            scores = []

            # 1. Simple ratio (similar to SequenceMatcher)
            simple_ratio = fuzz.ratio(query_lower, candidate_lower) / 100.0
            scores.append(simple_ratio)

            # 2. Partial ratio (good for substring matches)
            partial_ratio = fuzz.partial_ratio(query_lower, candidate_lower) / 100.0
            scores.append(partial_ratio * 0.9)  # Slight penalty for partial

            # 3. Token set ratio (word-order independent)
            token_ratio = fuzz.token_set_ratio(query_lower, candidate_lower) / 100.0
            scores.append(token_ratio * 0.95)

            # 4. Levenshtein-based similarity
            max_len = max(len(query_lower), len(candidate_lower))
            if max_len > 0:
                edit_distance = Levenshtein.distance(query_lower, candidate_lower)
                levenshtein_ratio = 1 - (edit_distance / max_len)
                scores.append(levenshtein_ratio)

            best_score = max(scores)
        else:
            # Fallback to SequenceMatcher
            best_score = SequenceMatcher(None, query_lower, candidate_lower).ratio()

        if best_score >= adjusted_threshold:
            matches.append((candidate, best_score))

    # Sort by score and return top matches
    return sorted(matches, key=lambda x: x[1], reverse=True)[:8]


def phonetic_match(word1: str, word2: str) -> bool:
    """
    Simple phonetic matching for common spelling variations.
    Handles cases like 'ph' vs 'f', double letters, etc.
    """
    # Normalize both words
    def normalize(w):
        w = w.lower()
        # Common phonetic substitutions
        w = w.replace('ph', 'f')
        w = w.replace('ck', 'k')
        w = w.replace('ee', 'e')
        w = w.replace('oo', 'o')
        w = w.replace('ss', 's')
        w = w.replace('tt', 't')
        w = w.replace('ll', 'l')
        # Remove duplicate consonants
        w = re.sub(r'(.)\1+', r'\1', w)
        return w

    return normalize(word1) == normalize(word2)


def highlight_matches(text: str, terms: list[str]) -> str:
    """
    Wrap matched terms in <mark> tags for highlighting.
    """
    if not text or not terms:
        return text

    result = text
    for term in terms:
        if term:
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            result = pattern.sub(lambda m: f'<mark>{m.group()}</mark>', result)

    return result


def get_all_titles_and_authors(g: Graph) -> tuple[list[str], list[str]]:
    """
    Get all book titles and author names for fuzzy matching suggestions.
    """
    titles_query = """
    PREFIX lib: <http://example.org/library#>
    SELECT DISTINCT ?title WHERE { ?book lib:hasTitle ?title . }
    """

    authors_query = """
    PREFIX lib: <http://example.org/library#>
    SELECT DISTINCT ?authorName WHERE { ?author lib:hasAuthorName ?authorName . }
    """

    titles = [str(row.title) for row in g.query(titles_query)]
    authors = [str(row.authorName) for row in g.query(authors_query)]

    return titles, authors

def initialize_tfidf(books: list[dict]) -> None:
    """Initialize TF-IDF vectorizer with book corpus."""
    global _tfidf_vectorizer, _tfidf_matrix, _tfidf_book_uris

    if not SKLEARN_AVAILABLE or _tfidf_vectorizer is not None:
        return

    # Build corpus from books
    corpus = []
    _tfidf_book_uris = []

    for book in books:
        text_parts = []
        if book.get('title'):
            text_parts.append(book['title'])
        if book.get('author'):
            text_parts.append(book['author'])
        if book.get('genre'):
            text_parts.append(book['genre'])
        if book.get('description'):
            text_parts.append(book['description'])

        combined = ' '.join(text_parts)
        corpus.append(combined)
        _tfidf_book_uris.append(book.get('uri', ''))

    if corpus:
        _tfidf_vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2)  # Include bigrams
        )
        _tfidf_matrix = _tfidf_vectorizer.fit_transform(corpus)
        print(f"TF-IDF initialized with {len(corpus)} documents")


def get_tfidf_scores(query: str, book_uris: list[str]) -> dict[str, float]:
    """
    Get TF-IDF similarity scores for books matching a query.

    Args:
        query: Search query text
        book_uris: List of book URIs to score

    Returns:
        Dict mapping book URI to TF-IDF score
    """
    global _tfidf_vectorizer, _tfidf_matrix, _tfidf_book_uris

    if not SKLEARN_AVAILABLE or _tfidf_vectorizer is None or _tfidf_matrix is None:
        return {}

    try:
        # Transform query
        query_vec = _tfidf_vectorizer.transform([query])

        # Compute cosine similarity with all documents
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(query_vec, _tfidf_matrix).flatten()

        # Map to book URIs
        uri_to_score = {}
        for i, uri in enumerate(_tfidf_book_uris):
            if uri in book_uris:
                uri_to_score[uri] = float(similarities[i])

        return uri_to_score
    except Exception as e:
        print(f"TF-IDF scoring error: {e}")
        return {}


def calculate_combined_score(
    book: dict,
    query_terms: list[str],
    query_text: str,
    semantic_scores: dict[str, float],
    tfidf_scores: dict[str, float]
) -> tuple[float, dict]:
    """
    Calculate a combined relevance score using multiple signals.

    Returns:
        Tuple of (final_score, score_breakdown)
    """
    uri = book.get('uri', '')

    # Word-based score (existing method)
    word_score = calculate_relevance_score(book, query_terms)
    # Normalize to 0-1 range (max possible is ~200 for exact title + author + description)
    word_score_normalized = min(word_score / 150.0, 1.0)

    # Semantic score (from embeddings)
    semantic_score = semantic_scores.get(uri, 0.0)

    # TF-IDF score
    tfidf_score = tfidf_scores.get(uri, 0.0)

    # Combine scores with weights
    # If we have semantic scores, weight them more heavily
    if semantic_scores:
        final_score = (
            0.35 * word_score_normalized +
            0.45 * semantic_score +
            0.20 * tfidf_score
        )
    else:
        # Fallback when semantic search is not available
        final_score = (
            0.70 * word_score_normalized +
            0.30 * tfidf_score
        )

    # Bonus for exact title match
    title_lower = book.get('title', '').lower()
    query_lower = query_text.lower()
    if query_lower in title_lower or title_lower in query_lower:
        final_score += 0.15

    # Cap at 1.0
    final_score = min(final_score, 1.0)

    # Score breakdown for debugging/display
    breakdown = {
        'word': round(word_score_normalized * 100, 1),
        'semantic': round(semantic_score * 100, 1),
        'tfidf': round(tfidf_score * 100, 1),
        'final': round(final_score * 100, 1)
    }

    return final_score, breakdown


# Genre mappings for natural language parsing
FICTION_GENRES = [
    "mystery", "romance", "science fiction", "sci-fi", "fantasy",
    "thriller", "historical fiction", "horror", "adventure"
]

NONFICTION_GENRES = [
    "science", "history", "biography", "technology", "philosophy", "arts", "nature", "sports"
]

ALL_GENRES = FICTION_GENRES + NONFICTION_GENRES

# Genre name normalization
GENRE_NORMALIZATION = {
    "sci-fi": "ScienceFiction",
    "science fiction": "ScienceFiction",
    "historical fiction": "HistoricalFiction",
    "mystery": "Mystery",
    "romance": "Romance",
    "fantasy": "Fantasy",
    "thriller": "Thriller",
    "horror": "Horror",
    "adventure": "Adventure",
    "science": "Science",
    "history": "History",
    "biography": "BiographyGenre",
    "technology": "Technology",
    "philosophy": "Philosophy",
    "arts": "Arts",
    "nature": "Nature",
    "sports": "Sports",
}

# Age group mappings
AGE_GROUPS = {
    "early childhood": ("EarlyChildhood", 0, 5),
    "toddler": ("EarlyChildhood", 0, 5),
    "preschool": ("EarlyChildhood", 0, 5),
    "children": ("Children", 6, 8),
    "kids": ("Children", 6, 8),
    "middle grade": ("MiddleGrade", 9, 12),
    "tween": ("MiddleGrade", 9, 12),
    "young adult": ("YoungAdult", 13, 17),
    "ya": ("YoungAdult", 13, 17),
    "teen": ("YoungAdult", 13, 17),
    "adult": ("Adult", 18, 99),
}


def parse_natural_language_query(query: str) -> dict:
    """
    Parse a natural language query to extract search parameters.

    Examples:
    - "Fantasy books for children with 8 years old"
    - "Science fiction for adults"
    - "Mystery books"
    - "Books by Stephen King"
    """
    query_lower = query.lower().strip()

    result = {
        "genre": None,
        "genre_instance": None,
        "age": None,
        "age_group": None,
        "category": None,  # fiction or non-fiction
        "author": None,
        "title_keywords": [],
        "status": None,
    }

    # Extract age (e.g., "8 years old", "for 8 year olds", "age 8")
    age_patterns = [
        r"(\d+)\s*years?\s*old",
        r"for\s*(\d+)\s*year",
        r"age\s*(\d+)",
        r"(\d+)\s*yo\b",
    ]
    for pattern in age_patterns:
        match = re.search(pattern, query_lower)
        if match:
            result["age"] = int(match.group(1))
            break

    # Extract genre
    for genre_key, genre_value in GENRE_NORMALIZATION.items():
        if genre_key in query_lower:
            result["genre"] = genre_value
            result["genre_instance"] = f"{genre_value}Instance"
            # Determine category based on genre
            if genre_key in FICTION_GENRES:
                result["category"] = "fiction"
            else:
                result["category"] = "non-fiction"
            break

    # Extract age group by name
    for age_key, (age_group, min_age, max_age) in AGE_GROUPS.items():
        if age_key in query_lower:
            result["age_group"] = age_group
            # If no specific age, use midpoint of range
            if result["age"] is None:
                result["age"] = (min_age + max_age) // 2
            break

    # If we have an age but no age group, determine the age group
    if result["age"] is not None and result["age_group"] is None:
        age = result["age"]
        if age <= 5:
            result["age_group"] = "EarlyChildhood"
        elif age <= 8:
            result["age_group"] = "Children"
        elif age <= 12:
            result["age_group"] = "MiddleGrade"
        elif age <= 17:
            result["age_group"] = "YoungAdult"
        else:
            result["age_group"] = "Adult"

    # Extract author (e.g., "by Stephen King", "author Stephen King")
    author_patterns = [
        r"by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        r"author\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
    ]
    for pattern in author_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            result["author"] = match.group(1)
            break

    # Extract availability status
    if "available" in query_lower:
        result["status"] = "Available"
    elif "checked out" in query_lower or "borrowed" in query_lower:
        result["status"] = "CheckedOut"

    # Extract category if not already set
    if result["category"] is None:
        if "fiction" in query_lower and "non-fiction" not in query_lower and "nonfiction" not in query_lower:
            result["category"] = "fiction"
        elif "non-fiction" in query_lower or "nonfiction" in query_lower:
            result["category"] = "non-fiction"

    # If no specific parameters found, use entire query as title search
    if not any([result["genre"], result["age_group"], result["author"],
                result["category"], result["status"]]):
        result["title_keywords"] = query.strip()

    return result


def build_sparql_query(params: dict) -> str:
    """Build a SPARQL query from parsed parameters."""

    # Base query parts
    select_vars = ["?book", "?title", "?authorName", "?genreLabel", "?ageGroupLabel", "?year", "?description", "?status"]

    where_clauses = [
        "?book rdf:type ?bookType .",
        "?bookType rdfs:subClassOf* lib:Book .",
        "?book lib:hasTitle ?title .",
    ]

    optional_clauses = [
        "OPTIONAL { ?book lib:hasAuthor ?author . ?author lib:hasAuthorName ?authorName . }",
        "OPTIONAL { ?book lib:hasGenre ?genre . ?genre rdfs:label ?genreLabel . }",
        "OPTIONAL { ?book lib:forAgeGroup ?ageGroup . ?ageGroup lib:hasAgeGroupName ?ageGroupLabel . }",
        "OPTIONAL { ?book lib:hasPublicationYear ?year . }",
        "OPTIONAL { ?book lib:hasDescription ?description . }",
        "OPTIONAL { ?book lib:hasStatus ?statusObj . ?statusObj rdfs:label ?status . }",
    ]

    filters = []

    # Add genre filter
    if params.get("genre_instance"):
        where_clauses.append(f"?book lib:hasGenre lib:{params['genre_instance']} .")

    # Add age group filter
    if params.get("age_group"):
        where_clauses.append(f"?book lib:forAgeGroup lib:{params['age_group']} .")
    elif params.get("age") is not None:
        # Filter by age range
        age = params["age"]
        where_clauses.extend([
            "?book lib:forAgeGroup ?ageGroupFilter .",
            "?ageGroupFilter lib:hasMinAge ?minAge .",
            "?ageGroupFilter lib:hasMaxAge ?maxAge .",
        ])
        filters.append(f"FILTER(?minAge <= {age} && ?maxAge >= {age})")

    # Add author filter
    if params.get("author"):
        author = params["author"].replace("'", "\\'")
        where_clauses.append("?book lib:hasAuthor ?authorFilter .")
        where_clauses.append("?authorFilter lib:hasAuthorName ?authorFilterName .")
        filters.append(f"FILTER(CONTAINS(LCASE(?authorFilterName), LCASE('{author}')))")

    # Add category filter (fiction/non-fiction)
    if params.get("category") == "fiction":
        where_clauses.append("?book rdf:type/rdfs:subClassOf* lib:FictionBook .")
    elif params.get("category") == "non-fiction":
        where_clauses.append("?book rdf:type/rdfs:subClassOf* lib:NonFictionBook .")

    # Add status filter
    if params.get("status"):
        where_clauses.append(f"?book lib:hasStatus lib:{params['status']} .")

    # Add multi-field keyword filter (title, description, author)
    if params.get("title_keywords"):
        search_text = params["title_keywords"]
        # Tokenize the search text and filter stop words
        tokens = tokenize_query(search_text)

        if tokens:
            # Build OR conditions for each token across multiple fields
            token_filters = []
            for token in tokens:
                token_escaped = token.replace("'", "\\'")
                token_filter = f"""(
                    CONTAINS(LCASE(?title), LCASE('{token_escaped}')) ||
                    CONTAINS(LCASE(COALESCE(?description, '')), LCASE('{token_escaped}')) ||
                    CONTAINS(LCASE(COALESCE(?authorName, '')), LCASE('{token_escaped}'))
                )"""
                token_filters.append(token_filter)

            # Combine token filters with OR (any token can match)
            if token_filters:
                combined_filter = " || ".join(token_filters)
                filters.append(f"FILTER({combined_filter})")
        else:
            # If no tokens after filtering, use the original search text
            title_search = search_text.replace("'", "\\'")
            filters.append(f"""FILTER(
                CONTAINS(LCASE(?title), LCASE('{title_search}')) ||
                CONTAINS(LCASE(COALESCE(?description, '')), LCASE('{title_search}')) ||
                CONTAINS(LCASE(COALESCE(?authorName, '')), LCASE('{title_search}'))
            )""")

    # Build the query
    query = f"""
    PREFIX lib: <http://example.org/library#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>

    SELECT DISTINCT {' '.join(select_vars)}
    WHERE {{
        {' '.join(where_clauses)}
        {' '.join(optional_clauses)}
        {' '.join(filters)}
    }}
    ORDER BY ?title
    """

    return query


def search_books(g: Graph, query_text: str, page: int = 1, per_page: int = 12, use_semantic: bool = True) -> dict:
    """
    Search books using natural language query with optional semantic enhancement.
    Returns a dict with results, pagination info, suggestions, and search metadata.

    Args:
        g: RDF graph
        query_text: Natural language search query
        page: Page number for pagination
        per_page: Results per page
        use_semantic: Whether to use semantic search (embeddings)
    """
    params = parse_natural_language_query(query_text)
    sparql_query = build_sparql_query(params)

    # Get query terms for relevance scoring and highlighting
    query_terms = tokenize_query(query_text)
    if not query_terms:
        # Fall back to the original query if no tokens after filtering
        query_terms = [query_text.strip()]

    # Execute SPARQL query to get initial results
    sparql_results = []
    for row in g.query(sparql_query):
        book = {
            "uri": str(row.book),
            "title": str(row.title) if row.title else "",
            "author": str(row.authorName) if row.authorName else "Unknown",
            "genre": str(row.genreLabel) if row.genreLabel else "",
            "age_group": str(row.ageGroupLabel) if row.ageGroupLabel else "",
            "year": int(row.year) if row.year else None,
            "description": str(row.description) if row.description else "",
            "status": str(row.status) if row.status else "Unknown",
        }
        sparql_results.append(book)

    # Get semantic search results if enabled and query looks semantic
    semantic_scores: dict[str, float] = {}
    search_mode = "text"  # Track which search mode was used

    if use_semantic and params.get("title_keywords"):
        try:
            from app.embeddings import get_embeddings_service

            embeddings_service = get_embeddings_service()
            if embeddings_service.initialized:
                semantic_results = embeddings_service.semantic_search(query_text, top_k=100)
                semantic_scores = dict(semantic_results)
                search_mode = "semantic" if semantic_scores else "text"

                # If SPARQL returned no results but semantic search found matches,
                # fetch those books from the graph
                if not sparql_results and semantic_scores:
                    all_books = get_all_books(g, limit=200)
                    semantic_uris = set(semantic_scores.keys())
                    sparql_results = [b for b in all_books if b.get('uri') in semantic_uris]

        except ImportError:
            pass  # Embeddings not available
        except Exception as e:
            print(f"Semantic search error: {e}")

    # Get TF-IDF scores
    book_uris = [b.get('uri', '') for b in sparql_results]
    tfidf_scores = get_tfidf_scores(query_text, book_uris)

    # Calculate combined scores
    results = []
    for book in sparql_results:
        score, breakdown = calculate_combined_score(
            book, query_terms, query_text, semantic_scores, tfidf_scores
        )
        book["_score"] = score
        book["_score_breakdown"] = breakdown
        book["_match_percentage"] = int(breakdown['final'])
        results.append(book)

    # Sort by combined score (descending), then by title
    results.sort(key=lambda x: (-x["_score"], x["title"]))

    # Generate suggestions if no results found
    suggestions = []
    if not results:
        titles, authors = get_all_titles_and_authors(g)
        all_candidates = titles + authors

        # Find similar matches for the query
        similar_matches = find_similar(query_text, all_candidates, threshold=0.35)
        suggestions = [match[0] for match in similar_matches]

    # Calculate pagination
    total_count = len(results)
    total_pages = (total_count + per_page - 1) // per_page if total_count > 0 else 1
    start = (page - 1) * per_page
    end = start + per_page
    paginated_results = results[start:end]

    # Add highlighted versions of title and description
    for book in paginated_results:
        book["title_highlighted"] = highlight_matches(book["title"], query_terms)
        book["description_highlighted"] = highlight_matches(book["description"], query_terms)
        book["author_highlighted"] = highlight_matches(book["author"], query_terms)

    return {
        "results": paginated_results,
        "params": params,
        "query_terms": query_terms,
        "suggestions": suggestions,
        "search_mode": search_mode,
        "pagination": {
            "page": page,
            "per_page": per_page,
            "total_count": total_count,
            "total_pages": total_pages,
            "has_prev": page > 1,
            "has_next": page < total_pages,
        }
    }


def get_all_books(g: Graph, limit: int = 50) -> list:
    """Get all books with their details."""

    query = f"""
    PREFIX lib: <http://example.org/library#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT DISTINCT ?book ?title ?authorName ?genreLabel ?ageGroupLabel ?year ?description ?status
    WHERE {{
        ?book rdf:type ?bookType .
        ?bookType rdfs:subClassOf* lib:Book .
        ?book lib:hasTitle ?title .
        OPTIONAL {{ ?book lib:hasAuthor ?author . ?author lib:hasAuthorName ?authorName . }}
        OPTIONAL {{ ?book lib:hasGenre ?genre . ?genre rdfs:label ?genreLabel . }}
        OPTIONAL {{ ?book lib:forAgeGroup ?ageGroup . ?ageGroup lib:hasAgeGroupName ?ageGroupLabel . }}
        OPTIONAL {{ ?book lib:hasPublicationYear ?year . }}
        OPTIONAL {{ ?book lib:hasDescription ?description . }}
        OPTIONAL {{ ?book lib:hasStatus ?statusObj . ?statusObj rdfs:label ?status . }}
    }}
    ORDER BY ?title
    LIMIT {limit}
    """

    results = []
    for row in g.query(query):
        book = {
            "uri": str(row.book),
            "title": str(row.title) if row.title else "",
            "author": str(row.authorName) if row.authorName else "Unknown",
            "genre": str(row.genreLabel) if row.genreLabel else "",
            "age_group": str(row.ageGroupLabel) if row.ageGroupLabel else "",
            "year": int(row.year) if row.year else None,
            "description": str(row.description) if row.description else "",
            "status": str(row.status) if row.status else "Unknown",
        }
        results.append(book)

    return results


def get_books_by_genre(g: Graph, genre: str) -> list:
    """Get all books of a specific genre."""
    params = {"genre_instance": f"{genre}Instance"}
    sparql_query = build_sparql_query(params)
    results = []
    for row in g.query(sparql_query):
        book = {
            "uri": str(row.book),
            "title": str(row.title) if row.title else "",
            "author": str(row.authorName) if row.authorName else "Unknown",
            "genre": str(row.genreLabel) if row.genreLabel else "",
            "age_group": str(row.ageGroupLabel) if row.ageGroupLabel else "",
            "year": int(row.year) if row.year else None,
            "description": str(row.description) if row.description else "",
            "status": str(row.status) if row.status else "Unknown",
        }
        results.append(book)
    return results


def get_books_by_age_group(g: Graph, age_group: str) -> list:
    """Get all books for a specific age group."""
    params = {"age_group": age_group}
    sparql_query = build_sparql_query(params)
    results = []
    for row in g.query(sparql_query):
        book = {
            "uri": str(row.book),
            "title": str(row.title) if row.title else "",
            "author": str(row.authorName) if row.authorName else "Unknown",
            "genre": str(row.genreLabel) if row.genreLabel else "",
            "age_group": str(row.ageGroupLabel) if row.ageGroupLabel else "",
            "year": int(row.year) if row.year else None,
            "description": str(row.description) if row.description else "",
            "status": str(row.status) if row.status else "Unknown",
        }
        results.append(book)
    return results


def get_statistics(g: Graph) -> dict:
    """Get library statistics."""

    # Total books
    books_query = """
    PREFIX lib: <http://example.org/library#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT (COUNT(DISTINCT ?book) as ?bookCount)
    WHERE {
        ?book lib:hasTitle ?title .
    }
    """

    # Total authors
    authors_query = """
    PREFIX lib: <http://example.org/library#>
    SELECT (COUNT(DISTINCT ?author) as ?authorCount)
    WHERE {
        ?author lib:hasAuthorName ?name .
    }
    """

    # Available books
    available_query = """
    PREFIX lib: <http://example.org/library#>
    SELECT (COUNT(DISTINCT ?book) as ?availableCount)
    WHERE {
        ?book lib:hasStatus lib:Available .
    }
    """

    stats = {
        "total_books": 0,
        "total_authors": 0,
        "available_books": 0,
    }

    for row in g.query(books_query):
        stats["total_books"] = int(row.bookCount)

    for row in g.query(authors_query):
        stats["total_authors"] = int(row.authorCount)

    for row in g.query(available_query):
        stats["available_books"] = int(row.availableCount)

    return stats


def initialize_ai_services(g: Graph) -> None:
    """
    Initialize all AI services (embeddings, recommendations, TF-IDF).
    Should be called once when the application starts.
    """
    print("Initializing AI services...")

    # Get all books for initialization
    all_books = get_all_books(g, limit=200)

    # Initialize TF-IDF
    initialize_tfidf(all_books)

    # Initialize embeddings service
    try:
        from app.embeddings import get_embeddings_service
        embeddings_service = get_embeddings_service()
        embeddings_service.initialize(all_books)
    except ImportError as e:
        print(f"Embeddings service not available: {e}")
    except Exception as e:
        print(f"Error initializing embeddings: {e}")

    # Initialize recommendation service
    try:
        from app.recommendations import get_recommendation_service
        recommendation_service = get_recommendation_service()
        recommendation_service.initialize(g, all_books)
    except ImportError as e:
        print(f"Recommendation service not available: {e}")
    except Exception as e:
        print(f"Error initializing recommendations: {e}")

    print("AI services initialized")


def get_similar_books_for_result(book_uri: str, limit: int = 5) -> list[dict]:
    """
    Get similar books for a given book URI.
    Used to display "You might also like" recommendations.
    """
    try:
        from app.recommendations import get_recommendation_service
        recommendation_service = get_recommendation_service()
        return recommendation_service.get_similar_books(book_uri, limit)
    except Exception:
        return []


def get_recommendations_for_results(results: list[dict], limit: int = 6) -> list[dict]:
    """
    Get recommendations based on search results.
    Returns books similar to the top search results.
    """
    try:
        from app.recommendations import get_recommendation_service
        recommendation_service = get_recommendation_service()
        return recommendation_service.get_recommendations_for_query(results, limit)
    except Exception:
        return []
